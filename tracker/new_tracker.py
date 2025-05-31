# -------------------------------------------------
# Imports
# -------------------------------------------------
import os, glob, numpy as np
from PIL import Image
import torch, yaml
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Resize
import progressbar

from tracker.model.network import XMem
from inference.inference_core import InferenceCore
from tracker.util.mask_mapper import MaskMapper
from tracker.util.range_transform import im_normalization
from tools.painter import mask_painter
# ★ NEW: fast guided filter (edge-preserving, CRF-like)
from guided_filter_pytorch.guided_filter import GuidedFilter


class NewTracker:
    """
    XMem 기반 비디오 객체 추적기 + 경계 보존 Guided-Filter 후처리
    """
    # -------------------------------------------------
    # 초기화
    # -------------------------------------------------
    def __init__(self, xmem_checkpoint: str, device: str,
                 sam_model=None, model_type=None) -> None:
        """
        Args
        ----
        xmem_checkpoint : str
            XMem 가중치 경로
        device : str
            'cuda:0' 형태
        """
        # ── config 로드 ───────────────────────────────
        with open("tracker/config/config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # ── XMem 초기화 ───────────────────────────────
        network = XMem(config, xmem_checkpoint).to(device).eval()
        self.tracker = InferenceCore(network, config)

        # ── 데이터 변환 ───────────────────────────────
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        # ── Guided Filter (fast CRF-like smoothing) ──
        # radius=4 → 9×9 window, eps=1e-3 → smoothness 조정
        self.guided_filter = GuidedFilter(r=4, eps=1e-3)

        # ── 기타 멤버 변수 ───────────────────────────
        self.device = device
        self.mapper = MaskMapper()
        self.initialised = False

        # (SAM refinement 코드 생략)

    # -------------------------------------------------
    # 내부 헬퍼: 확률맵 Guided-Filter 정제
    # -------------------------------------------------
    @torch.no_grad()
    def _refine_probs(self, frame_tensor: torch.Tensor,
                      probs: torch.Tensor) -> torch.Tensor:
        """
        frame_tensor : (3,H,W)
        probs        : (C,H,W)
        """
        # ① guidance 이미지를 1채널(gray)로 변환  ---------------------------
        guide_gray = frame_tensor.mean(dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)

        # ② 클래스별 guided filtering  --------------------------------------
        probs_refined = torch.empty_like(probs)
        for c in range(probs.shape[0]):
            p = probs[c:c+1].unsqueeze(0)          # (1,1,H,W)
            p_ref = self.guided_filter(guide_gray, p)  # ASSERT 통과
            probs_refined[c] = p_ref.squeeze(0)

        # ③ softmax로 재정규화  --------------------------------------------
        probs_refined = F.softmax(probs_refined, dim=0)
        return probs_refined
    # -------------------------------------------------
    # public API: 프레임 추적
    # -------------------------------------------------
    @torch.no_grad()
    def track(self, frame: np.ndarray,
              first_frame_annotation: np.ndarray | None = None):
        """
        Args
        ----
        frame : (H,W,3) uint8 RGB
        first_frame_annotation : (H,W) uint8 [객체 ID 라벨맵] | None

        Returns
        -------
        final_mask      : (H,W) uint8 [라벨맵]
        final_mask_prob : (H,W) uint8 [동일]
        painted_image   : (H,W,3) uint8 [결과 시각화]
        """
        # ── 초기 프레임 처리 ───────────────────────────
        if first_frame_annotation is not None:     # 최초 프레임
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.as_tensor(mask, device=self.device)
            self.tracker.set_all_labels(list(self.mapper.remappings.values()))
        else:
            mask, labels = None, None

        # ── 네트워크 입력 준비 ───────────────────────
        frame_tensor = self.im_transform(frame).to(self.device)

        # ── XMem 스텝 ─────────────────────────────────
        probs, _ = self.tracker.step(frame_tensor, mask, labels)  # (C,H,W)

        # ── ★ Guided-Filter 후처리 ───────────────
        probs = self._refine_probs(frame_tensor, probs)

        # ── 라벨맵 생성 ─────────────────────────────
        out_mask = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)

        # remap (label → 원본 ID)
        final_mask = np.zeros_like(out_mask)
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        # ── 시각화 ──────────────────────────────────
        painted_image = frame.copy()
        for obj_id in range(1, final_mask.max() + 1):
            if not (final_mask == obj_id).any():
                continue
            painted_image = mask_painter(painted_image,
                                         (final_mask == obj_id).astype(np.uint8),
                                         mask_color=obj_id + 1)

        return final_mask, final_mask, painted_image

    # -------------------------------------------------
    # 기타 유틸리티 함수
    # -------------------------------------------------
    @torch.no_grad()
    def resize_mask(self, mask: torch.Tensor, size: int = 480):
        """원본 마스크를 비율 유지하여 size × ? 로 리사이즈"""
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return F.interpolate(mask, (int(h / min_hw * size),
                                    int(w / min_hw * size)),
                             mode="nearest")

    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()
        torch.cuda.empty_cache()


# -------------------------------------------------
# Example Usage (Main – 그대로 두어도 됨)
# -------------------------------------------------
if __name__ == "__main__":
    video_path_list = glob.glob(
        "/ssd1/gaomingqi/datasets/davis/JPEGImages/480p/horsejump-high/*.jpg")
    video_path_list.sort()
    frames = [np.array(Image.open(p).convert("RGB")) for p in video_path_list]
    frames = np.stack(frames, 0)

    first_frame_path = "/ssd1/gaomingqi/datasets/davis/Annotations/480p/horsejump-high/00000.png"
    first_frame_annotation = np.array(Image.open(first_frame_path).convert("P"))

    device = "cuda:2"
    XMEM_checkpoint = "/ssd1/gaomingqi/checkpoints/XMem-s012.pth"

    tracker = NewTracker(XMEM_checkpoint, device)

    painted_frames = []
    for ti, frame in enumerate(frames):
        if ti == 0:
            mask, prob, painted = tracker.track(frame, first_frame_annotation)
        else:
            mask, prob, painted = tracker.track(frame)
        painted_frames.append(painted)

    tracker.clear_memory()

    print(f"max GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.1f} MB")

    save_path = "/ssd1/gaomingqi/results/TAM/horsejump-high-guided"
    os.makedirs(save_path, exist_ok=True)
    for ti, painted_frame in enumerate(progressbar.progressbar(painted_frames)):
        Image.fromarray(painted_frame).save(f"{save_path}/{ti:05d}.png")
