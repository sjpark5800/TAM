<!-- ![](./assets/track-anything-logo.jpg) -->

# AI602 : Track Anything Model (TAM)
## 20257016 Seojeong Park 


<div align=center>
<img src="./assets/tam.png"/>
</div>

## Setting
```shell
conda create -n tam python=3.10
pip install -r requirements.txt
```

## Prediction for DAVIS 2016


```shell
# reproduced results
python app.py

# for improved version
python new_app_for_davis.py
```



## Evaluation for DAVIS 2016

```shell
# for reproduced version
python sav_evaluator.py --gt_root /workspace/TAM/DAVIS_2016/Annotations/480p/ --pred_root /workspace/TAM/result

# improved results
python sav_evaluator.py --gt_root /workspace/TAM/DAVIS_2016/Annotations/480p/ --pred_root /workspace/TAM/result_new
```
