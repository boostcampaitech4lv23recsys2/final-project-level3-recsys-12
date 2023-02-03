# 딥러닝 모델 환경 정리

## 요약
0. 반드시 backend/data/train.tsv로 저장합니다.
1. 모델 개발자 제외한 사람들:  
    - gcp 폴더 이동
    - ``` python main.py ``` 실행
2. 모델 개발자:
    - model/AE_model 폴더 이동
    - ``` python train.py ``` 실행
    - 모델 학습 후
    - gcp 폴더 이동
    - ``` python main.py --mode 0``` 실행

## 자세한 과정

0. 최신 반영 데이터가 backend/data/train.tsv로 저장됩니다.
1. model/AE_model/train.py을 실행해서 학습 진행  
    이때, 매 epoch마다 최대 RECALL@K가 나오면, model/saved_model/best_model.pt에  저장됩니다.
2. 모델을 학습시킬 사람이 gcp/main.py를 실행해서 학습된 모델을 GCP 상으로 전송합니다.
3. GCP 상의 모델 == 업데이트된 모델을 업데이트하려면 gcp/main.py을 실행해서 모델을 업데이트하면 됩니다.