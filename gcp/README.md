# 학습한 딥러닝 model 업로드하고 다운로드 편하게 하기


<br></br>
# 초기 설정
- ``` pip install --upgrade google-cloud-storage ```
- 노션 - 환경설정 - key 다운로드해서 gcp/keys 디렉토리에 저장합니다.

<br></br>
# 실행 방법
## 딥러닝 모델을 backend/inference로 불러오기
-  python main.py

## 학습한 딥러닝 모델을 GCP로 저장오기
-  python main.py --mode 0

<br></br>
# 기타 문제
- 권준혁한테 문의하세요.

<br></br>
# GCP storage 사용 이유
- 딥러닝 모델의 용량이 GitHub으로 tracking되게 하기에는 100MB를 넘었습니다. (약 270MB)
- 그렇다고 모델 학습 후 모델을 로컬 컴퓨터로 다운로드 후에 슬랙이나 카톡으로 전송하면 파일이 깨져서 사용이 불가능했습다.
- 그래서 GCP storage API를 활용해서 이를 해결했습니다.
- 이를 통해 얻을 수 있는 추가적인 장점은 다음과 같습니다:
1. 웹개발 담당자가 필요할 때 모델 변경을 쉽게할 수 있습니다.
2. Airflow를 활용해서 모델을 GCP에 미리 저장할 수 있습니다.
3. 새롭게 학습된 모델과 기존의 모델의 차이가 발생하는데, GCP에 학습된 모델을 별도로 저장하면서 웹개발 담당자가 학습된 모델이 필요할 때에만 불러와서, 이 차이로 인해 발생하는 오류를 줄일 수 있습니다.