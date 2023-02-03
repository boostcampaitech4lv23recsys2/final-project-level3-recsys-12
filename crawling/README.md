# 오늘의 집 사이트를 크롤링 합니다. 
#### 인터넷 상태를 확인하고 실행해 주세요.

<br></br>
# args
-- just_update : 해당 옵션을 활성화 하면 탐색하지 않은 데이터만 크롤링합니다. 비활성화 시 전체 데이터를 크롤링합니다.

-- target(list) : 크롤링을 진행할 타겟을 지정합니다. housecode hi hc house item card color가 있습니다.


<br></br>
# 실행 방법
## 전체 데이터 탐색 시
-  python main.py

## 업데이트만 진행 시
-  python main.py --just_update

## 색상을 제외하고 업데이트 진행 시
- python main.py --just_update --target housecode hi hc house item card

<br></br>
# 주의사항
- housecode 크롤링이 완료된 후 hi, hc, house를 크롤링하세요.
- hi 크롤링이 완료된 후 item을 크롤링하세요.
- hc 크롤링이 완료된 후 card를 크롤링하세요.
- color는 순서에 관계 없이 크롤링 해도 무방합니다.

<br></br>
# 구조
crawling</br>
┣ data</br>
┃ ┣ hi_interaction</br>
┃ ┃ ┣ house_item_interaction_{house번호}.csv</br>
┃ ┃ ┣ ...</br>
┃ ┣ hc_interaction</br>
┃ ┃ ┣ house_card_interaction_{house번호}.csv</br>
┃ ┃ ┣ ...</br>
┃ ┣ card.csv</br>
┃ ┣ house.csv</br>
┃ ┣ item.csv</br>
┃ ┣ house_code.csv</br>
┃ ┣ house_floor_color.csv</br>
┃ ┣ house_wall_color.csv</br>
┃ ┣ house_main_color.csv</br>
┃ ┣ hi_interaction.csv</br>
┃ ┣ hc_interaction.csv</br>
┣ args.py</br>
┣ config.py</br>
┣ crawling_modules.py</br>
┣ main.py</br>
┣ requirements.txt</br>
┣ utils.py</br>
┣ README.md</br>

<br></br>
# preprocess.py 진행 시 예상 오류
## 1. AttributeError: 'Series' object has no attribute '_is_builtin_func'
- 해당 문제의 경우 pandas용 tqdm을 인식하지 못해서 그렇습니다.
- pip install tqdm==4.62.2 하시면 해결 됩니다.
## 2. jpype._jvmfinder.JVMNotFoundException: No JVM shared library file (libjvm.so) found. Try setting up the JAVA_HOME environment variable properly.
- 해당 문제의 경우, konlpy 토크나아지 사용 시 발생하는 에러입니다.
- 리눅스 기준으로 다음과 같이 해결합니다.
    - apt-get update
    - apt-get install sudo
    - sudo apt install default-jd
- MacOS의 경우 아래를 확인해보세요.
    - https://github.com/konlpy/konlpy/issues/353
- 전 과정이 정상적으로 실행되는지 확인하기 위해 테스트 코드를 진행합니다:
```
python preprocess.py --data_path data/ --test 1
```