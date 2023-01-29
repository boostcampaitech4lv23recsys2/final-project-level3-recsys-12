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