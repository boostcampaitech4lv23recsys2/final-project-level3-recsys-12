<p align="center"><img src="https://cdn-images-1.medium.com/fit/t/1600/480/1*du7p50wS_fIsaC_lR18qsg.png"></p>

## 환경 설정

0. 기본 설정
```
> apt-get update
> apt install curl
```

1. Poetry 설치하기
```
> curl -sSL https://install.python-poetry.org | python3 -
```
- 자신의 환경에 맞게 Poetry를 설치해줍니다.
-> https://python-poetry.org/docs/

2. Poetry 가상환경 구축
```
> cd backend
> poetry install
```
backend 폴더로 이동 후 준비되어 있는 poetry.lock 파일과 poetry.toml 파일을 이용하여 final/backend 경로에 다음 명령어를 실행합니다. 
이를 통해 poetry 가상환경을 구축하고 프로젝트 의존성 파일들을 설치합니다.

3. FastAPI backend 서버를 실행합니다.
```
> poetry run python __main__.py
```
