# Study-01 — Handwritten Digit Recognition

MNIST 손글씨 숫자 인식 프로그램. 데스크톱(Tkinter) 버전과 웹(Flask) 버전을
같은 모델·같은 전처리로 구현한다. (인공지능활용실습 / 2025-09-15)

## Project Overview

사용자가 마우스로 그린 0~9 숫자를 CNN이 실시간으로 인식하고 신뢰도를 함께
출력한다. 학습은 MNIST(7만 장, 28x28 흑백)로 하며, 학습된 가중치 하나를
데스크톱 버전과 웹 버전이 공유한다.

## Commands

```bash
pip install -r requirements.txt     # 라이브러리 설치
python train_model.py               # 모델 학습 (model/mnist_cnn.pt 생성)
python train_model.py --epochs 8    # 더 오래 학습
python train_model.py --force       # 기존 모델이 있어도 다시 학습
python desktop_version/digit_recognition.py   # 데스크톱 실행
python web_version/app.py                     # 웹 서버 실행 (localhost:5000)
```

윈도우 탐색기에서 더블 클릭으로 실행:

- `run_desktop.bat` — 필요하면 학습한 뒤 데스크톱 창을 띄운다
- `run_web.bat` — 필요하면 학습한 뒤 브라우저를 열고 서버를 띄운다

## Tech Stack

- Python 3.10+ (개발/검증 환경 3.13)
- PyTorch + torchvision — CNN 학습, MNIST 다운로드
- Pillow + NumPy — 캔버스 그림을 28x28 텐서로 변환
- Tkinter — 데스크톱 GUI (표준 라이브러리)
- Flask — 웹 서버, HTML5 Canvas 프런트엔드

## Architecture

```
Study-01/
├── train_model.py            # 학습 스크립트 (CLI 옵션 제공)
├── common/                   # 두 버전이 공유하는 코드
│   ├── model.py              # DigitCNN 정의, 체크포인트 로딩
│   └── preprocess.py         # MNIST 규격 전처리 + 추론 헬퍼
├── desktop_version/          # Tkinter 버전
├── web_version/              # Flask 버전
├── model/mnist_cnn.pt        # 학습된 가중치 (git 제외, 학습 시 생성)
└── data/                     # MNIST 원본 (git 제외, 최초 실행 시 다운로드)
```

핵심 규칙: **인식 파이프라인은 `common/`에만 둔다.** 두 프런트엔드는 그림을
PIL 이미지로 만들어 `predict_digit()`에 넘기기만 하며, 전처리나 모델 코드를
각자 복사해 두지 않는다. 그래야 데스크톱과 웹의 인식 결과가 항상 같다.

전처리는 MNIST 제작 방식을 그대로 따른다.

1. 흰 배경·검은 글씨를 반전해 검은 배경·흰 글씨로 변환
2. 잉크 영역(bounding box)만 잘라내기
3. 긴 변을 20px로 맞춰 비율 유지 축소
4. 무게중심을 28x28 프레임 중앙에 맞춰 배치
5. MNIST 평균/표준편차(0.1307 / 0.3081)로 정규화

이 정규화를 생략하면 학습 데이터와 입력 분포가 달라져 인식률이 크게 떨어진다.

## Code Style

- 모든 코드와 주석은 영어로 작성한다.
- 새로 만드는 모든 파일에는 생성 날짜를 주석(`Created: YYYY-MM-DD`)으로 남긴다.
- 함수는 `snake_case`, 클래스는 `PascalCase`, 상수는 `UPPER_SNAKE_CASE`.
- 공개 함수에는 docstring을 붙이고, 타입 힌트를 사용한다.
- 주석은 코드가 말하지 못하는 "왜"를 적는다. 동작 설명 반복은 하지 않는다.
- 사용자에게 보이는 오류는 다음 행동을 알려 준다(예: "run train_model.py first").
- 폴더명·파일명에는 영문과 숫자만 쓴다(한글 경로는 인코딩 오류의 원인).

## Development Notes

- 현재 상태: 데스크톱·웹 두 버전 모두 동작. 테스트 정확도 약 99%.
- 학습은 기본 3 에폭, CPU 기준 수 분. GPU가 있으면 자동으로 사용한다.
- 학습 시 약간의 랜덤 affine 변형을 준다. 마우스로 그린 글씨는 MNIST보다
  거칠어서, 변형을 주고 학습해야 실제 입력에 더 잘 견딘다.
- 인식이 잘 안 되면: 상자 가운데에 크게, 선을 굵게 그린다.
- 제한 사항: 숫자 한 글자만 인식한다. 여러 글자·문자·한글은 지원하지 않는다.
