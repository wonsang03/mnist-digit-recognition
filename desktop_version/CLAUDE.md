# desktop_version — Tkinter 데스크톱 버전

상위 폴더의 `../CLAUDE.md` 규칙을 그대로 따르고, 여기에는 데스크톱 전용 규칙만
적는다. (Created: 2025-09-15)

## Project Overview

Tkinter 창에 280x280 캔버스를 띄우고, 마우스로 그린 숫자를 `common/`의 CNN으로
인식해 예측값·신뢰도·상위 3개 후보를 보여준다.

## Commands

```bash
python desktop_version/digit_recognition.py   # 프로젝트 루트에서 실행
..\run_desktop.bat                            # 탐색기에서 더블 클릭
```

## Tech Stack

- Tkinter — 표준 라이브러리, 별도 설치 불필요
- Pillow — 캔버스와 같은 내용을 담는 오프스크린 이미지

## Architecture

- `digit_recognition.py` — 파일 하나로 끝나는 단일 창 애플리케이션
  - `DigitRecognizerApp` — 위젯 구성, 그리기 이벤트, 인식 처리
  - `_paint()` — 캔버스와 Pillow 이미지에 **같은 획을 두 번** 그린다
  - `clear_canvas()` / `recognize()` — [Clear] / [Recognize] 버튼 동작

### 왜 그림을 두 번 그리는가

Tk 캔버스 위젯은 픽셀을 다시 읽어올 방법이 이식성 있게 제공되지 않는다
(`postscript` 방식은 Ghostscript가 필요하다). 그래서 사용자가 보는 캔버스와
별개로 `PIL.ImageDraw`용 이미지를 하나 더 유지하고, 모델에는 그쪽을 넘긴다.
그리기 로직을 고칠 때는 두 대상에 동일하게 반영해야 한다.

## Code Style

- 상위 `CLAUDE.md`의 규칙을 따른다(영어 코드/주석, 파일 상단 생성일).
- 색상·크기 같은 UI 값은 파일 상단에 상수로 모은다.
- GUI 스레드를 막지 않는다. 오래 걸리는 작업은 넣지 않는다.
- 모델 파일이 없는 등 실패 상황은 `messagebox`로 알리고 종료한다.

## Development Notes

- 단축키: [Enter] 인식, [Esc] 지우기.
- 붓 굵기 `BRUSH_RADIUS = 11`은 280px 캔버스를 28x28로 줄였을 때 MNIST와 비슷한
  선 두께가 되도록 맞춘 값이다. 캔버스 크기를 바꾸면 같이 조정한다.
- 창 크기는 고정(`resizable(False, False)`)이다. 캔버스 크기와 전처리 가정이
  묶여 있기 때문이다.
