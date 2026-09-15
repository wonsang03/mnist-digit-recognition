# web_version — Flask 웹 버전

상위 폴더의 `../CLAUDE.md` 규칙을 그대로 따르고, 여기에는 웹 전용 규칙만 적는다.
(Created: 2025-09-15)

## Project Overview

브라우저의 HTML5 Canvas에 숫자를 그리면 base64 PNG로 서버에 보내고, 서버가
`common/`의 CNN으로 인식해 예측값·신뢰도·10개 클래스 점수·모델이 실제로 본
28x28 이미지를 JSON으로 돌려준다.

## Commands

```bash
python web_version/app.py                    # http://localhost:5000
python web_version/app.py --port 8080        # 포트 변경
python web_version/app.py --debug            # 자동 리로드
..\run_web.bat                               # 탐색기에서 더블 클릭
```

## Tech Stack

- Flask — 라우팅과 템플릿
- 순수 HTML/CSS/JavaScript — 빌드 도구·프레임워크 없음

## Architecture

```
web_version/
├── app.py                  # Flask 앱, /predict 엔드포인트
├── templates/index.html    # 화면 구조
└── static/
    ├── css/style.css       # 스타일 (데스크톱 버전과 같은 색상 팔레트)
    └── js/app.js           # 캔버스 그리기 + fetch 호출
```

### API

| 메서드 | 경로 | 요청 | 응답 |
| --- | --- | --- | --- |
| GET | `/` | – | 그리기 페이지 |
| POST | `/predict` | `{"image": "data:image/png;base64,..."}` | `{digit, confidence, scores[10], preview}` |
| GET | `/health` | – | `{status, model_ready}` |

오류는 HTTP 상태 코드와 `{"error": "..."}` 로 돌려준다. 400은 잘못된 이미지,
503은 학습된 모델이 없는 경우다.

## Code Style

- 상위 `CLAUDE.md`의 규칙을 따른다(영어 코드/주석, 파일 상단 생성일).
- JavaScript는 의존성 없이 바닐라로 작성하고, IIFE로 전역 오염을 막는다.
- 색상은 CSS 변수(`:root`)로 관리한다.
- 서버에서 온 값은 그대로 신뢰하지 않는다. 디코딩 실패는 400으로 처리한다.
- 인식 중에는 버튼을 비활성화해 중복 요청을 막는다.

## Development Notes

- 브라우저 캔버스는 그리지 않은 영역이 투명하므로, 서버에서 흰 배경에 합성한 뒤
  전처리한다. 이 단계를 빼면 배경이 검게 해석돼 인식이 전부 어긋난다.
- `touch-action: none`으로 태블릿에서 스크롤 대신 그리기가 되게 했다.
- CSS로 캔버스가 축소될 수 있으므로, 좌표 계산에 `getBoundingClientRect()` 비율을
  적용한다.
- 개발용 서버다. 외부 공개용이 아니며 기본 바인딩은 `127.0.0.1`이다.
