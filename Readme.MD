# 쑥쑥 (Ssuk-Ssuk) 
> **음성인식 AI 및 음소 분석 기반, 개인화된 아동 발음 오류 교정 및 언어 놀이 서비스**
>
> 💡 **Team 39 NexStep** | 캡스톤 디자인 및 창업 프로젝트 B (산학 트랙) 

---

## 1. 프로젝트 개요 및 핵심 기능

### 👥 팀 정보 (Team Information)
* **팀명:** Team 39 NexStep
* **팀 구성원:** 
  * **김다은** (Backend & AI Architecture Lead)
  * **Mukhtorova, Madina Marat Kizi** (Frontend & UI/UX Specialist)
* **지도 교수:** 오유란 교수님

<br>

### 🧩 기획 배경 (Pain Point)
* **치료 기관 방문의 물리적 부담:** 오프라인 언어 치료 기관의 부족, 거리에 따른 물리적 접근성 한계 및 높은 비용 부담으로 인한 지속적 치료의 어려움
* **낮은 사회적 인식 및 낙인:** 아동의 언어 지연에 대한 주변의 시선과 편견으로 인해 적극적인 초기 치료 시기를 상실하는 문제 발생
* **가정 학습의 한계:** 전문 지식이 없는 보호자가 가정 내에서 아동을 정밀하게 가이드하고 일상적인 발음 훈련 페이스를 유지하는 데 한계 존재

<br>

### 🌟 주요 서비스 기능 (Key Features)
1. **개인 맞춤형 발음 연습**
   * 아동이 직접 발화하고 녹음하면, 실시간으로 정확한 발음 정오답 여부와 음소 단위 피드백을 전달하는 아동 친화형 인터랙션 환경 제공
2. **보상 기반의 반복 학습 구조**
   * 집중력이 짧은 아동 유저의 지속적인 참여를 유도하고, 자기주도적 훈련을 이끌어내기 위한 게미피케이션(Gamification) 요소 및 보상 시스템 체계화
3. **보호자용 대시보드**
   * 데이터베이스에 기록된 아동의 누적 학습 통계 데이터(오류 음소, 발화량, 훈련 세션 완료율)를 직관적인 그래프로 시각화하여 한눈에 파악할 수 있는 관리 화면 지원

<br>

### 🏗️ 핵심 기술 모듈 (Core Technical Modules)

| 모듈명 | 기술적 역할 및 파이프라인 |
| :--- | :--- |
| **Analyze**<br>*(AI 음소 분석 엔진)* | 아동의 발성 주파수 음성 데이터를 수신하여 파인튜닝된 **Wav2Vec 2.0** 알고리즘으로 디코딩한 뒤, **g2pK 형태소 분석기** 기반의 표준 발음과 대조한다. 이를 통해 초성/중성/종성 단위를 분리하고 편집 거리(Edit Distance)를 계산하여 누락(탈락)되거나 변형(대치)된 음소 에러를 정밀 추적한다. |
| **Session**<br>*(맞춤형 복습 알고리즘)* | `phoneme_errors` 테이블에 실시간으로 축적된 아동별 취약 음소 가중치를 알고리즘 내부에서 분석한다. 다음 학습 세션이 생성될 때 아동이 지속적으로 틀리는 조음 위치의 단어를 최적의 비율로 동적 배치하여 자연스러운 반복 학습을 유도한다. |
| **Report**<br>*(AI 성장 리포트)* | 데이터베이스 내의 수치화된 아동 발화 로그 및 점수를 통합 파싱(Parsing)한다. 이후 **OpenAI API (LLM)** 환경에 프롬프트 제약 조건을 주입하여 비전문가인 보호자가 아동의 성장 상태를 쉽게 이해할 수 있도록 자연어 형태의 요약 리포트와 가정 내 코칭 가이드를 자동 생성한다. |

---

## 2. 기술 스택 및 폴더 구조 (Tech Stack & Project Architecture)

### 🛠️ 기술 스택 (Technical Stacks)
* **Frontend:** React Native, JavaScript
* **Backend:** Python 3.10+, FastAPI, SQLAlchemy (ORM)
* **Database:** PostgreSQL
* **AI / ML & NLP:** PyTorch, HuggingFace Wav2Vec 2.0, g2pK, OpenAI API 

<br>

### 📁 디렉토리 구조 (Directory Structure)
본 프로젝트는 시스템의 확장성과 유지보수성을 확보하기 위해 백엔드와 프론트엔드가 완전히 분리된 독립적 아키텍처로 설계되었다.

```text
capstone-project/
├── backend/                  # FastAPI 기반 비동기 백엔드 및 AI 파이프라인
│   ├── db/                   # 데이터베이스 세션 연결 및 트랜잭션 관리
│   ├── models/               # SQLAlchemy 기반 PostgreSQL 테이블 객체 (Entity) 정의
│   ├── routers/              # 핵심 API 엔드포인트 제어 (Analyze, Session, Report 등)
│   ├── schemas/              # Pydantic 기반 데이터 요청/응답(DTO) 유효성 검증 규격
│   ├── services/             # Wav2Vec 2.0 알고리즘 및 LLM 연동 비즈니스 로직 핵심 레이어
│   ├── main.py               # 애플리케이션 진입점 및 예외 핸들링 미들웨어 설정 파일
│   └── requirements.txt      # 가상환경 구축 및 의존성 패키지 명세서
│
├── frontend/                 # React Native & Expo 기반 크로스 플랫폼 모바일 앱
│   ├── app/                  # 메인 네비게이션 레이아웃 및 뷰(View) 페이지 스크린
│   ├── assets/images/        # 아동 친화형 UI 컴포넌트용 그래픽 및 이모지 리소스
│   ├── components/           # 재사용 가능한 아토믹 디자인 단위 UI 컴포넌트 셋
│   ├── constants/            # 테마 컬러, 자소-음소 매칭 규칙 등 전역 상수 정의
│   ├── hooks/                # 오디오 스트리밍 녹음 및 API 통신 제어용 Custom Hooks
│   ├── lib/scripts/          # 데이터 파싱 및 편집 거리(Edit Distance) 보조 헬퍼 스크립트
│   └── package.json          # 프론트엔드 종속성 라이브러리 명세서
│
├── demo/                     # [Archive] 프로젝트 초기 기능 검증용 프로토타입 코드 가이드
├── docs/                     # 프로젝트 산출물 문서 보관 폴더
└── .gitignore                # 환경 변수(.env) 및 의존성 모듈 업로드 차단 설정 파일
```
---

## 3. 설치 및 빌드와 실행 방법 (Installation & Run Guide)
본 시스템은 보안 및 환경 독립성을 확보하기 위해 API Key와 개인 접속 정보를 소스코드에 포함하지 않는다. 아래 가이드에 따라 로컬 환경 설정을 완료한 후에는 시스템을 구동할 수 있다.

### 📋 사전 준비 사항 (Prerequisites)
시스템 가동을 위해 아래 프로그램들이 로컬 컴퓨터에 미리 설치되어 있어야 한다.
* **Python 3.10 이상:** 백엔드 인공지능 및 API 서버 구동용
* **Node.js 18.x 이상 및 npm:** 프론트엔드 Expo 가동용
* **PostgreSQL 14 이상:** 데이터베이스 인스턴스

---

### 🛠️ [Step 1] 환경 변수 및 데이터베이스 설정

* **환경 변수 파일(`.env`) 생성**
  * 프로젝트 보안 수칙에 따라 실제 API Key는 레포지토리에 업로드되지 않는다.
  * `backend/` 디렉토리 루트에 생성되어 있는 `.env.example` 파일을 복사하여 `.env` 파일을 생성한다.
  * 생성한 `.env` 파일 내부의 안내 주석을 참고하여, 사용자 본인의 PostgreSQL 접속 정보 및 OpenAI API Key를 입력한다.
  ```env
  # backend/.env 파일 예시
  DATABASE_URL=postgresql://[사용자_ID]:[사용자_PW]@localhost:5432/ssukssuk
  OPENAI_API_KEY=your_actual_openai_api_key_here
  ```

* **초기 데이터베이스 스키마 및 프로토데이터(Proto-data) 구축**
  > 💡 **안내:** 과제 검증 및 실제 구동을 위한 기초 데이터셋(언어 치료용 기본 단어 아카이브 및 테스트 데이터)이 덤프 파일 형태로 포함되어 있다.
  
  1. 로컬 데이터베이스 인스턴스에 `ssukssuk` 스키마(데이터베이스)를 생성한다.
  2. 터미널에서 아래 명령어를 실행하여 테이블 구조와 기초 데이터를 일괄 세팅한다.
  ```bash
  # PostgreSQL 가동 후 백엔드 내에 포함된 덤프 파일 적용
  psql -U postgres -d ssukssuk < ./backend/db/dump_ssukssuk.sql
  ```

### ⚡ [Step 2] 백엔드(FastAPI) 설치 및 서버 구동

* **가상환경 구축 및 활성화**
  * 백엔드 디렉토리로 이동한 후, 파이썬 독립 가상환경을 생성하고 활성화한다.
  ```bash
  cd backend
  python -m venv venv
  
  # Windows 환경에서 활성화
  .\venv\Scripts\activate
  
  # Mac / Linux 환경에서 활성화
  source venv/bin/activate
  ```

* **의존성 패키지 설치**
  * 프로젝트 구동 및 AI 파이프라인 처리에 필요한 패키지 의존성을 일괄 설치한다.
  ```bash
  pip install -r requirements.txt
  ```
* **로컬 개발 서버 가동**
  * Uvicorn을 통해 로컬 개발 서버를 가동한다.
  ```bash
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
  ```

* **API 개발 문서 확인**
  * 💡 **Swagger 안내:** 서버 구동에 성공하면 [http://localhost:8000/docs](http://localhost:8000/docs) 주소를 통해 대화형 Swagger API 문서 영역에 진입할 수 있다. 해당 화면에서 모든 엔드포인트의 명세를 확인하고 직접 API 테스트를 수행한다.

### 📱 [Step 3] 프론트엔드(React Native & Expo) 설치 및 앱 실행

* **종속성 라이브러리 설치**
  * 프론트엔드 디렉토리로 이동하여 모바일 인터랙션 환경 구동에 필요한 패키지 모듈을 설치한다.
  ```bash
  cd ../frontend
  npm install
  ```

* **Expo Go 개발 인프라 가동**
  * Expo 개발 서버를 가동한다.
  ```bash
  npx expo start
  ```

* 💡 **디바이스 빌드 및 동기화:** 터미널 스크린에 출력되는 **QR 코드**를 테스트용 디바이스(iOS/Android)의 **Expo Go 애플리케이션** 내부 카메라로 스캔한다. 스캔이 완료되면 실시간 아동 발음 연습 화면과 보호자 대시보드 인터페이스가 모바일 기기에 성공적으로 빌드 및 동기화된다.


### 🧪 [Step 4] 핵심 기술 모듈 모의 평가 및 실험 결과 검증

본 프로젝트의 기술적 신뢰도와 예외 처리 안정성을 수동으로 즉각 검증할 수 있도록 테스트 도구와 모의 평가용 오디오 파일셋이 포함되어 있다.

* **AI 음소 분석 파이프라인 백엔드 유닛 테스트 실행**
  * 준비된 테스트 스크립트를 통해 Wav2Vec 2.0 음소 매칭 및 정오답 탐지 엔진이 정상 연산되는지 검증한다.
  ```bash
  cd ../backend
  pytest tests/test_analyze_engine.py
  ```

* **실험 데이터 가시성 확보 및 확인**
  * **모의 평가용 아동 음성 샘플:** `backend/tests/samples/` 디렉토리 내에 보고서 평가(V 단원)에 활용되었던 실제 아동 발화 예시 `.wav` 파일(예: `sample_bob_sang.wav`)이 탑재되어 있다.
  * **검증 방법:** 앱 구동 중 녹음 기능 외에도, 해당 파일들을 Swagger API의 `POST /api/analyze` 엔드포인트에 직접 업로드하여 백엔드가 음소 수준 오류 데이터(`phoneme_errors`)를 안정적으로 추출하고 DB 트랜잭션 예외 처리를 수행하는 과정을 즉각 확인한다.