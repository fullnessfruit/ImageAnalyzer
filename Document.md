# ImageAnalyzer 프로젝트 코드 문서

## 문서 편집 원칙
이 문서는 AI가 코드베이스를 이해하고 올바른 코드를 생성하도록 돕기 위한 문서다
- 코드를 보면 바로 알 수 있는 자명한 내용은 생략
- 각 파일과 클래스와 함수의 **의도**, **로직**, **시스템 설명**, **중요한 아키텍처 결정**을 기록
- 이 문서는 인간보다 AI가 주로 보는 문서이므로 수정 이력은 불필요하며 코드의 각 클래스와 함수의 현재 상태를 기술하고, 더 이상 유효하지 않은 내용은 즉시 삭제 해야함

## 프로그래밍 작업 원칙
### General Development Approach
- Think before acting. Read existing files before writing code.
- Reason thoroughly.
- Prefer editing over rewriting whole files.
- Do not re-read files you have already read unless the file may have changed.
- Test your code before declaring done.
- No sycophantic openers or closing fluff.
- User instructions always override this file.
### Project-Specific Approach
- 기능을 구현하기 전에 먼저 이 Document.md를 확인하여, 비슷한 기능이나 유틸리티가 이미 존재하는지 확인
- 기존 코드와 기존 유틸리티 함수(예: `extractAccountFromUrl`, `normalizeUrl` 등)를 적극 재사용하고, 기존과 비슷한 로직을 만들어야 하는 경우가 생기면 가능한 공통 로직으로 만들어서 최대한 같은 로직을 중복 구현하지 않도록 해야함
- 요구사항이 불분명하거나 여러 해석이 가능한 경우, 추측하지 말고 사용자에게 질문
- 코드 수정 후 Document.md도 함께 갱신
- 하나의 정보를 담은 로그는 반드시 한 줄로 작성 (Linux grep 같은 도구로 검색 용이)
  - 좋은 예: `Logger.log('✅ Task completed - id: ${taskId}, duration: ${duration}ms, result: ${result}')`
  - 나쁜 예: 여러 개의 Logger.log 호출로 관련 정보 분산
- 문제의 원인을 바로 파악하기 어려운 경우, 먼저 원인 분석에 도움이 되는 로그를 추가하고 다음 발생 시 로그를 기반으로 재분석
- 사용량 절약을 위해, 어렵지 않은 작업(단순 텍스트 수정, 로그 추가, 간단한 리팩터링 등)은 Gemini CLI를 실행하여 처리할 수 있음. 단, Gemini에게 작업을 넘기기 전에 반드시 사용자에게 먼저 질문하여 넘길지 여부를 확인받을 것

---

## 프로젝트 개요

이미지를 입력받아 다섯 가지 분석을 수행하는 로컬 API 서버:
1. **OCR** — PaddleOCR(DBNet 검출 + CRNN 인식) + manga-ocr(일본어 폴백)로 텍스트 추출 후 searchStrings.tsv와 매칭
2. **캐릭터 인식** — YOLO로 캐릭터 영역 검출 → CLIP 임베딩 추출 → SQLite DB와 코사인 유사도 비교
3. **얼굴 인식** — 얼굴 검출 모델로 얼굴 영역 검출 → ArcFace 임베딩 추출 → SQLite DB와 코사인 유사도 비교
4. **의상 인식** — 캐릭터 영역에서 얼굴 영역을 마스킹한 후 CLIP 임베딩 추출 → SQLite DB와 코사인 유사도 비교
5. **전신 인식** — pretrained YOLO(COCO)로 사람 영역 검출 → 얼굴 마스킹 → CLIP 임베딩 추출 → SQLite DB와 코사인 유사도 비교

크롬 확장 프로그램에서 `POST /analyze`로 호출.

---

## 아키텍처

### 파이프라인 흐름
```
이미지 → POST /analyze
  ├─ OCR: PaddleOCR DBNet 검출 → CRNN 인식(중국어 모델) → 폴백(manga-ocr/한국어/영어) → searchStrings 매칭
  ├─ YOLO 검출 + 얼굴 검출 + YOLO-person 검출 (병렬 실행, 결과를 아래에서 공유)
  ├─ 캐릭터: YOLO 캐릭터 영역 → 크롭 → CLIP 임베딩 → DB 코사인 매칭
  ├─ 얼굴: face-det 얼굴 영역 → 크롭 → ArcFace 임베딩 → DB 코사인 매칭
  ├─ 의상: YOLO 캐릭터 영역 - face-det 얼굴 영역 → 마스킹 크롭 → CLIP 임베딩 → DB 코사인 매칭
  └─ 전신: YOLO-person 사람 영역 - face-det 얼굴 영역 → 마스킹 크롭 → CLIP 임베딩 → DB 코사인 매칭
→ JSON 응답
```

### 모델 로딩 전략
모든 ONNX 모델은 서버 시작 시 한 번만 로딩하여 메모리에 유지 (`inference.ts`의 모듈 레벨 싱글톤 세션: `clipSession`, `arcfaceSession`, `faceDetSession`, `yoloSession`; `ocr.ts`의 `detSession`, `recSessions`, `mangaEncoderSession`, `mangaDecoderSession`). CLIP, ArcFace, 얼굴검출, PaddleOCR, manga-ocr 모델은 첫 실행 시 `model-downloader.ts`에서 자동 다운로드. YOLO 캐릭터 모델은 별도 학습 필요.

### 스토리지
- **SQLite** (`db/embeddings.db`): character_embeddings, face_embeddings, costume_embeddings, person_embeddings 테이블
- **config.json**: 유사도 임계값
- **searchStrings.tsv**: OCR 검색 문자열 (한 줄에 하나, 탭 구분으로 복합 조건)

---

## 파일별 상세 설명

### config.json
```json
{
  "similarityThreshold": { "character": 0.75, "face": 0.80, "costume": 0.75, "person": 0.70 }
}
```
매 `/analyze` 요청마다 다시 읽음 (서버 재시작 없이 설정 변경 반영).

### searchStrings.tsv
OCR 검색 문자열을 한 줄에 하나씩 기록. 매 `/analyze` 요청마다 다시 읽음.

**매칭 규칙**: 탭을 포함하지 않는 줄은 단순 문자열 매칭. 탭으로 구분된 줄은 모든 부분 문자열이 OCR 텍스트 내에 각각 존재해야 매칭 성공.

```
大西亜玖璃
#桐生美也	Uehara Ayumu	大西亜玖璃
```
위 예시에서 1행은 "大西亜玖璃"가 있으면 매칭. 2행은 탭으로 구분된 세 문자열이 이미지 내에 모두 존재해야 매칭.

### server/src/index.ts
**역할**: 메인 서버 엔트리 포인트

- `main()`: 초기화 순서 — 모델 다운로드 → DB 초기화 → ONNX 모델 로딩 → OCR 초기화(PaddleOCR + manga-ocr) → Express 서버 시작
- `POST /analyze`: multipart `image` 필드를 받아 OCR + YOLO + 얼굴 검출 + YOLO-person 검출 병렬 실행 → 캐릭터/얼굴/의상/전신 인식 병렬 처리. 응답에 `_detections` 필드 포함 (raw 검출 수, 디버깅용)
- `findBestMatch()`: 캐릭터/얼굴/의상/전신 인식에서 공통 사용하는 DB 임베딩 매칭 헬퍼
- `recognizePersons()`: 사람 영역 내 얼굴 없으면 마스킹 없이 원본 크롭 사용 (fallback)
- `GET /health`: 헬스 체크 엔드포인트
- 모든 Origin에 대해 CORS 허용 (크롬 확장 프로그램 접근용)
- `loadConfig()`를 요청마다 호출하여 재시작 없이 설정 변경 반영

### server/src/ocr.ts
**역할**: OCR 파이프라인 — PaddleOCR (텍스트 검출 + 인식) + manga-ocr (일본어 폴백)

**모듈 레벨 세션**: `detSession` (DBNet 검출), `recSessions` (Map: "ch"/"ko"/"en" → CRNN 세션), `recDicts` (Map: 언어 → 문자 사전), `mangaEncoderSession`/`mangaDecoderSession`/`mangaVocab`

**파이프라인 흐름**:
1. `detectTextRegions()` — PaddleOCR DBNet으로 텍스트 영역 검출. 최대변 960px 리사이즈(32배수), ImageNet 정규화. 출력 score map을 이진화 → 연결 컴포넌트(DFS 스택) → 바운딩 박스. `DET_UNCLIP_RATIO`로 확장.
2. `recognizeRegion()` — 영역 크롭 후 고정 높이(48px) 비율 리사이즈, [-1,1] 정규화 → CRNN 인식 → CTC 디코딩.
3. `performOCR()` — 1차: 중국어 모델(CJK+영어 커버)로 전체 인식 → `shouldFallback()` 판정 → 일본어/CJK → manga-ocr, 한국어 → 한국어 모델, 영어 → 영어 모델. 최종 텍스트 결합 후 searchStrings 매칭.

**CTC 디코딩**: class 0 = blank, class i (i>0) = dict[i-1]. 연속 동일 클래스 제거, blank 제거. 각 타임스텝의 softmax 확률로 confidence 계산.

**manga-ocr**: VisionEncoderDecoder 구조. ViT 인코더(224×224, [-1,1] 정규화) → BERT 디코더(autoregressive greedy, `[CLS]` 시작, `[SEP]` 종료). WordPiece 토큰 → `##` 제거하여 문자열 생성.

**폴백 판정** (`shouldFallback`): confidence < 0.8, 텍스트 길이 2 이하인데 confidence < 0.9, 특수문자가 50% 이상, 일본어 비율이 애매한 경우 등.

**언어 감지** (`detectLang`): 히라가나/카타카나 비율로 일본어, 한글 비율로 한국어, CJK 한자 비율로 중국어, 그 외 영어.

### server/src/inference.ts
**역할**: 모든 ONNX 모델 추론 로직

**모듈 레벨 세션**: `clipSession`, `arcfaceSession`, `faceDetSession`, `yoloSession`, `yoloPersonSession` — `loadModels()`에서 한 번 설정 후 이후 모든 호출에서 사용.

**전처리**:
- `preprocessForCLIP`: 224×224 리사이즈, ImageNet mean/std 정규화, NCHW 레이아웃
- `preprocessForArcFace`: 112×112 리사이즈, [-1, 1] 정규화, NCHW 레이아웃
- `preprocessForYOLO`: 640×640 리사이즈, [0, 1] 정규화, 박스 좌표 복원용 스케일 팩터 반환
- `preprocessForFaceDet`: 레터박싱(비율 유지 + 패딩) 640×640, mean 127.5 / scale 128 정규화. `ratio`, `padX`, `padY` 반환하여 좌표 복원에 사용. 세로로 긴 이미지 등 극단적 비율에서도 얼굴 검출이 정상 동작.

**YOLO 출력 파싱**: YOLOv8 출력 shape `[1, numFields, numBoxes]` (전치 형식). 처음 4개 필드는 cx/cy/w/h, 나머지는 클래스 신뢰도. IoU 임계값 0.45로 NMS 적용.

**얼굴 검출 출력 파싱 (SCRFD)**: InsightFace det_10g 모델은 9개 텐서 출력 — 3개 FPN 레벨(stride 8/16/32) × (score [N,1] + bbox [N,4] + landmarks [N,10]). 텐서 컬럼 수(1/4/10)로 그룹핑, 앵커 수(내림차순)로 정렬하여 stride 매칭. 앵커당 2개, 앵커 좌표 기반 bbox 디코딩. IoU 임계값 0.4로 NMS 적용.

`detectPersons(imageBuffer)` — COCO pretrained YOLO로 사람 영역만 검출 (class 0 = person). `parseYOLOOutput` 공유.

`cosineSimilarity(a, b)` — 캐릭터/얼굴/의상/전신 매칭에 공통 사용되는 유틸리티.

`cropRegion(imageBuffer, box)` — sharp로 영역 추출, 좌표를 음수가 되지 않도록 클램프.

`extractCostumeRegion(imageBuffer, characterBox, faceBoxes)` — 캐릭터 영역을 크롭한 뒤, 해당 영역 내 얼굴을 찾아 회색(128,128,128)으로 마스킹. 얼굴 위 30% 패딩을 추가하여 머리카락도 마스킹. 캐릭터 영역 내에 얼굴이 없으면 null 반환.

### server/src/db.ts
**역할**: SQLite 데이터베이스 관리

동일 스키마의 네 테이블:
- `character_embeddings`: name, image_path (unique), embedding (BLOB)
- `face_embeddings`: name, image_path (unique), embedding (BLOB)
- `costume_embeddings`: name, image_path (unique), embedding (BLOB)
- `person_embeddings`: name, image_path (unique), embedding (BLOB)

`isImageRegistered()` — `image_path`로 중복 등록 방지.

임베딩은 Float32Array의 raw 버퍼로 저장.

### server/src/model-downloader.ts
**역할**: 첫 실행 시 ONNX 모델 자동 다운로드

모델 목록:
- `clip`: Qdrant/clip-ViT-B-32-vision (`clip-vit-base-patch32.onnx`)
- `arcface`: public-data/insightface buffalo_l w600k_r50 (`arcface-w600k-r50.onnx`)
- `facedet`: public-data/insightface buffalo_l det_10g (`face-detection.onnx`)
- `yolo-person`: COCO pretrained YOLOv8n (`yolo-person.onnx`) — 자동 다운로드 시도, 실패 시 `training/export_person_model.py`로 수동 변환
- `paddleocr-det`: OleehyO/paddleocrv4 DBNet 텍스트 검출 (`paddleocr-det.onnx`)
- `paddleocr-rec-ch`: monkt/paddleocr-onnx 중국어 CRNN 인식 (`paddleocr-rec-ch.onnx`) — CJK+영어 커버, 1차 인식 모델
- `paddleocr-dict-ch`: 중국어 모델 문자 사전 (`paddleocr-dict-ch.txt`)
- `paddleocr-rec-ko`: 한국어 CRNN 인식 (`paddleocr-rec-ko.onnx`) — 한국어 폴백용
- `paddleocr-dict-ko`: 한국어 모델 문자 사전 (`paddleocr-dict-ko.txt`)
- `paddleocr-rec-en`: 영어 CRNN 인식 (`paddleocr-rec-en.onnx`) — 영어 폴백용
- `paddleocr-dict-en`: 영어 모델 문자 사전 (`paddleocr-dict-en.txt`)
- `manga-ocr-encoder`: l0wgear/manga-ocr-2025-onnx ViT 인코더 (`manga-ocr-encoder.onnx`) — 일본어 폴백용
- `manga-ocr-decoder`: BERT 디코더 (`manga-ocr-decoder.onnx`)
- `manga-ocr-vocab`: WordPiece 어휘 (`manga-ocr-vocab.txt`)

HTTP 301/302 리다이렉트 처리. stdout에 다운로드 진행률 표시. 파일이 이미 존재하면 스킵.

### scripts/register-characters.ts
**역할**: 캐릭터 임베딩 일괄 등록

`data/characters/` 하위 폴더 스캔. 폴더명 = 캐릭터 이름. 각 이미지에서 CLIP 임베딩 추출 후 `character_embeddings` 테이블에 저장. 이미 등록된 이미지(상대 경로 기준)는 스킵.

### scripts/register-faces.ts
**역할**: 얼굴 임베딩 일괄 등록

`data/faces/` 하위 폴더 스캔. 폴더명 = 인물 이름. 각 이미지에서 얼굴 검출 (가장 큰 얼굴 사용) → ArcFace 임베딩 추출 → `face_embeddings` 테이블에 저장. 이미 등록된 이미지는 스킵.

### scripts/register-costumes.ts
**역할**: 의상 임베딩 일괄 등록

`data/costumes/` 하위 폴더 스캔. 폴더명 = 해당 의상을 입었던 캐릭터 이름. 각 이미지에서 CLIP 임베딩 추출 후 `costume_embeddings` 테이블에 저장. 이미 등록된 이미지는 스킵. 의상 이미지는 얼굴 없이 의상만 크롭된 상태 권장.

### scripts/register-persons.ts
**역할**: 전신 임베딩 일괄 등록

`data/persons/` 하위 폴더 스캔. 폴더명 = 인물 이름. 각 이미지에서 얼굴 검출 → 얼굴 마스킹(의상 인식과 동일한 `extractCostumeRegion` 사용, 전체 이미지를 character box로 전달) → CLIP 임베딩 추출 → `person_embeddings` 테이블에 저장. 얼굴이 검출되지 않으면 마스킹 없이 원본 사용.

### training/export_person_model.py
**역할**: pretrained YOLOv8n(COCO)를 ONNX로 변환하는 폴백 스크립트. 자동 다운로드 실패 시 사용.

### scripts/update-config.ts
**역할**: searchStrings.tsv 관리 CLI 도구

`--add "문자열"` / `--remove "문자열"` / `--list`

### training/train.py
**역할**: YOLOv8 Nano 파인튜닝 + ONNX 변환

- `data.yaml` 자동 생성 (클래스: ["character", "face"])
- `data/yolo-training/images` + `labels` 데이터로 학습
- `best.pt` → `models/yolo-characters.onnx`로 변환
- CLI 인자로 커스텀 경로 지원 (Google Colab용)

### training/augment.py
**역할**: 학습 데이터 증강

`data/characters/` 이미지를 읽어 다음 증강을 랜덤 조합 적용:
- 실루엣 변환
- 색상 변형 (밝기, 대비, 채도)
- 회전 (-30° ~ +30°)
- 좌우 반전
- 가우시안 블러

원본 + 증강 이미지를 `data/yolo-training/images/`에, YOLO 형식 라벨을 `data/yolo-training/labels/`에 저장. 기본 원본 1개당 증강 5개 생성.

**라벨 형식**: 전체 이미지 바운딩 박스 (`class_id 0.5 0.5 1.0 1.0`) — 소스 이미지가 대상 주변으로 밀착 크롭되어 있다고 가정.
