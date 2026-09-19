# JEV-CPU

**GPU 없이 CPU에서 돌아가는 [SemIf](https://github.com/TheoLeeCJ/SemIf) (Jev semantic-if 엔진) 포트 + 웹 UI.**

SemIf는 "텍스트를 생성하지 않고, 선언된 옵션 토큰의 로짓(logits)만 읽어" 실행 시점의
의미 기반 분기(*semantic if*)를 내리는 엔진입니다. 원본은 **CUDA GPU(4B BF16 모델)** 를 요구하지만,
JEV-CPU는 **CPU / float32 + 소형 모델**로 동일한 엔진을 그대로 구동합니다.

> 원본 SemIf 문서는 [`README.SemIf-upstream.md`](./README.SemIf-upstream.md) 참고.

---

## 핵심: 왜 CPU에서 되는가

SemIf에서 GPU를 강제하는 지점은 `src/semif_phase1/core.py` 의 `load_causal_model()` **단 한 곳**입니다:

```python
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise ValueError("Expose exactly one CUDA GPU ...")
...
dtype=torch.bfloat16, device_map={"": "cuda:0"}   # ← GPU 고정
```

반면 실제 스코어링 로직은 **디바이스 독립적**으로 작성돼 있습니다:

- `direct.py` / `shared.py` → `device = next(model.parameters()).device` 를 따라감
- `torch.cuda.synchronize` 는 `device.type == "cuda"` 일 때만 호출 (CPU면 no-op)
- 옵션 로짓 슬롯 추출 · softmax 는 순수 연산

따라서 **로더만 CPU로 바꾸면**(`semif_cpu.py`) 원본 스코어링 코드를 수정 없이 CPU에서 사용할 수 있습니다.

---

## 구성 파일 (JEV-CPU 추가분)

| 파일 | 설명 |
|------|------|
| `semif_cpu.py` | CPU / float32 모델 로더 shim + `direct.score()` 호출 예제. SemIf 소스는 `./src` 를 자동 탐지. |
| `server.py` | 순수 표준 라이브러리 웹 서버(**포트 1122**). 모델 1회 로드 후 재사용. 3분할 웹 UI 제공. |
| `src/semif_phase1/` | 원본 SemIf 엔진 (그대로) |

---

## 빠른 시작

```bash
python3 -m venv .venv && source .venv/bin/activate      # (Debian: apt install python3-venv)
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install transformers accelerate

# 1) CLI 데모 — 옵션 확률(semantic if) 출력
python semif_cpu.py

# 2) 웹 UI — http://localhost:1122
python server.py
```

최초 실행 시 `Qwen/Qwen3-0.6B`(~2.4GB, float32)를 Hugging Face에서 내려받아 CPU에 로드합니다.

### 요구 사항
- Python 3.10+
- RAM ~3GB 이상 (0.6B float32 기준). GPU 불필요.
- 인터넷(모델 최초 다운로드용)

---

## 웹 UI

`server.py` 는 세 개의 패널로 구성된 단일 페이지를 제공합니다:

```
┌─────────────────────┬──────────────────────┐
│ ① 판단할 데이터      │                      │
│   (State / Evidence) │   ③ 판단 결과        │
├─────────────────────┤   (Results)          │
│ ② 기준 추가          │   옵션별 확률 막대    │
│   (Criteria)         │                      │
└─────────────────────┴──────────────────────┘
```

- **① 판단할 데이터** — 리뷰·티켓·로그·JSON 등 판단 대상(`state`)
- **② 기준 추가** — 질문(criterion) + 옵션(2개 이상)을 동적으로 추가
- **③ 판단 결과** — 각 기준별로 모델이 읽은 옵션 확률과 선택 결과를 막대로 표시

### API

```
GET  /api/health
POST /api/decide
     { "state": "...", "criteria": [ { "id","question","options":[{"id","description"},...] } ] }
```

---

## 검증된 결과 (CPU, Qwen3-0.6B float32)

| 입력(state) | 기준 | 결과 | forward |
|------|------|------|---------|
| "I was double charged and need a refund before Friday." | 처리 팀 라우팅 | **billing 100%** ✅ | ~1s |
| 영어 감성 리뷰 | 감성 분류 | **negative 99.9%** ✅ | ~1s |

- 모델 로드 ≈ 5–17초, 결정 1건당 CPU forward **≈ 1초** (텍스트 생성이 없어 빠름)

### ⚠️ 소형 모델 정확도 주의
`Qwen/Qwen3-0.6B` 는 SemIf 공식 안내대로 *"작은 모델이라 정확도가 낮을 수 있음"* 에 해당합니다.
특히 **짧은 한국어 라벨의 감성 분류**에서 오답이 관찰됩니다(라우팅 등 명확한 과제는 정확).
정확도가 중요하면 `semif_cpu.py` 의 `MODEL` 을 더 큰 모델(예: MiniCPM5-2B / Qwen3.5-4B)로 교체하세요.
단 4B 모델은 원본 SemIf처럼 `transformers` 의 네이티브 Qwen3.5 지원과 더 많은 RAM이 필요합니다.

---

## 크레딧 / 라이선스

- 엔진 원본: **[TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf)** — "Semantic ifs from open models."
  브라우저 데모: <https://openjev.com/>
- 개념: TypeSafe 의 *Jev* 인터페이스 패턴
- 원본 라이선스 및 서드파티 고지는 [`LICENSE`](./LICENSE), [`THIRD_PARTY.md`](./THIRD_PARTY.md) 를 따릅니다.

JEV-CPU 는 SemIf 위에 CPU 로더 shim 과 웹 UI 를 더한 것으로, 원본 스코어링 로직을 변경하지 않습니다.
