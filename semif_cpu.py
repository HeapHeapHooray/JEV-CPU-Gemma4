#!/usr/bin/env python3
"""
SemIf 를 이 Linux CPU PC에서 로컬 실행하기 위한 얇은 shim.

SemIf( github.com/TheoLeeCJ/SemIf )의 유일한 GPU 강제 지점은
core.load_causal_model() 뿐이다. 스코어링 로직(direct.score / shared.score_shared)은
device = next(model.parameters()).device 를 따라가므로 CPU에서 그대로 동작한다.
여기서는 CUDA 로더 대신 CPU(float32) 로더를 써서 모델을 올리고,
SemIf 의 실제 direct.score() 를 호출해 '옵션 확률(semantic if)'을 읽는다.

전제:
  - SemIf 저장소가 /tmp/SemIf 에 clone 되어 있음 (SEMIF_DIR 로 변경 가능)
  - pip install torch transformers accelerate  (CPU)

사용:
  python semif_cpu.py
"""
import os
import sys
import time

# SemIf 소스 위치 탐지: 1) SEMIF_DIR 환경변수, 2) 이 파일과 같은 레포의 ./src,
# 3) /tmp/SemIf (개발용 clone). 처음 발견되는 곳을 사용한다.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.environ.get("SEMIF_DIR"),
    _HERE,                # 레포 루트에 src/semif_phase1 이 있는 경우 (JEV-CPU)
    "/tmp/SemIf",         # 개발용 clone
]
for _base in _CANDIDATES:
    if _base and os.path.isdir(os.path.join(_base, "src", "semif_phase1")):
        SEMIF_DIR = _base
        break
else:
    raise RuntimeError(
        "SemIf 소스를 찾을 수 없습니다. SEMIF_DIR 환경변수로 경로를 지정하세요 "
        "(src/semif_phase1 를 포함해야 함)."
    )
sys.path.insert(0, os.path.join(SEMIF_DIR, "src"))

import torch
import transformers
from semif_phase1.direct import score as direct_score


# openjev.com/SemIf 가 쓰는 가장 작은 모델. 원격 로드는 40자 커밋 revision 을 요구한다.
MODEL = "Qwen/Qwen3-0.6B"
REVISION = os.environ.get("QWEN_REV", "main")  # 필요시 40자 커밋 해시로 고정


def load_causal_model_cpu(source: str, revision: str):
    """core.load_causal_model 의 CPU 버전 (CUDA 검사/ device_map 제거)."""
    common = {"trust_remote_code": False}
    if revision and revision != "main":
        common["revision"] = revision
    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        source,
        config=config,
        dtype=torch.float32,      # CPU 안정성 우선
        low_cpu_mem_usage=True,
        **common,
    )
    model.eval()
    metadata = {
        "source": source,
        "revision": revision,
        "dtype": "float32",
        "device": "cpu",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return model, tokenizer, metadata


def main():
    print(f"[load] {MODEL} @ {REVISION}  (CPU / float32)")
    t0 = time.time()
    model, tokenizer, meta = load_causal_model_cpu(MODEL, REVISION)
    print(f"[load] done in {time.time()-t0:.1f}s "
          f"(params={sum(p.numel() for p in model.parameters())/1e9:.2f}B, "
          f"transformers={transformers.__version__})")

    # SemIf 의 결정(row) 스키마: state(증거) + question(기준) + options(2~16개)
    rows = [
        {
            "id": "sentiment",
            "state": "배송이 3일이나 늦었고 고객센터는 연결도 안 됐어요. 정말 실망입니다.",
            "question": "Classify the customer's sentiment.",
            "options": [
                {"id": "positive", "description": "Positive / satisfied"},
                {"id": "neutral", "description": "Neutral"},
                {"id": "negative", "description": "Negative / dissatisfied"},
            ],
        },
        {
            "id": "route",
            "state": "I was double charged and need a refund before Friday.",
            "question": "Which team should handle this ticket?",
            "options": [
                {"id": "billing", "description": "Billing / payments"},
                {"id": "tech", "description": "Technical support"},
                {"id": "sales", "description": "Sales"},
            ],
        },
    ]

    for row in rows:
        print(f"\n=== decision: {row['id']} ===")
        r = direct_score(model, tokenizer, row, meta)
        pairs = sorted(zip(r["option_ids"], r["probabilities"]),
                       key=lambda x: -x[1])
        winner = pairs[0][0]
        print(f"  → 선택: {winner}")
        for oid, p in pairs:
            bar = "█" * int(p * 30)
            print(f"    {oid:10s} {p*100:5.1f}%  {bar}")
        print(f"  (forward {r['forward_seconds']:.1f}s, {r['input_tokens']} tok, "
              f"readout: {r['readout']})")


if __name__ == "__main__":
    main()
