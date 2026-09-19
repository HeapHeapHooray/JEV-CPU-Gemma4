#!/usr/bin/env python3
"""
SemIf 웹 UI 서버 (CPU 백엔드).

- 백엔드 엔진: SemIf( /tmp/SemIf )의 direct.score() — 텍스트 생성 없이 옵션 logits 읽기
- 모델: Qwen/Qwen3-0.6B (CPU / float32), 서버 시작 시 1회 로드 후 재사용
- UI: 좌상단=판단할 데이터(state), 좌하단=기준 추가, 우측=판단 결과
- 포트: 8080

  . .venv/bin/activate && python server.py
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from semif_cpu import load_causal_model_cpu, direct_score, MODEL, REVISION

PORT = 8080
_LOCK = threading.Lock()
_STATE = {"model": None, "tokenizer": None, "meta": None}


def get_engine():
    if _STATE["model"] is None:
        print(f"[engine] loading {MODEL} @ {REVISION} (CPU/float32) ...", flush=True)
        t0 = time.time()
        m, tok, meta = load_causal_model_cpu(MODEL, REVISION)
        _STATE.update(model=m, tokenizer=tok, meta=meta)
        print(f"[engine] ready in {time.time()-t0:.1f}s", flush=True)
    return _STATE["model"], _STATE["tokenizer"], _STATE["meta"]


def decide(state, criteria):
    """criteria: [{id, question, options:[{id, description}]}] -> 결과 리스트."""
    model, tok, meta = get_engine()
    results = []
    with _LOCK:
        for crit in criteria:
            row = {
                "id": crit.get("id") or "criterion",
                "state": state,
                "question": crit.get("question", ""),
                "options": crit.get("options", []),
            }
            try:
                r = direct_score(model, tok, row, meta)
                pairs = sorted(
                    zip(r["option_ids"], r["probabilities"]), key=lambda x: -x[1]
                )
                results.append({
                    "id": row["id"],
                    "question": row["question"],
                    "winner": pairs[0][0],
                    "ranked": [{"id": o, "prob": p} for o, p in pairs],
                    "forward_seconds": round(r["forward_seconds"], 2),
                    "input_tokens": r["input_tokens"],
                    "error": None,
                })
            except Exception as exc:  # SemIf validate_row 등에서 발생
                results.append({
                    "id": row["id"], "question": row["question"],
                    "winner": None, "ranked": [], "error": str(exc),
                })
    return results


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, json.dumps({"ok": True, "model": MODEL,
                                        "loaded": _STATE["model"] is not None}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/decide":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            state = payload.get("state")
            criteria = payload.get("criteria", [])
            if not state or not isinstance(criteria, list) or not criteria:
                raise ValueError("state 와 criteria(1개 이상)가 필요합니다")
            t0 = time.time()
            results = decide(state, criteria)
            self._send(200, json.dumps({
                "results": results,
                "total_seconds": round(time.time() - t0, 2),
                "model": MODEL,
            }, ensure_ascii=False))
        except Exception as exc:
            self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=False))


PAGE = r"""<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JEV-CPU · Semantic Decisions on CPU</title>
<style>
:root{
  --bg:#0e1116; --panel:#161b22; --panel2:#1c232d; --border:#2a323d;
  --txt:#e6edf3; --muted:#8b98a5; --accent:#4c8dff; --accent2:#2ea043;
  --danger:#f85149; --chip:#22303f;
}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;
  background:var(--bg);color:var(--txt);height:100vh;overflow:hidden}
header{display:flex;align-items:center;gap:12px;padding:12px 20px;border-bottom:1px solid var(--border);
  background:var(--panel)}
header h1{font-size:16px;margin:0;font-weight:650}
header .tag{font-size:11px;color:var(--muted);border:1px solid var(--border);padding:2px 8px;border-radius:20px}
header .spacer{flex:1}
#status{font-size:12px;color:var(--muted)}
.grid{display:grid;grid-template-columns:minmax(340px,1fr) minmax(360px,1.1fr);
  grid-template-rows:1fr 1fr;gap:14px;padding:14px;height:calc(100vh - 53px)}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:12px;
  display:flex;flex-direction:column;overflow:hidden}
.panel > .head{padding:10px 14px;border-bottom:1px solid var(--border);font-weight:600;
  display:flex;align-items:center;gap:8px;font-size:13px}
.panel > .head .num{width:20px;height:20px;border-radius:6px;background:var(--chip);color:var(--accent);
  display:grid;place-items:center;font-size:11px;font-weight:700}
.panel > .body{padding:14px;overflow:auto;flex:1}
/* left column stack */
#pData{grid-row:1;grid-column:1}
#pCrit{grid-row:2;grid-column:1}
#pResult{grid-row:1 / span 2;grid-column:2}
textarea,input,select{width:100%;background:var(--panel2);border:1px solid var(--border);color:var(--txt);
  border-radius:8px;padding:8px 10px;font:inherit;resize:vertical}
textarea{min-height:120px;height:100%}
#pData .body{display:flex}
label.mini{font-size:11px;color:var(--muted);display:block;margin:8px 0 4px}
button{cursor:pointer;border:1px solid var(--border);background:var(--panel2);color:var(--txt);
  border-radius:8px;padding:7px 12px;font:inherit;font-weight:600}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
button.primary:disabled{opacity:.5;cursor:not-allowed}
button.mini{padding:3px 8px;font-size:12px;font-weight:500}
.crit{border:1px solid var(--border);border-radius:10px;padding:10px;margin-bottom:10px;background:var(--panel2)}
.crit .row{display:flex;gap:8px;align-items:center;margin-bottom:6px}
.crit .row input{flex:1}
.opt{display:flex;gap:6px;align-items:center;margin:4px 0}
.opt input:first-child{flex:0 0 90px}
.opt input:last-child{flex:1}
.crit .del{color:var(--danger);border-color:transparent;background:transparent}
.toolbar{display:flex;gap:8px;margin-top:6px}
/* results */
.res{border:1px solid var(--border);border-radius:10px;padding:12px;margin-bottom:12px;background:var(--panel2)}
.res h3{margin:0 0 2px;font-size:13px}
.res .q{color:var(--muted);font-size:12px;margin-bottom:10px}
.res .winner{display:inline-block;background:rgba(46,160,67,.15);color:var(--accent2);
  border:1px solid rgba(46,160,67,.4);padding:2px 10px;border-radius:20px;font-weight:700;font-size:12px;margin-bottom:10px}
.bar{margin:7px 0}
.bar .lab{display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px}
.bar .track{height:8px;background:#0c0f14;border-radius:6px;overflow:hidden}
.bar .fill{height:100%;background:linear-gradient(90deg,#4c8dff,#7aa8ff);border-radius:6px;transition:width .4s}
.bar.top .fill{background:linear-gradient(90deg,#2ea043,#57d977)}
.res .meta{color:var(--muted);font-size:11px;margin-top:8px}
.err{color:var(--danger);font-size:12px}
.empty{color:var(--muted);text-align:center;margin-top:40px;font-size:13px}
.hint{color:var(--muted);font-size:11px;margin-top:6px}
</style></head><body>
<header>
  <h1>⚡ JEV-CPU</h1><span class="tag">Semantic decisions · CPU · SemIf engine</span>
  <span class="spacer"></span>
  <span id="status">엔진 준비 확인 중…</span>
</header>
<div class="grid">

  <section class="panel" id="pData">
    <div class="head"><span class="num">1</span> 판단할 데이터 (State / Evidence)</div>
    <div class="body">
      <textarea id="state" placeholder="여기에 판단 대상 데이터를 입력하세요.&#10;예) 고객 리뷰, 티켓 내용, 로그, JSON 등"></textarea>
    </div>
  </section>

  <section class="panel" id="pCrit">
    <div class="head"><span class="num">2</span> 기준 추가 (Criteria)</div>
    <div class="body">
      <div id="crits"></div>
      <div class="toolbar">
        <button class="mini" onclick="addCrit()">+ 기준 추가</button>
        <button class="primary" id="runBtn" onclick="run()" style="margin-left:auto">▶ 판단 실행</button>
      </div>
      <div class="hint">각 기준 = 질문 + 옵션(2개 이상). 모델이 옵션별 확률을 읽어 결정합니다.</div>
    </div>
  </section>

  <section class="panel" id="pResult">
    <div class="head"><span class="num">3</span> 판단 결과 (Results)</div>
    <div class="body" id="results"><div class="empty">기준을 추가하고 <b>판단 실행</b>을 누르세요.</div></div>
  </section>

</div>
<script>
let CID=0;
function el(h){const t=document.createElement('template');t.innerHTML=h.trim();return t.content.firstChild;}
function addCrit(q,opts){
  CID++;
  const c=el(`<div class="crit" data-cid="${CID}">
    <div class="row">
      <input class="q" placeholder="기준/질문 (예: 고객 감성을 분류)" value="${q||''}">
      <button class="del mini" title="삭제">✕</button>
    </div>
    <div class="opts"></div>
    <button class="mini addopt">+ 옵션</button>
  </div>`);
  c.querySelector('.del').onclick=()=>c.remove();
  c.querySelector('.addopt').onclick=()=>c.querySelector('.opts').appendChild(optRow());
  const ob=c.querySelector('.opts');
  (opts||[['positive','긍정'],['negative','부정']]).forEach(o=>ob.appendChild(optRow(o[0],o[1])));
  document.getElementById('crits').appendChild(c);
}
function optRow(id,desc){
  const o=el(`<div class="opt">
    <input placeholder="id" value="${id||''}">
    <input placeholder="설명(description)" value="${desc||''}">
    <button class="del mini">✕</button></div>`);
  o.querySelector('.del').onclick=()=>o.remove();
  return o;
}
function collect(){
  const state=document.getElementById('state').value.trim();
  const crits=[...document.querySelectorAll('.crit')].map(c=>({
    id:'c'+c.dataset.cid,
    question:c.querySelector('.q').value.trim(),
    options:[...c.querySelectorAll('.opt')].map(o=>{
      const i=o.querySelectorAll('input');
      return {id:i[0].value.trim(),description:i[1].value.trim()};
    }).filter(o=>o.id&&o.description)
  }));
  return {state,criteria:crits};
}
async function run(){
  const {state,criteria}=collect();
  const box=document.getElementById('results');
  if(!state){box.innerHTML='<div class="err">좌측 상단에 판단할 데이터를 입력하세요.</div>';return;}
  if(!criteria.length){box.innerHTML='<div class="err">기준을 1개 이상 추가하세요.</div>';return;}
  const btn=document.getElementById('runBtn');btn.disabled=true;btn.textContent='판단 중…';
  box.innerHTML='<div class="empty">CPU 추론 중… (모델 최초 로드 시 시간이 걸릴 수 있어요)</div>';
  try{
    const res=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({state,criteria})});
    const data=await res.json();
    if(!res.ok){box.innerHTML='<div class="err">'+(data.error||'오류')+'</div>';return;}
    render(data);
  }catch(e){box.innerHTML='<div class="err">'+e+'</div>';}
  finally{btn.disabled=false;btn.textContent='▶ 판단 실행';}
}
function render(data){
  const box=document.getElementById('results');
  box.innerHTML='';
  data.results.forEach(r=>{
    if(r.error){
      box.appendChild(el(`<div class="res"><h3>${r.question||r.id}</h3><div class="err">⚠ ${r.error}</div></div>`));
      return;
    }
    const bars=r.ranked.map((o,i)=>`
      <div class="bar ${i===0?'top':''}">
        <div class="lab"><span>${o.id}</span><span>${(o.prob*100).toFixed(1)}%</span></div>
        <div class="track"><div class="fill" style="width:${(o.prob*100).toFixed(1)}%"></div></div>
      </div>`).join('');
    box.appendChild(el(`<div class="res">
      <h3>${r.question||r.id}</h3>
      <div class="winner">→ ${r.winner}</div>
      ${bars}
      <div class="meta">forward ${r.forward_seconds}s · ${r.input_tokens} tokens</div>
    </div>`));
  });
  box.appendChild(el(`<div class="meta" style="text-align:right">총 ${data.total_seconds}s · ${data.model}</div>`));
}
async function health(){
  try{const d=await (await fetch('/api/health')).json();
    document.getElementById('status').textContent=(d.loaded?'엔진 로드됨':'첫 요청 시 모델 로드')+' · '+d.model;
  }catch(e){document.getElementById('status').textContent='서버 연결 실패';}
}
// seed
addCrit('고객의 감성을 분류',[['positive','긍정/만족'],['neutral','중립'],['negative','부정/불만']]);
addCrit('어느 팀이 처리해야 하는가',[['billing','결제/환불'],['tech','기술지원'],['sales','영업']]);
document.getElementById('state').value='배송이 3일이나 늦었고 이중 청구까지 됐어요. 환불이 필요합니다. 정말 실망입니다.';
health();
</script>
</body></html>"""


if __name__ == "__main__":
    print(f"[server] SemIf UI on http://0.0.0.0:{PORT}  (Ctrl+C to stop)", flush=True)
    get_engine()  # 시작 시 미리 로드
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
