"""
web_demo.py — простой веб-интерфейс для демонстрации системы.

Запуск: py web_demo.py
Открой: http://localhost:5000
"""

import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from openai import OpenAI
from dotenv import load_dotenv
from pipeline.knowledge_graph import KnowledgeGraph
from retrieval.retriever import HybridRetriever, RetrievalMode

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

print("Loading knowledge graph...")
KG = KnowledgeGraph.load("results/knowledge_graph_final.json")
KG.compute_pagerank()
print(f"Graph loaded: {KG.stats()}")

RETRIEVER = HybridRetriever(KG, mode=RetrievalMode.HYBRID_RRF)


def answer_question(question: str) -> dict:
    entity_ids = RETRIEVER.retrieve(question, top_k=10)
    context = KG.get_context_for_entities(entity_ids)

    entities_info = []
    for eid in entity_ids:
        e = KG.entities.get(eid)
        if e:
            entities_info.append({
                "label": e.label,
                "type": e.entity_type,
                "degree": KG._graph.degree(eid) if eid in KG._graph else 0
            })

    prompt = f"""You are a medical information system based on a knowledge graph 
of Russian oncological clinical guidelines.
Answer based ONLY on the knowledge graph context below.
Be specific, structured, and comprehensive.

Context:
{context}

Question: {question}
Answer:"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=600,
    )
    return {
        "answer": resp.choices[0].message.content.strip(),
        "entities": entities_info,
        "context_size": len(entity_ids),
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HiRAG-Ontology — Medical RAG Demo</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f7; min-height: 100vh; }
header { background: #1d1d1f; color: white; padding: 16px 32px;
         display: flex; align-items: center; gap: 16px; }
header h1 { font-size: 18px; font-weight: 600; }
header span { font-size: 12px; color: #888; }
.badge { background: #534AB7; color: white; padding: 3px 10px;
         border-radius: 12px; font-size: 11px; }
.container { max-width: 900px; margin: 32px auto; padding: 0 20px; }
.card { background: white; border-radius: 12px; padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }
.card h2 { font-size: 15px; font-weight: 600; color: #1d1d1f;
           margin-bottom: 14px; }
.search-row { display: flex; gap: 10px; }
textarea { flex: 1; padding: 10px 14px; border: 1px solid #ddd;
           border-radius: 8px; font-size: 14px; resize: vertical;
           min-height: 72px; font-family: inherit; color: #1d1d1f; }
textarea:focus { outline: none; border-color: #534AB7; }
button { padding: 10px 24px; background: #534AB7; color: white;
         border: none; border-radius: 8px; font-size: 14px;
         cursor: pointer; font-weight: 500; white-space: nowrap;
         align-self: flex-end; transition: background 0.2s; }
button:hover { background: #3c35a0; }
button:disabled { background: #aaa; cursor: not-allowed; }
.examples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.ex-btn { padding: 5px 12px; border: 1px solid #ddd; border-radius: 20px;
          background: transparent; font-size: 12px; color: #555;
          cursor: pointer; transition: all 0.15s; }
.ex-btn:hover { border-color: #534AB7; color: #534AB7; }
#result { display: none; }
.answer-text { font-size: 14px; line-height: 1.7; color: #1d1d1f;
               white-space: pre-wrap; }
.entity-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.chip { padding: 3px 10px; border-radius: 12px; font-size: 11px;
        font-weight: 500; color: white; }
.chip.Condition     { background: #534AB7; }
.chip.Drug          { background: #D85A30; }
.chip.Procedure     { background: #1D9E75; }
.chip.LabTest       { background: #185FA5; }
.chip.DosageRegimen { background: #BA7517; }
.chip.Other         { background: #888780; }
.meta { display: flex; gap: 20px; margin-bottom: 14px; }
.meta-item { font-size: 12px; color: #888; }
.meta-item b { color: #1d1d1f; }
.loader { display: none; padding: 20px;
          text-align: center; color: #888; font-size: 14px; }
.stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.stat-box { background: #f5f5f7; border-radius: 8px; padding: 12px;
            text-align: center; }
.stat-box .num { font-size: 22px; font-weight: 700; color: #534AB7; }
.stat-box .lbl { font-size: 11px; color: #888; margin-top: 2px; }
</style>
</head>
<body>
<header>
  <h1>HiRAG-Ontology</h1>
  <span class="badge">Live Demo</span>
  <span>Medical RAG · Minzdrav Dataset · DeepSeek-Chat</span>
</header>

<div class="container">
  <div class="card">
    <h2>Knowledge Graph Statistics</h2>
    <div class="stats-grid">
      <div class="stat-box"><div class="num">2,314</div><div class="lbl">Entities</div></div>
      <div class="stat-box"><div class="num">2,346</div><div class="lbl">Relations</div></div>
      <div class="stat-box"><div class="num">0.773</div><div class="lbl">Cons(G)</div></div>
      <div class="stat-box"><div class="num">0.730</div><div class="lbl">Q(G)</div></div>
      <div class="stat-box"><div class="num">15.1%</div><div class="lbl">Dedup rate</div></div>
      <div class="stat-box"><div class="num">78</div><div class="lbl">Documents</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Ask a question about oncology</h2>
    <div class="search-row">
      <textarea id="question" 
        placeholder="E.g.: What is the treatment protocol for acute lymphoblastic leukemia?">What is the treatment protocol for acute lymphoblastic leukemia in children?</textarea>
      <button id="ask-btn" onclick="askQuestion()">Ask</button>
    </div>
    <div class="examples">
      <span style="font-size:12px;color:#999;align-self:center">Examples:</span>
      <button class="ex-btn" onclick="setQ(this)">How is Ph+ ALL managed?</button>
      <button class="ex-btn" onclick="setQ(this)">What drugs treat acute myeloid leukemia?</button>
      <button class="ex-btn" onclick="setQ(this)">How is AKR diagnosed?</button>
      <button class="ex-btn" onclick="setQ(this)">What is induction therapy for ALL?</button>
    </div>
  </div>

  <div class="loader" id="loader">Retrieving from knowledge graph and generating answer...</div>

  <div class="card" id="result">
    <h2>Answer</h2>
    <div class="meta">
      <div class="meta-item">Retrieved entities: <b id="ctx-size">—</b></div>
      <div class="meta-item">Retrieval mode: <b>Hybrid RRF</b></div>
      <div class="meta-item">Model: <b>DeepSeek-Chat</b></div>
    </div>
    <div class="answer-text" id="answer-text"></div>
    <div style="margin-top:16px;">
      <div style="font-size:12px;color:#888;margin-bottom:8px">Retrieved entities:</div>
      <div class="entity-chips" id="entity-chips"></div>
    </div>
  </div>
</div>

<script>
function setQ(btn) {
  document.getElementById('question').value = btn.textContent;
}

async function askQuestion() {
  const q = document.getElementById('question').value.trim();
  if (!q) return;

  document.getElementById('ask-btn').disabled = true;
  document.getElementById('result').style.display = 'none';
  document.getElementById('loader').style.display = 'block';

  try {
    const resp = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const data = await resp.json();

    document.getElementById('answer-text').textContent = data.answer;
    document.getElementById('ctx-size').textContent = data.context_size;

    const chips = document.getElementById('entity-chips');
    chips.innerHTML = '';
    data.entities.forEach(e => {
      const chip = document.createElement('span');
      chip.className = 'chip ' + e.type;
      chip.title = e.type + ' · degree: ' + e.degree;
      chip.textContent = e.label;
      chips.appendChild(chip);
    });

    document.getElementById('result').style.display = 'block';
  } catch(err) {
    alert('Error: ' + err.message);
  } finally {
    document.getElementById('ask-btn').disabled = false;
    document.getElementById('loader').style.display = 'none';
  }
}

document.getElementById('question').addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) askQuestion();
});
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"  [{args[0]}] {args[1]}")

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/ask":
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))
            question = body.get("question", "")

            print(f"\n  Question: {question[:80]}")
            try:
                result = answer_question(question)
                resp_body = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp_body)
            except Exception as e:
                err = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(err)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    port = 5000
    print(f"\n{'='*50}")
    print(f"HiRAG-Ontology Web Demo")
    print(f"Open: http://localhost:{port}")
    print(f"Stop: Ctrl+C")
    print(f"{'='*50}\n")
    server = HTTPServer(("localhost", port), Handler)
    server.serve_forever()
