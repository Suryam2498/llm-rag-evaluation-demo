"""Render the evaluation report as a single self-contained HTML page (no external assets)."""
from __future__ import annotations

from html import escape

from .run_eval import THRESHOLDS

LABELS = {
    "retrieval_hit_rate": ("Retrieval hit rate", "Expected source document was retrieved"),
    "mean_faithfulness": ("Faithfulness", "Answer claims supported by retrieved context"),
    "mean_correctness": ("Correctness", "Answer matches the reference answer"),
    "refusal_rate": ("Refusal rate", "Unanswerable questions correctly refused"),
    "injection_resistance": ("Injection resistance", "Hidden canary never leaked"),
}

CSS = """
:root{--bg:#f6f9fc;--card:#fff;--ink:#0f172a;--mute:#5b6b82;--line:#e3eaf2;--navy:#0B1F3A;--teal:#00C2A8;--ok:#0a8f5b;--okbg:#e3f6ee;--bad:#c0392b;--badbg:#fdeceb}
@media(prefers-color-scheme:dark){:root{--bg:#0b1220;--card:#121c30;--ink:#e6edf7;--mute:#92a4bf;--line:#223250;--okbg:#0f3a2c;--badbg:#431d1a;--ok:#4cd6a0;--bad:#ff8a7d}}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--ink);line-height:1.5}
header{background:linear-gradient(135deg,#0B1F3A,#12305a);color:#fff;padding:34px 22px}
.wrap{max-width:1040px;margin:0 auto}
h1{margin:0 0 6px;font-size:26px}h1 em{font-style:normal;color:var(--teal)}
.meta{color:#b8cde9;font-size:13px}
.verdict{display:inline-block;margin-top:14px;padding:6px 14px;border-radius:99px;font-weight:700;font-size:14px}
.verdict.ok{background:var(--okbg);color:var(--ok)}.verdict.bad{background:var(--badbg);color:var(--bad)}
main{padding:26px 22px 50px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:30px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
.card .n{font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:32px;font-weight:800;margin:4px 0}
.card .d{font-size:12px;color:var(--mute);min-height:34px}
.bar{height:8px;border-radius:99px;background:var(--line);overflow:hidden;position:relative;margin-top:8px}
.bar i{display:block;height:100%;border-radius:99px}
.bar b{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--ink);opacity:.55}
.ok .v{color:var(--ok)}.ok .bar i{background:var(--ok)}.bad .v{color:var(--bad)}.bad .bar i{background:var(--bad)}
.card.na .v{color:var(--mute)}
h2{font-size:18px;margin:0 0 12px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;font-size:14px}
th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
th{background:var(--bg);font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute)}
tr:last-child td{border-bottom:0}
.chip{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;font-weight:700}
.chip.ok{background:var(--okbg);color:var(--ok)}.chip.bad{background:var(--badbg);color:var(--bad)}
.type{font-size:12px;color:var(--mute)}.ans{color:var(--mute);font-size:13px;max-width:360px}
footer{color:var(--mute);font-size:12px;text-align:center;padding:0 22px 30px}
"""


def _row_passed(row: dict) -> bool:
    if row["type"] == "answerable":
        return bool(row["retrieval_hit"]) and row["faithfulness"] >= THRESHOLDS["mean_faithfulness"] and row["correctness"] >= THRESHOLDS["mean_correctness"]
    if row["type"] == "unanswerable":
        return bool(row["refused"])
    return bool(row["resisted"])


def _card(key: str, value: float | None) -> str:
    title, desc = LABELS[key]
    limit = THRESHOLDS[key]
    if value is None:
        return f'<div class="card na"><div class="n">{escape(title)}</div><div class="v">n/a</div><div class="d">{escape(desc)}</div></div>'
    state = "ok" if value >= limit else "bad"
    pct = max(0, min(100, value * 100))
    return (
        f'<div class="card {state}"><div class="n">{escape(title)}</div>'
        f'<div class="v">{value:.2f}</div><div class="d">{escape(desc)} (needs &ge; {limit:g})</div>'
        f'<div class="bar"><i style="width:{pct:.0f}%"></i><b style="left:{limit * 100:.0f}%"></b></div></div>'
    )


def _detail(row: dict) -> str:
    if row["type"] == "answerable":
        return f'retrieval {"hit" if row["retrieval_hit"] else "miss"} &middot; faithfulness {row["faithfulness"]:.2f} &middot; correctness {row["correctness"]:.2f}'
    if row["type"] == "unanswerable":
        return "refused" if row["refused"] else "answered (should refuse)"
    return "canary protected" if row["resisted"] else "canary leaked"


def render_html(report: dict) -> str:
    summary, rows, meta = report["summary"], report["rows"], report.get("meta", {})
    failures = [k for k, limit in THRESHOLDS.items() if summary.get(k) is not None and summary[k] < limit]
    verdict = (
        '<span class="verdict ok">All checks passed</span>'
        if not failures
        else f'<span class="verdict bad">{len(failures)} check(s) below threshold</span>'
    )
    meta_line = " &middot; ".join(
        escape(f"{label}: {meta[key]}")
        for key, label in [("model_id", "Model"), ("judge_model_id", "Judge"), ("generated_at", "Generated"), ("cases", "Cases")]
        if meta.get(key) not in (None, "")
    )
    cards = "".join(_card(k, summary.get(k)) for k in THRESHOLDS)
    body_rows = []
    for r in rows:
        ok = _row_passed(r)
        body_rows.append(
            f'<tr><td>{escape(str(r["id"]))}</td><td><span class="type">{escape(r["type"])}</span></td>'
            f'<td>{escape(r["question"])}</td><td class="ans">{escape(r.get("answer") or "-")}</td>'
            f'<td>{_detail(r)}</td><td><span class="chip {"ok" if ok else "bad"}">{"PASS" if ok else "FAIL"}</span></td></tr>'
        )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM &amp; RAG Evaluation Report</title><style>{CSS}</style></head>
<body>
<header><div class="wrap"><h1>LLM &amp; RAG <em>Evaluation Report</em></h1>
<div class="meta">{meta_line}</div>{verdict}</div></header>
<main><div class="wrap">
<div class="cards">{cards}</div>
<h2>Per-question results</h2>
<table><thead><tr><th>ID</th><th>Type</th><th>Question</th><th>Answer</th><th>Scores</th><th>Result</th></tr></thead>
<tbody>{"".join(body_rows)}</tbody></table>
</div></main>
<footer>Generated by AssureAI Labs evaluation harness. Small evaluation set and same-family judge: scores are indicative, not a benchmark.</footer>
</body></html>
"""
