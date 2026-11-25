# LLM & RAG Evaluation Demo

A small retrieval-augmented (RAG) support assistant on **Amazon Bedrock**, plus an automated evaluation harness that scores it for **retrieval quality, faithfulness, correctness, refusal behaviour and prompt-injection resistance**. Built by [AssureAI Labs](https://assureailabs.in) to show how I test AI applications.

The knowledge base is a fictional utility company ("Acme Utilities"), so there is no real client data here.

## What gets evaluated

| Check | How | Why it matters |
|---|---|---|
| Retrieval hit rate | Deterministic: did the expected source document come back? | Bad retrieval makes good answers impossible |
| Faithfulness | LLM-as-judge: are the answer's claims supported by the retrieved context? | Catches hallucination |
| Correctness | LLM-as-judge against a reference answer | Catches wrong or incomplete answers |
| Refusal rate | Deterministic: unanswerable questions must get the refusal sentence | Stops confident guessing |
| Injection resistance | Deterministic canary: the hidden system-prompt token must never leak | Basic prompt-injection / data-leak check |

Thresholds live in `rag_eval/run_eval.py`; the run exits non-zero if any are missed, so it can gate a CI pipeline.

## Project layout

```
rag_eval/        llm.py (Bedrock Converse + fake LLM), retriever.py, pipeline.py, metrics.py, run_eval.py
data/kb/         Knowledge base documents
data/eval_set.jsonl   Answerable, unanswerable and injection test cases
promptfoo/       Promptfoo regression and safety config
tests/           Offline unit tests (fake LLM, no AWS needed)
reports/         Generated evaluation reports
```

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows (use source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
pytest -q                                            # offline tests, no AWS required

copy .env.example .env                               # then set region and model
aws configure                                        # or export AWS_* environment variables
python -m rag_eval.run_eval                          # live evaluation on Bedrock, writes reports/
```

Promptfoo (needs Node.js):

```bash
npx promptfoo@latest eval -c promptfoo/promptfooconfig.yaml
npx promptfoo@latest view
```

## Design notes

- **The harness is tested without a model.** A fake LLM lets CI verify the scoring logic on every push, including that a deliberately bad assistant is caught.
- **Judge and generator are configurable separately** (`JUDGE_MODEL_ID`), because grading with a stronger model than the one under test gives more reliable scores.
- **Temperature 0** keeps runs repeatable. LLM judges are still noisy, so treat small score changes with caution.
- **Cost:** the evaluation set is small and uses a low-cost model; set an AWS Budget alert before running.

## Latest results

**Live evaluation report:** https://suryam2498.github.io/llm-rag-evaluation-demo/

Run on Amazon Bedrock (`us.amazon.nova-lite-v1:0`, us-east-1), 10 test cases:

| Metric | Score | Threshold |
|---|---|---|
| Retrieval hit rate | 1.0 | 0.8 |
| Faithfulness | 1.0 | 0.8 |
| Correctness | 0.967 | 0.7 |
| Refusal rate | 1.0 | 1.0 |
| Injection resistance | 1.0 | 1.0 |

**Limitations:** the evaluation set is small (10 cases) and the judge is the same model family as the generator, so these scores are optimistic. Next steps: a larger adversarial set and a stronger, separate judge model.

## Contact

AssureAI Labs: LLM/RAG evaluation, test automation, performance and security testing. hello@assureailabs.in
