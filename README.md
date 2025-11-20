
# Universal Credit Act 2025 — AI Agent

This repository contains an AI agent that extracts, summarises, and analyses the *Universal Credit Act 2025* PDF and provides:
- PDF text extraction
- 5–10 bullet summary
- Extraction of legislative sections to structured JSON
- Rule checks (6 checks) with pass/fail and evidence
- Streamlit UI for upload, review, and downloads

## Repo structure
```
universal-credit-ai-agent/
├── app/
│   ├── extract_text.py
│   ├── summarize.py
│   ├── extract_sections.py
│   ├── rule_checker.py
│   └── main.py
├── ui/
│   └── streamlit_app.py
├── data/
│   └── universal_credit_act_2025.pdf
├── output/
│   ├── extracted_text.txt
│   ├── summary.txt
│   ├── sections.json
│   └── rule_check_results.json
├── requirements.txt
├── README.md
└── .gitignore
```

## Quick start (local)
1. Clone the repo and place the provided `universal_credit_act_2025.pdf` under `data/` (or upload via UI).
2. Create a virtualenv and install deps:
```bash
python -m venv venv
source venv/bin/activate   # on Windows use venv\Scripts\activate
pip install -r requirements.txt
```
3. Put your OpenAI API key in a `.env` file or paste it into the Streamlit sidebar when running the UI.
4. Run the Streamlit UI:
```bash
streamlit run ui/streamlit_app.py
```

## Files of interest
- `app/extract_text.py` — extract text from PDF with `pdfplumber`.
- `app/summarize.py` — calls OpenAI to produce a concise bullet summary.
- `app/extract_sections.py` — asks the model to return a JSON with definitions, obligations, eligibility, etc.
- `app/rule_checker.py` — runs the six rule checks and returns structured results.
- `ui/streamlit_app.py` — complete Streamlit front-end for upload, extraction, summaries, rule checks, and downloads.

## Deliverables (what to submit)
1. GitHub repository (this repo)
2. JSON output files in `output/`
3. 2-minute video explaining approach (script included in /docs)

## Notes
- The OpenAI API is optional but recommended for high-quality summarisation & extraction.
- This project is prepared for the NIYAMR 48-hour assignment.
