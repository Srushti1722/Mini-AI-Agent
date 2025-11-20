
import streamlit as st
import pdfplumber
import json
import os
import re
from io import BytesIO
from typing import Optional
try:
    from openai import OpenAI
   
    OPENAI_INSTALLED = True
except Exception:
    OPENAI_INSTALLED = False
st.set_page_config(page_title="Universal Credit AI Agent", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .big-title { font-size:26px; font-weight:700; margin-bottom: 0.25rem; }
      .muted { color: #6c757d; font-size:12px; }
      .card { padding: 12px 16px; border-radius:8px; background:#f8f9fa; }
      .small { font-size:13px; }
      .mono { font-family: monospace; background:#fff; padding:6px; border-radius:4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n\n".join(pages).strip()

def save_output(filename: str, content, folder: str = "output") -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
    with open(path, mode, encoding="utf-8" if mode=="w" else None) as f:
        if mode == "w":
            f.write(content)
        else:
            f.write(content)
    return path

def heuristic_summary(text: str, max_bullets: int = 7) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "(no text)"
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    keywords = ["standard allowance", "LCWRA", "limited capability", "uplift", "CPI", "Secretary of State",
                "pre-2026", "severe conditions", "terminally ill", "assessment", "Northern Ireland", "ESA"]
    scored = []
    for s in sentences:
        s_lower = s.lower()
        score = sum(s_lower.count(k.lower()) * 3 for k in keywords)
        score += min(len(s.split()), 60) / 60.0
        scored.append((score, s.strip()))
    scored.sort(reverse=True, key=lambda x: x[0])
    bullets, used = [], set()
    for score, s in scored:
        if len(bullets) >= max_bullets:
            break
        key = s[:120]
        if key in used or score <= 0.1:
            continue
        used.add(key)
        if len(s) > 360:
            s = s[:360].rsplit(" ", 1)[0] + "..."
        bullets.append(f"- {s}")
    if not bullets:
        bullets = ["- " + text[:360] + ("..." if len(text) > 360 else "")]
    return "\n".join(bullets)

def heuristic_extract_sections(text: str) -> dict:
    lower = text.lower()
    sections = {k:"" for k in ["definitions","obligations","responsibilities","eligibility","payments","penalties","record_keeping"]}
    mapping = {
        "definitions": ["definition", "meaning of", "interpretation"],
        "eligibility": ["pre-2026", "eligible", "entitled", "claimant", "terminally ill", "severe conditions"],
        "payments": ["standard allowance", "uplift", "cpi", "lcwra", "esa ir", "allowance"],
        "obligations": ["must exercise", "must", "secure", "ensure"],
        "responsibilities": ["secretary of state", "department for communities", "responsible"],
        "penalties": ["penalt", "enforc", "assessment", "review", "withhold"],
        "record_keeping": ["assessment period", "benefit week", "continuous entitlement", "record", "evidence", "report"]
    }
    for k, kws in mapping.items():
        for kw in kws:
            if kw in lower:
                m = re.search(r"(.{0,240}\b" + re.escape(kw) + r"\b.{0,240})", text, flags=re.I|re.S)
                if m:
                    sections[k] = m.group(0).strip()
                    break
    return sections

def run_local_rule_checks(sections: dict) -> list:
    rules = [
        "Act must define key terms",
        "Act must specify eligibility criteria",
        "Act must specify responsibilities of the administering authority",
        "Act must include enforcement or penalties",
        "Act must include payment calculation or entitlement structure",
        "Act must include record-keeping or reporting requirements"
    ]
    flat = " ".join(sections.values()).lower()
    results = []
    checks = {
        rules[0]: lambda f: bool(re.search(r"\bdefinition|interpretation\b", f)) or bool(sections.get("definitions")),
        rules[1]: lambda f: bool(re.search(r"\beligible|entitled|claimant|pre-2026|severe conditions|terminally ill\b", f)) or bool(sections.get("eligibility")),
        rules[2]: lambda f: bool(re.search(r"\bsecretary of state|department for communities|responsible\b", f)) or bool(sections.get("responsibilities")),
        rules[3]: lambda f: bool(re.search(r"\bassessment|review|determin|enforc|penalt\b", f)) or bool(sections.get("penalties")),
        rules[4]: lambda f: bool(re.search(r"\bstandard allowance|uplift|cpi|amount|lcwra|esa\b", f)) or bool(sections.get("payments")),
        rules[5]: lambda f: bool(re.search(r"\bassessment period|benefit week|continuous entitlement|record|evidence|report\b", f)) or bool(sections.get("record_keeping")),
    }
    for r in rules:
        ok = checks[r](flat)
        evidence = None
        if ok:
            for kw in ["definition","eligible","secretary of state","assessment","cpi","assessment period","continuous entitlement","lcwra"]:
                m = re.search(r"([^\.\n]{0,120}\b" + re.escape(kw) + r"\b[^\.\n]{0,120}\.)", flat, flags=re.I)
                if m:
                    evidence = m.group(0).strip()
                    break
            if not evidence:
                evidence = "Keyword/context found in text (heuristic)."
            results.append({"rule": r, "status": "pass", "evidence": evidence, "confidence": 90})
        else:
            results.append({"rule": r, "status": "fail", "evidence": "No matching keywords/sections found (heuristic).", "confidence": 40})
    return results

def ai_chat_completion(prompt: str, api_key: str, model: str="gpt-4o-mini",
                       alt_model: str="gpt-3.5-turbo", max_tokens: int=1400, temperature: float=0.0) -> Optional[str]:
    if not OPENAI_INSTALLED:
        return None
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":"You are a helpful legal analyst assistant. Provide concise, structured outputs when requested."},
                {"role":"user","content":prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        try:
            return response.choices[0].message.content
        except Exception:
            return getattr(response.choices[0], "text", str(response))
    except Exception as e:
        err = str(e).lower()
        if "quota" in err or "insufficient_quota" in err or "429" in err or "rate limit" in err:
            try:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=alt_model,
                    messages=[
                        {"role":"system","content":"You are a helpful legal analyst assistant. Provide concise, structured outputs when requested."},
                        {"role":"user","content":prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                try:
                    return response.choices[0].message.content
                except Exception:
                    return getattr(response.choices[0], "text", str(response))
            except Exception as e2:
                return f"[AI error] fallback failed: {e2}"
        return f"[AI error] {e}"

def extract_json_from_text(maybe_text: str) -> Optional[dict]:
    if not maybe_text:
        return None
    try:
        return json.loads(maybe_text)
    except Exception:
        pass
    braces = [(maybe_text.find('{'), maybe_text.rfind('}')), (maybe_text.find('['), maybe_text.rfind(']'))]
    candidates = []
    for start, end in braces:
        if start != -1 and end != -1 and end > start:
            candidates.append(maybe_text[start:end+1])
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            try:
                fixed = c.replace("'", '"')
                return json.loads(fixed)
            except Exception:
                continue
    return None

# ---------- UI layout ----------
st.markdown('<div class="big-title">Universal Credit Act 2025 — Mini AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Upload the Act PDF, then run the pipeline. Use AI mode for higher-quality outputs.</div>', unsafe_allow_html=True)
st.write("")

# Sidebar - Settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API key (sk-...)", type="password", placeholder="Paste your key here (optional)")
    use_ai = st.checkbox("Use AI (OpenAI)", value=False)
    if use_ai and not OPENAI_INSTALLED:
        st.error("OpenAI SDK not installed. Run: pip install openai")
    st.markdown("**Model fallback:** gpt-4o-mini → gpt-3.5-turbo")
    st.markdown("**Outputs saved to**: `output/`")
    with st.expander("Quick help"):
        st.write("""
        1. Upload or use the bundled PDF.  
        2. Click *Extract text now* → *Summarize* → *Extract Sections* → *Run Rule Checks*.  
        3. Download JSON files from the Results section.
        """)

# Main columns
col_left, col_right = st.columns([2.2, 1])

with col_left:
    st.subheader("1) Input PDF")
    uploaded_file = st.file_uploader("Upload Universal Credit Act PDF", type=["pdf"], help="Or click 'Use bundled PDF' (if present).")
    use_bundled = st.button("Use bundled PDF (data/universal_credit_act_2025.pdf)")
    pdf_bytes = None
    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
    elif use_bundled:
        bundled = "data/universal_credit_act_2025.pdf"
        if os.path.exists(bundled):
            with open(bundled, "rb") as f:
                pdf_bytes = f.read()
        else:
            st.warning("Bundled PDF not found. Place it at data/universal_credit_act_2025.pdf or upload a file.")

    st.markdown("---")
    st.subheader("2) Text (editable)")
    ta_placeholder = st.empty()
    ta_val = st.session_state.get('extracted_text', "")
    user_text = st.text_area("Text to analyze (you can edit before sending to AI)", value=ta_val, height=320)

with col_right:
    st.subheader("Actions")
    btn_extract = st.button("Extract text now")
    btn_summarize = st.button("Summarize")
    btn_extract_sections = st.button("Extract Sections (structured)")
    btn_rule_checks = st.button("Run Rule Checks")

    st.markdown("---")
    st.markdown("**Status**")
    status_box = st.empty()
    status_box.info("Ready — upload PDF or use bundled file, then extract.")

def safe_update_status(msg: str, kind: str = "info"):
    if kind == "info":
        status_box.info(msg)
    elif kind == "success":
        status_box.success(msg)
    elif kind == "warning":
        status_box.warning(msg)
    elif kind == "error":
        status_box.error(msg)
    else:
        status_box.write(msg)

if btn_extract:
    if not pdf_bytes:
        safe_update_status("No PDF loaded. Upload or use bundled PDF.", "error")
    else:
        safe_update_status("Extracting text...", "info")
        try:
            extracted_text = extract_text_from_pdf_bytes(pdf_bytes)
            st.session_state['extracted_text'] = extracted_text
            save_output("extracted_text.txt", extracted_text)
            safe_update_status("Extraction complete — saved to output/extracted_text.txt", "success")
        except Exception as e:
            safe_update_status(f"Extraction failed: {e}", "error")

# Summarize
if btn_summarize:
    if not user_text.strip():
        safe_update_status("No text available for summary. Extract or paste text first.", "error")
    else:
        safe_update_status("Generating summary...", "info")
        summary_text = None
        if use_ai and api_key and OPENAI_INSTALLED:
            prompt = (
                "You are a concise legal summariser. Summarise the following Act in 5-10 bullet points focusing on: "
                "Purpose, Key definitions, Eligibility, Obligations, Enforcement elements, Payments. Produce only bullet points "
                "each starting with a dash (-). Text:\n\n" + user_text
            )
            ai_out = ai_chat_completion(prompt, api_key, model="gpt-4o-mini", max_tokens=1400)
            if ai_out and not ai_out.startswith("[AI error]"):
                summary_text = ai_out.strip()
            else:
                summary_text = heuristic_summary(user_text)
                if ai_out:
                    safe_update_status(f"AI fallback: {ai_out}", "warning")
        else:
            summary_text = heuristic_summary(user_text)
        st.session_state['summary'] = summary_text
        save_output("summary.txt", summary_text)
        safe_update_status("Summary saved to output/summary.txt", "success")

# Extract Sections
if btn_extract_sections:
    if not user_text.strip():
        safe_update_status("No text available for extraction. Extract or paste text first.", "error")
    else:
        safe_update_status("Extracting structured sections...", "info")
        sections_json = None
        if use_ai and api_key and OPENAI_INSTALLED:
            prompt = (
                "Extract the following from the Act text and return strictly valid JSON with keys: "
                "\"definitions\", \"obligations\", \"responsibilities\", \"eligibility\", \"payments\", \"penalties\", \"record_keeping\". "
                "For each key write concise text (short paragraphs). Return only JSON and nothing else.\n\nText:\n\n" + user_text
            )
            ai_out = ai_chat_completion(prompt, api_key, model="gpt-4o-mini", max_tokens=1800)
            parsed = extract_json_from_text(ai_out) if ai_out else None
            if parsed:
                sections_json = parsed
            else:
                sections_json = heuristic_extract_sections(user_text)
                if ai_out:
                    # keep a small snippet of AI output for debugging
                    sections_json["_ai_raw_snippet"] = ai_out[:1500]
                    safe_update_status("AI returned non-JSON; using heuristic fallback (AI raw kept).", "warning")
        else:
            sections_json = heuristic_extract_sections(user_text)
        st.session_state['sections_json'] = sections_json
        save_output("sections.json", json.dumps(sections_json, indent=2))
        safe_update_status("Sections saved to output/sections.json", "success")

# Run Rule Checks
if btn_rule_checks:
    sections_here = st.session_state.get('sections_json', None)
    if not sections_here and not user_text.strip():
        safe_update_status("No sections or text available — extract sections or paste text first.", "error")
    else:
        safe_update_status("Running rule checks...", "info")
        rule_results = None
        if use_ai and api_key and OPENAI_INSTALLED and sections_here:
            prompt = (
                "You are a rules engine. Given the following sections JSON, evaluate these rules:\n"
                "1. Act must define key terms\n"
                "2. Act must specify eligibility criteria\n"
                "3. Act must specify responsibilities of the administering authority\n"
                "4. Act must include enforcement or penalties\n"
                "5. Act must include payment calculation or entitlement structure\n"
                "6. Act must include record-keeping or reporting requirements\n\n"
                "Return strictly valid JSON: an array of objects with keys: rule, status (pass/fail), evidence (short), confidence (0-100).\n\nSections JSON:\n" + json.dumps(sections_here, indent=2)
            )
            ai_out = ai_chat_completion(prompt, api_key, model="gpt-4o-mini", max_tokens=1200)
            parsed = extract_json_from_text(ai_out) if ai_out else None
            if parsed:
                rule_results = parsed
            else:
                rule_results = run_local_rule_checks(sections_here if sections_here else heuristic_extract_sections(user_text))
                if ai_out:
                    # keep AI raw snippet for debugging
                    rule_results = {"local_results": rule_results, "_ai_raw_snippet": ai_out[:1200]}
                    safe_update_status("AI returned non-JSON for rule checks; used local checks (AI raw kept).", "warning")
        else:
            sections_here = sections_here if sections_here else heuristic_extract_sections(user_text)
            rule_results = run_local_rule_checks(sections_here)
        st.session_state['rule_results'] = rule_results
        save_output("rule_check_results.json", json.dumps(rule_results, indent=2))
        safe_update_status("Rule checks saved to output/rule_check_results.json", "success")

# ---------- Results display (collapsible) ----------
st.markdown("---")
st.header("Results & Downloads")

with st.expander("Extracted text (preview)"):
    if 'extracted_text' in st.session_state:
        st.write(st.session_state['extracted_text'][:8000])
        st.download_button("Download extracted_text.txt", data=st.session_state['extracted_text'], file_name="extracted_text.txt")
    else:
        st.info("No extracted text yet. Use 'Extract text now'.")

with st.expander("Summary"):
    if 'summary' in st.session_state:
        st.text_area("Summary", value=st.session_state['summary'], height=200)
        st.download_button("Download summary.txt", data=st.session_state['summary'], file_name="summary.txt")
    else:
        st.info("No summary yet.")

with st.expander("Sections JSON"):
    if 'sections_json' in st.session_state:
        st.json(st.session_state['sections_json'])
        st.download_button("Download sections.json", data=json.dumps(st.session_state['sections_json'], indent=2), file_name="sections.json")
    else:
        st.info("No sections extracted yet.")

with st.expander("Rule Check Results"):
    if 'rule_results' in st.session_state:
        st.write(st.session_state['rule_results'])
        st.download_button("Download rule_check_results.json", data=json.dumps(st.session_state['rule_results'], indent=2), file_name="rule_check_results.json")
    else:
        st.info("No rule checks run yet.")

st.markdown("")
st.caption("")
