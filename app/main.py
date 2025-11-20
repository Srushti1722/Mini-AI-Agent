from app.extract_text import extract_text
from app.summarize import summarize_text
from app.extract_sections import extract_sections
from app.rule_checker import simple_rule_check
import json, os

PDF_PATH = 'data/universal_credit_act_2025.pdf'
OUT_DIR = 'output'
os.makedirs(OUT_DIR, exist_ok=True)

def run_workflow(api_key=None):
    text = extract_text(PDF_PATH)
    with open(os.path.join(OUT_DIR,'extracted_text.txt'),'w',encoding='utf8') as f:
        f.write(text)

    summary = summarize_text(text, api_key=api_key)
    with open(os.path.join(OUT_DIR,'summary.txt'),'w',encoding='utf8') as f:
        f.write(summary)

    sections = extract_sections(text, api_key=api_key)
    with open(os.path.join(OUT_DIR,'sections.json'),'w',encoding='utf8') as f:
        f.write(json.dumps(sections, indent=2))

    rules = simple_rule_check(sections)
    with open(os.path.join(OUT_DIR,'rule_check_results.json'),'w',encoding='utf8') as f:
        f.write(json.dumps(rules, indent=2))

    return {'summary': summary, 'sections': sections, 'rules': rules}

if __name__ == '__main__':
    print('Run run_workflow(api_key=YOUR_KEY) to execute end-to-end.')
