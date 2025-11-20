import json

RULES = [
    "Act must define key terms",
    "Act must specify eligibility criteria",
    "Act must specify responsibilities of the administering authority",
    "Act must include enforcement or penalties",
    "Act must include payment calculation or entitlement structure",
    "Act must include record-keeping or reporting requirements"
]

def simple_rule_check(sections_json):
    results = []
    text = json.dumps(sections_json).lower()
    for r in RULES:
        # simple heuristics
        if 'definition' in text or 'definitions' in text:
            status = 'pass'
            confidence = 85
            evidence = 'Found "definition(s)" keyword in extracted sections.'
        else:
            status = 'fail'
            confidence = 40
            evidence = 'Keyword not found (heuristic).'
        results.append({'rule': r, 'status': status, 'evidence': evidence, 'confidence': confidence})
    return results
