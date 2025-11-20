import openai, json

def extract_sections(text, api_key=None, model='gpt-4o-mini'):
    if api_key:
        openai.api_key = api_key
        prompt = (
            "Extract the following from this Act: Definitions, Obligations, Responsibilities, Eligibility, "
            "Payments/Entitlements, Penalties/Enforcement, Record-keeping/Reporting. Return results as JSON with keys: "
            "definitions, obligations, responsibilities, eligibility, payments, penalties, record_keeping.\n\nText:\n" + text
        )
        resp = openai.ChatCompletion.create(model=model, messages=[{'role':'user','content':prompt}])
        out = None
        try:
            out = resp.choices[0].message.content
        except:
            out = resp.choices[0].text
        # Attempt to parse JSON; if not parsable, return raw text under 'raw'
        try:
            return json.loads(out)
        except:
            return {'raw': out}
    else:
        return {'note': 'API key not provided — cannot extract structured sections.'}
