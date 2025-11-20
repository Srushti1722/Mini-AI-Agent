import os
import openai

def summarize_text(full_text, api_key=None, model='gpt-4o-mini'):
    if api_key:
        openai.api_key = api_key
        prompt = "Summarize the following Act in 5-10 bullet points focusing on: Purpose, Key definitions, Eligibility, Obligations, Enforcement elements.\n\n" + full_text
        resp = openai.ChatCompletion.create(model=model, messages=[{'role':'user','content':prompt}])
        # adapt to response shape
        try:
            return resp.choices[0].message.content
        except:
            return resp.choices[0].text
    else:
        # fallback heuristic summary
        return '(fallback) OpenAI API key not provided. Provide text manually or set API key.'
