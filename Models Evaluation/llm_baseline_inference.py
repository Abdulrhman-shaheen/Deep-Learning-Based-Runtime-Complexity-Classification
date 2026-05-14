# LLM Baseline Inference Script (Ollama/Local LLM)
# Requirements: ollama (https://ollama.com/download), requests, tqdm
# Usage: Run ollama in the background, e.g. `ollama run codellama` or `ollama run starcoder`

import json
import requests
from tqdm import tqdm

OLLAMA_URL = 'http://localhost:11434/api/generate'  # Default ollama endpoint
MODEL = 'codellama'  # or 'starcoder', etc.
PROMPT_TEMPLATE = '''
You are an expert in analyzing Python code for computational complexity.
Given a Python function, classify its time complexity as one of the following:
- O(1)
- O(log n)
- O(n)
- O(n log n)
- O(n^2)

Respond with only the complexity class (e.g., O(n)), nothing else.

Python code:
{code}
'''

INPUT_PATH = 'Dataset/data.jsonl'  # Path to your code samples
OUTPUT_PATH = './llm_predictions.jsonl'


results = []
MAX_SAMPLES = 1000  # Limit for quick testing

with open(INPUT_PATH, 'r', encoding='utf-8') as fin, open(OUTPUT_PATH, 'w', encoding='utf-8') as fout:
    for idx, line in enumerate(tqdm(fin, desc='Classifying with LLM')):
        if idx >= MAX_SAMPLES:
            break
        item = json.loads(line)
        code = item['code']
        prompt = PROMPT_TEMPLATE.format(code=code)
        
        # Query ollama
        response = requests.post(OLLAMA_URL, json={
            'model': MODEL,
            'prompt': prompt,
            'stream': False
        })
        if response.status_code == 200:
            result = response.json()['response'].strip()
        else:
            result = 'ERROR'
        
        item['llm_prediction'] = result
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        results.append(item)

print(f'Done. Saved predictions to {OUTPUT_PATH}')
