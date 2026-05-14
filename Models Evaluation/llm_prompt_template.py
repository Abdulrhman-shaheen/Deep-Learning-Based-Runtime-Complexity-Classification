# LLM Prompt Template for Complexity Classification

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
