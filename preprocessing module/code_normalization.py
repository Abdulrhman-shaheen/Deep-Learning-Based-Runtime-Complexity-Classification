import ast

def normalize_indentation(code: str) -> str:
    tree = ast.parse(code)
    return ast.unparse(tree)


# Test

code1 = "def count_pairs(arr):\n    count = 0\n    for i in range(len(arr)):\n        for j in range(len(arr)):\n            if arr[i] < arr[j]:\n                count += 1\n    return count\n"
code2 = "def count_pairs(arr):\n count = 0\n for i in range(len(arr)):\n   for j in range(len(arr)):\n        if arr[i] < arr[j]:\n             count += 1\n return count\n"

normalized_code1 = normalize_indentation(code1)
normalized_code2 = normalize_indentation(code2)

print(normalized_code1 == normalized_code2)  # Should print True


code_a = '''def max_value(arr):
            m = arr[0]
            for x in arr:
                if x > m:
                    m = x
            return m'''



code_b = '''def max_value(arr):
    m = arr[0]
    for x in arr:
        if x > m:
            m = x
    return m'''


normalized_code3 = normalize_indentation(code_a)
normalized_code4 = normalize_indentation(code_b)

print(normalized_code3)
print(normalized_code4)

tree = ast.parse(code_a)
# i want to see tree structure
print(ast.dump(tree, indent=4))