# LLM Baseline Evaluation Script
# Compares LLM predictions to ground-truth and prints accuracy, confusion matrix, and per-class accuracy


import json
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = './llm_predictions.jsonl'



# Only allow standard classes, map everything else to 'other complexity'
STANDARD_CLASSES = ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n^2)', 'O(n^3)']
def normalize_complexity(s):
    if not isinstance(s, str):
        return 'other complexity'
    s = s.strip().replace(' ', '').replace('O(n²)', 'O(n^2)').replace('O(n³)', 'O(n^3)')
    s = s.replace('O(nlogn)', 'O(nlogn)').replace('O(nlog(n))', 'O(nlogn)')
    s = s.replace('O(n^2.)', 'O(n^2)').replace('O(n^3.)', 'O(n^3)')
    # Map to standard classes
    mapping = {
        'O(1)': 'O(1)',
        'O(logn)': 'O(log n)',
        'O(log n)': 'O(log n)',
        'O(n)': 'O(n)',
        'O(nlogn)': 'O(n log n)',
        'O(nlog n)': 'O(n log n)',
        'O(nlog(n))': 'O(n log n)',
        'O(nlog(n))': 'O(n log n)',
        'O(n^2)': 'O(n^2)',
        'O(n^3)': 'O(n^3)'
    }
    s = mapping.get(s, 'other complexity')
    return s

true_labels = []
pred_labels = []

with open(INPUT_PATH, 'r', encoding='utf-8') as fin:
    for line in fin:
        item = json.loads(line)
        true = normalize_complexity(item.get('time_complexity') or item.get('complexity'))
        pred = normalize_complexity(item.get('llm_prediction'))
        if true and pred and pred.startswith('O('):
            true_labels.append(true)
            pred_labels.append(pred)


# Print overall accuracy
acc = accuracy_score(true_labels, pred_labels)
print('Accuracy:', acc)

# Print confusion matrix and save as PNG
labels = sorted(list(set(true_labels) | set(pred_labels)))
cm = confusion_matrix(true_labels, pred_labels, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print('Confusion Matrix:')
print(cm_df)



# Plot both count and normalized confusion matrices side by side
import numpy as np
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Print overall accuracy
acc = accuracy_score(true_labels, pred_labels)
print('Accuracy:', acc)

# Only use standard classes for confusion matrix
labels = STANDARD_CLASSES + ['other complexity']
cm = confusion_matrix(true_labels, pred_labels, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print('Confusion Matrix:')
print(cm_df)

# Plot only the counts confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Counts)')
plt.colorbar()
tick_marks = range(len(labels))
plt.xticks(tick_marks, labels, rotation=45, ha='right', fontsize=8)
plt.yticks(tick_marks, labels, fontsize=8)
plt.ylabel('True label')
plt.xlabel('Predicted label')
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=7)
plt.tight_layout()
plt.savefig('llm_confusion_matrix.png', dpi=300)
plt.close()
print('Confusion matrix (counts only) saved as llm_confusion_matrix.png')

# Print LaTeX table for main classes (edit as needed)
main_classes = STANDARD_CLASSES + ['other complexity']
print('\n% ---------- TABLE ----------')
print('\\begin{table}[H]')
print('\\centering')
print('\\small')
print('\\begin{tabular}{|l|c|c|c|}')
print('\\hline')
print('\\textbf{Class} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-score} \\')
print('\\hline')
for cls in main_classes:
    if cls in report:
        p = report[cls]['precision']
        r = report[cls]['recall']
        f1 = report[cls]['f1-score']
        print(f'{cls} & {p:.2f} & {r:.2f} & {f1:.2f} \\')
    else:
        print(f'{cls} & 0.00 & 0.00 & 0.00 \\')
print('\\hline')
print('\\end{tabular}')
print('\\caption{LLM (CodeLlama) zero-shot classification results on the test subset}')
print('\\label{tab:llm_results}')
print('\\end{table}')
