# Machine Learning-Based Runtime Complexity Classification

> Predicting asymptotic runtime complexity from source code using machine learning — without executing the program.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

## 🎯 Core Goal

Build ML models that predict the **asymptotic runtime class** from source code only, without executing the program:

The project compares **different code representations** and **model architectures** on the same task.



## 📊 Dataset

- **4000+ Python code samples** with ground-truth complexity labels
- Randomized variable names, constants, and formatting
- Same dataset used for all model comparisons

### Program Templates

| Pattern | Examples |
|---------|----------|
| Single loops | `for i in range(n)` |
| Nested loops | `for i in range(n): for j in range(n)` |
| Logarithmic loops | `while n > 0: n //= 2` |
| Recursion | Fibonacci, tree traversal |
| Built-in operations | `sorted()`, `list.sort()` |


## 🔬 Approaches (Parallel Tracks)

### Track 1 — Static Feature Baseline

#### Representation
Hand-engineered features extracted from AST:

| Feature Category | Features |
|-----------------|----------|
| Loop Analysis | `num_loops`, `num_for`, `num_while`, `max_loop_depth`, `has_nested_loops` |
| Loop Bounds | `loop_bound_type` (constant/linear/log), `has_log_update` |
| Recursion | `recursion_flag`, `num_recursive_calls` |
| Control Flow | `has_break`, `has_continue`, `has_early_return`, `num_return` |
| Data Structures | `uses_list`, `uses_dict`, `uses_set`, `uses_sort` |

#### Models
- **Gradient Boosted Trees** (XGBoost/LightGBM)
- **Small MLP** (Multi-Layer Perceptron)

#### Purpose
- Interpretable baseline
- Demonstrates value of classical static analysis + ML

### Track 2 — Program as Graph (GNN)

#### Representation
Convert code into a graph:
- **Nodes**: loops, conditionals, variables, function calls
- **Edges**: nesting relationships, control flow, data dependency

#### Model
- Graph Neural Network (GCN / GIN)
- Graph-level classification

#### Purpose
- Capture structural patterns (nested loops) naturally
- Reduce reliance on manual feature design


### Track 3 — Token Sequence Model

#### Representation
Custom tokenizer for Python code:
- Preserves **keywords** (`for`, `while`, `if`, `def`)
- Preserves **builtins** (`sorted`, `range`, `len`)
- Normalizes **variables** → `<VAR>`
- Normalizes **numbers** → `<NUM>`
- Normalizes **strings** → `<STR>`
- Tracks **indentation** → `<INDENT>`

#### Model
**Transformer Encoder** (4 layers, 8 heads, 256-dim)

```
Code → Tokenizer → Embedding → Positional Encoding → Transformer Encoder → Classifier → Complexity
```

#### Purpose
- Test whether sequential models can infer complexity
- Compare structure-aware vs order-based learning


### Track 4 — LLM Baseline

- Frozen code LLM
- Zero-shot or prompt-based classification
- Comparison



