import joblib
import json
import ast
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent / "baseline_models"
CODE_TEST   = Path(__file__).resolve().parent / "code_test.py"

MODEL_PATHS = {
    "xgboost": BASE_DIR / "xgboost_model.joblib",
    "mlp":     BASE_DIR / "mlp_pipeline.joblib",
}
ENCODER_PATH      = BASE_DIR / "label_encoder.joblib"
FEATURE_INFO_PATH = BASE_DIR / "feature_info.json"


# ── Feature extraction ───────────────────────────────────────────────────────
def extract_features(source_code: str) -> dict:
    """
    Extract the same static AST features the training pipeline used.
    Returns a dict matching the original feature columns.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        raise ValueError(f"Could not parse code_test.py: {e}")

    features = {
        "num_loops":          0,
        "num_for":            0,
        "num_while":          0,
        "max_loop_depth":     0,
        "has_nested_loops":   0,
        "loop_bound_type":    "unknown",   # will be one-hot encoded later
        "has_log_update":     0,
        "uses_sort":          0,
        "recursion_flag":     0,
        "num_recursive_calls":0,
        "has_break":          0,
        "has_continue":       0,
        "has_early_return":   0,
        "uses_comprehension": 0,
        "uses_generator":     0,
    }

    func_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    def _loop_depth(node, depth=0):
        """Recursively find maximum loop nesting depth."""
        max_d = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                max_d = max(max_d, _loop_depth(child, depth + 1))
            else:
                max_d = max(max_d, _loop_depth(child, depth))
        return max_d

    for node in ast.walk(tree):
        # Loops
        if isinstance(node, ast.For):
            features["num_loops"] += 1
            features["num_for"]   += 1
        if isinstance(node, ast.While):
            features["num_loops"] += 1
            features["num_while"] += 1

        # Break / continue / return
        if isinstance(node, ast.Break):
            features["has_break"] = 1
        if isinstance(node, ast.Continue):
            features["has_continue"] = 1
        if isinstance(node, ast.Return):
            features["has_early_return"] = 1

        # Comprehensions & generators
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            features["uses_comprehension"] = 1
        if isinstance(node, ast.GeneratorExp):
            features["uses_generator"] = 1

        # Sorting
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name in ("sort", "sorted"):
                features["uses_sort"] = 1

        # Recursion
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in func_names:
                features["recursion_flag"]      = 1
                features["num_recursive_calls"] += 1

        # Logarithmic update patterns (//= 2, >>= 1, n = n // 2, etc.)
        if isinstance(node, ast.AugAssign):
            if isinstance(node.op, (ast.FloorDiv, ast.RShift)):
                features["has_log_update"] = 1
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(node.value, ast.BinOp) and isinstance(
                    node.value.op, (ast.FloorDiv, ast.RShift)
                ):
                    features["has_log_update"] = 1

    # Max loop depth & nested flag
    features["max_loop_depth"] = _loop_depth(tree)
    if features["max_loop_depth"] >= 2:
        features["has_nested_loops"] = 1

    # Simple heuristic for loop_bound_type
    if features["has_log_update"]:
        features["loop_bound_type"] = "log"
    elif features["num_loops"] > 0:
        features["loop_bound_type"] = "linear"
    else:
        features["loop_bound_type"] = "unknown"

    return features


def build_feature_vector(features: dict, feature_columns: list) -> pd.DataFrame:
    """One-hot encode loop_bound_type and align to training feature columns."""
    df = pd.DataFrame([features])
    df = pd.get_dummies(df, columns=["loop_bound_type"], prefix="loop")
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Load shared artefacts
    le           = joblib.load(ENCODER_PATH)
    feature_info = json.loads(FEATURE_INFO_PATH.read_text(encoding="utf-8"))
    feature_cols = feature_info["feature_columns"]

    # 2. Ask user which model to use
    print("\n=== Time Complexity Predictor ===")
    print("Available models:")
    print("  [1] XGBoost")
    print("  [2] MLP")
    choice = input("\nChoose a model (1 or 2): ").strip()

    if choice == "1":
        model_key  = "xgboost"
        model_name = "XGBoost"
    elif choice == "2":
        model_key  = "mlp"
        model_name = "MLP"
    else:
        print("Invalid choice. Exiting.")
        return

    model = joblib.load(MODEL_PATHS[model_key])
    print(f"\nLoaded model: {model_name}")

    # 3. Read and parse code_test.py
    source_code = CODE_TEST.read_text(encoding="utf-8")
    print(f"\nAnalysing: {CODE_TEST.name}")
    print("-" * 40)
    print(source_code.strip())
    print("-" * 40)

    # 4. Extract features
    features = extract_features(source_code)
    X        = build_feature_vector(features, feature_cols)

    # 5. Predict
    if model_key == "xgboost":
        y_enc       = model.predict(X)
        prediction  = le.inverse_transform(y_enc)[0]
        probas      = model.predict_proba(X)[0]
    else:  # mlp pipeline already includes scaler
        y_enc       = model.predict(X)
        prediction  = le.inverse_transform(y_enc)[0]
        probas      = model.predict_proba(X)[0]

    # 6. Display result
    print(f"\n{'='*40}")
    print(f"  Model      : {model_name}")
    print(f"  Prediction : {prediction}")
    print(f"{'='*40}")

    print("\nClass probabilities:")
    for cls, prob in sorted(zip(le.classes_, probas), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<12} {prob:.3f}  {bar}")


if __name__ == "__main__":
    main()