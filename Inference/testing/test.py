import json
import ast
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "baseline_models"

XGB_MODEL = MODEL_DIR / "xgboost_model.joblib"
XGB_ENCODER = MODEL_DIR / "xgboost_label_encoder.joblib"
MLP_MODEL = MODEL_DIR / "mlp_pipeline.joblib"
MLP_ENCODER = MODEL_DIR / "mlp_label_encoder.joblib"
FEATURE_INFO = MODEL_DIR / "feature_info.json"


def extract_features(code):
    tree = ast.parse(code)

    num_for = sum(isinstance(n, ast.For) for n in ast.walk(tree))
    num_while = sum(isinstance(n, ast.While) for n in ast.walk(tree))
    num_loops = num_for + num_while

    uses_sort = int(".sort(" in code or "sorted(" in code)
    has_break = int(any(isinstance(n, ast.Break) for n in ast.walk(tree)))
    has_continue = int(any(isinstance(n, ast.Continue) for n in ast.walk(tree)))
    uses_comprehension = int(any(isinstance(n, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)) for n in ast.walk(tree)))

    function_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    recursion_flag = 0
    num_recursive_calls = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in function_names:
                recursion_flag = 1
                num_recursive_calls += 1

    has_log_update = int(
        "/= 2" in code or "//= 2" in code or "*= 2" in code
        or "= n / 2" in code or "= n // 2" in code
    )

    has_early_return = int("return" in code)

    features = {
        "num_for": num_for,
        "num_while": num_while,
        "num_loops": num_loops,
        "max_loop_depth": 1 if num_loops else 0,
        "has_nested_loops": int(num_loops > 1),
        "has_log_update": has_log_update,
        "uses_sort": uses_sort,
        "recursion_flag": recursion_flag,
        "num_recursive_calls": num_recursive_calls,
        "uses_comprehension": uses_comprehension,
        "has_break": has_break,
        "has_continue": has_continue,
        "has_early_return": has_early_return,
        "uses_generator": uses_comprehension,
        "loop_unknown": 1,
    }

    return features


def prepare_input(code):
    with open(FEATURE_INFO, "r", encoding="utf-8") as f:
        feature_info = json.load(f)

    feature_columns = feature_info["feature_columns"]

    features = extract_features(code)
    df = pd.DataFrame([features])

    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def predict(model_choice, code):
    X = prepare_input(code)

    if model_choice == "1":
        model = joblib.load(XGB_MODEL)
        encoder = joblib.load(XGB_ENCODER)
    else:
        model = joblib.load(MLP_MODEL)
        encoder = joblib.load(MLP_ENCODER)

    pred = model.predict(X)
    result = encoder.inverse_transform(pred)[0]

    return result


def main():
    print("Choose model:")
    print("1 - XGBoost")
    print("2 - MLP")

    model_choice = input("Enter choice: ").strip()

    print("\nPaste your Python code.")
    print("When finished, type END on a new line.\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)

    code = "\n".join(lines)

    result = predict(model_choice, code)

    print("\nPrediction result:")
    print(result)


if __name__ == "__main__":
    main()