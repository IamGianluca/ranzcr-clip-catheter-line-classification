from pathlib import Path

target_cols = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]

path = Path("/home/gianluca/git/kaggle/ranzcr/")

data_path = path / "data"
train_image_path = data_path / "train"
test_image_path = data_path / "test"
train_folds_fpath = data_path / "train_folds.csv"
sample_submission_fpath = data_path / "sample_submission.csv"

metrics_path = path / "metrics"
models_path = path / "models"
submissions_path = path / "subs"

params_fpath = path / "params.yaml"
