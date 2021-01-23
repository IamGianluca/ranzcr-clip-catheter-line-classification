from pathlib import Path

path = Path("/home/gianluca/git/kaggle/ranzcr/")

data_path = path / "data"
train_img_path = data_path / "train"
test_img_path = data_path / "test"

metrics_path = path / "metrics"
models_path = path / "models"
submissions_path = path / "subs"

params_fpath = path / "params.yml"
