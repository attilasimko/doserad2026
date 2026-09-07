import os
from huggingface_hub import snapshot_download, login
from configs import default_data_path

login(os.getenv("HF_TOKEN"))
snapshot_download(
    repo_id="LMUK-RADONC-PHYS-RES/DoseRAD2026",
    repo_type="dataset",
    local_dir=default_data_path(),
)