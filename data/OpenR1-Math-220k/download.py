# 或者直接在 Python 代码中使用
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="open-r1/OpenR1-Math-220k",
    repo_type="dataset",
    local_dir="/data/home/scyb494/data",
    local_dir_use_symlinks=False
)