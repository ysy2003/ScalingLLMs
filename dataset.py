# from datasets import load_dataset

# ds = load_dataset("SALT-NLP/Design2Code-hf", split="train")
# print(ds)
# print(ds[0])

from huggingface_hub import snapshot_download

local_dir = "Design2Code"  # 你想存的本地目录名

snapshot_download(
    repo_id="SALT-NLP/Design2Code",
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 推荐关掉软链接，方便后面随便移动/打包
)
print("Dataset downloaded to", local_dir)
