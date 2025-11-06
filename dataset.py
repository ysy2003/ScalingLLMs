# Design2Code
from datasets import load_dataset

ds = load_dataset("SALT-NLP/Design2Code-hf", split="train")
print(ds)
print(ds[0])

# Web2Code
from datasets import load_dataset

ds = load_dataset("MBZUAI/Web2Code")

print(ds)
print(ds["train"][0])

# Pix2Code
from datasets import load_dataset

ds = load_dataset("N0zomu/pix2code-data")
print(ds)
print(ds["train"][0])

# WebSight
# from datasets import load_dataset

# ds = load_dataset("HuggingFaceM4/WebSight", "v0.2")
# print(ds)
# print(ds["train"][0])

# Plotbench
from datasets import load_dataset

ds = load_dataset("JetBrains-Research/PandasPlotBench")
print(ds)

# Chart2Code
# download and upload dataset locally



# Download 

# from huggingface_hub import snapshot_download

# local_dir = "Design2Code"  # 你想存的本地目录名

# snapshot_download(
#     repo_id="SALT-NLP/Design2Code",
#     repo_type="dataset",
#     local_dir=local_dir,
#     local_dir_use_symlinks=False,  # 推荐关掉软链接，方便后面随便移动/打包
# )
# print("Dataset downloaded to", local_dir)