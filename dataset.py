from datasets import load_dataset

ds = load_dataset("SALT-NLP/Design2Code-hf", split="train")
print(ds)
print(ds[0])
