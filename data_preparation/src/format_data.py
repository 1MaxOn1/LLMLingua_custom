import json
import os
from datasets import load_dataset

dataset = load_dataset("huuuyeah/meetingbank")
os.makedirs("results/meetingbank/origin/", exist_ok=True)
for split in dataset:
    data = []
    for idx, instance in enumerate(dataset[split]):
        temp = {}
        temp["idx"] = idx
        temp["prompt"] = instance["transcript"]
        temp["summary"] = instance["summary"]
        data.append(temp)
    with open(f"results/meetingbank/origin/meetingbank_{split}_formated.json", "w") as f:
        json.dump(data, f, indent=4)