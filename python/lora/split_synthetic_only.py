import json

synthetic_only = []
real_only = []

with open("data/mixed_train.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        if "synthetic_audio" in data["audio"]:
            synthetic_only.append(line)
        else:
            real_only.append(line)

with open("data/synthetic_only.jsonl", "w") as f:
    f.writelines(synthetic_only)

print(f"Created data/synthetic_only.jsonl with {len(synthetic_only)} samples")
