import pandas as pd
import json

path = "/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/CaptionQA/annotations.jsonl"

# Each line is one JSON object
df = pd.read_json(path, lines=True)

print("shape:", df.shape)
print("columns:", df.columns.tolist())
print(df.head(3))

# One row as Series
row0 = df.iloc[0]
print("\nrow 0:\n", row0)

# One row as dict (better for nested fields)
row0_dict = row0.to_dict()
print("\nrow 0 as dict:\n", row0_dict)

# If you want to pretty-print nested annotation fields:
print("\npretty row0:\n", json.dumps(row0_dict, indent=2, ensure_ascii=False))
