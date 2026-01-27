import pandas as pd

path = "/mnt/data-alpha-sg-02/team-agent/ai_glasses/datasets/ViRL39K/39Krelease.parquet"

df = pd.read_parquet(path)   # requires pyarrow or fastparquet
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print(df.head(10))            # first 3 rows

# See a single row (as a Series)
row0 = df.iloc[3:7]
print("\nrow 0:\n", row0)

# If you want it as a dict (better for nested/list fields):
print("\nrow 0 as dict:\n", row0.to_dict())
