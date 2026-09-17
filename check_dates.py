import pandas as pd

val = pd.read_csv("data/processed/val_processed.csv")
test = pd.read_csv("data/processed/test_processed.csv")

for name, df in [("val", val), ("test", test)]:
    print(name, df["draw_local_datetime"].min(), "->", df["draw_local_datetime"].max())