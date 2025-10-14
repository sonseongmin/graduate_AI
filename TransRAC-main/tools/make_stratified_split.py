from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("./RepCountA/annotation/all_4class.csv")

train_df, valid_df = train_test_split(
    df, test_size=0.2, stratify=df["type"], random_state=42
)

train_df.to_csv("./RepCountA/annotation/train_4class.csv", index=False)
valid_df.to_csv("./RepCountA/annotation/valid_4class.csv", index=False)
