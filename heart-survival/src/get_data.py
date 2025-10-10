# src/get_data.py
from sklearn.datasets import load_breast_cancer
import pandas as pd
from pathlib import Path

def main():
    ds = load_breast_cancer(as_frame=True)  # có sẵn trong sklearn
    df = pd.concat([ds.data, ds.target.rename("target")], axis=1)

    Path("data").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/breast_cancer.csv", index=False)

    print("Saved to data/breast_cancer.csv")
    print("Shape:", df.shape)
    print("Positive rate (target=1):", df["target"].mean())
    print(df.head())

if __name__ == "__main__":
    main()
