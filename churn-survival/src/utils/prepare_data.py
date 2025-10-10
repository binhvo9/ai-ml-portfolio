import pandas as pd                 # We use pandas to read and work with tables (spreadsheets).
from pathlib import Path            # Path helps us handle file paths in a clean way.

RAW = Path("/Users/binhvo/PyCharmMiscProject/ai-ml-portfolio/churn-survival/data/telco.csv")        # This is where the raw (original) CSV file lives.
CLEAN = Path("/Users/binhvo/PyCharmMiscProject/ai-ml-portfolio/churn-survival/data/telco_clean.csv")# This is where we will save the cleaned file.

def run():                          # This is the main function that does all the work.
    df = pd.read_csv(RAW)           # Read the raw CSV into a table called df.

    # Make "TotalCharges" a number. If a value is blank or weird, turn it into NaN (missing).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop any rows where "TotalCharges" is missing, so our math won’t break.
    df = df.dropna(subset=["TotalCharges"])

    # Turn the target "Churn" into 1/0: "yes" -> 1 (churned), anything else -> 0 (not churned).
    df["Churn"] = (df["Churn"].str.lower() == "yes").astype(int)

    # Pick the columns we want to keep. This makes the data smaller and focused.
    keep = ["gender","SeniorCitizen","Partner","Dependents","tenure",
            "PhoneService","MultipleLines","InternetService","OnlineSecurity",
            "OnlineBackup","DeviceProtection","TechSupport","Contract",
            "PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges","Churn"]

    df = df[keep]                   # Keep only those chosen columns.

    df.to_csv(CLEAN, index=False)   # Save the cleaned table to a new CSV (no row numbers).
    print(f"Saved {CLEAN} with shape {df.shape}")  # Tell us where it was saved and its size.

if __name__ == "__main__":          # If we run this file directly (not imported)...
    run()                           # ...run the main function.
