import json, joblib, pandas as pd                 # Tools: save text (json), save model (joblib), read tables (pandas)
from pathlib import Path                          # Nice way to work with file paths
from sklearn.model_selection import train_test_split  # Split data into train/test parts
from sklearn.compose import ColumnTransformer      # Mix different prep steps for columns
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Turn words to numbers; scale numbers
from sklearn.pipeline import Pipeline              # Chain steps together
from sklearn.linear_model import LogisticRegression # The model we train
from sklearn.metrics import roc_auc_score          # Score to check how good predictions are

DATA = Path("/Users/binhvo/PyCharmMiscProject/ai-ml-portfolio/churn-survival/data/telco_clean.csv")               # Where the cleaned data lives
ART = Path("artifacts"); ART.mkdir(exist_ok=True) # Make a folder to save results

def main():                                        # Our main program
    df = pd.read_csv(DATA)                         # Read the data into a table
    y = df["Churn"].values                         # Target: did the customer leave? (1/0)
    X = df.drop(columns=["Churn"])                 # Features: everything except the target

    num_cols = ["tenure","MonthlyCharges","TotalCharges"]         # Number columns
    cat_cols = [c for c in X.columns if c not in num_cols]        # The rest are words/categories

    pre = ColumnTransformer(                        # Build the data prep steps
        transformers=[
            ("num", StandardScaler(), num_cols),    # Numbers: make them on a similar scale
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),  # Words: turn into 0/1 flags
        ]
    )

    clf = Pipeline(steps=[("pre", pre),             # First: prep data
                         ("lr", LogisticRegression(max_iter=1000))])  # Then: train the model

    X_train, X_val, y_train, y_val = train_test_split(  # Split data: learn vs. check
        X, y, test_size=0.2, stratify=y, random_state=42)

    clf.fit(X_train, y_train)                       # Teach the model using training data
    p = clf.predict_proba(X_val)[:,1]               # Get “chance of churn” for each row
    auc = roc_auc_score(y_val, p)                   # Measure how well we rank churners
    print("Val ROC-AUC:", round(auc, 4))            # Show the score

    joblib.dump(clf, ART/"model.joblib")            # Save the trained model
    with open(ART/"columns.json","w") as f:         # Save column info for the API
        json.dump({"num": num_cols, "cat": cat_cols, "all": X.columns.tolist()}, f)
    print("Saved artifacts.")                       # Done!

if __name__ == "__main__":                          # If we run this file directly...
    main()                                          # ...start the program.
