import sys, joblib
import numpy as np

# Usage:
# python predict.py 5.1 3.5 1.4 0.2
# (sepal_length sepal_width petal_length petal_width)

if len(sys.argv) != 5:
    print("Usage: python predict.py <sepal_len> <sepal_wid> <petal_len> <petal_wid>")
    sys.exit(1)

features = np.array([[float(sys.argv[1]), float(sys.argv[2]),
                      float(sys.argv[3]), float(sys.argv[4])]])

model = joblib.load("../artifacts/iris_svc_pipeline.joblib")
pred = model.predict(features)[0]
labels = ["setosa", "versicolor", "virginica"]
print("Prediction:", labels[pred])
