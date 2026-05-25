from flask import Flask, request, render_template
from pickle import load
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "decision_tree_classifier_default_42.sav"
model = load(open(MODEL_PATH, "rb"))
# Compatibility for models trained with older scikit-learn versions.
if not hasattr(model, "monotonic_cst"):
    model.monotonic_cst = None

class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])
        
        # Model expects: sepal length, sepal width, petal length, petal width.
        data = [[val4, val3, val2, val1]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict.get(prediction, prediction)
    else:
        pred_class = None
    
    return render_template("index.html", prediction = pred_class)