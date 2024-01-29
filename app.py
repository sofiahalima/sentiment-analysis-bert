from flask import Flask, render_template, request, redirect
from nlp.classifier import Classify

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return render_template(
        "index.html",
        model_name="NA",
        text="NA",
        probability="NA",
        classification="NA"
    )


@app.route('/evaluate', methods=['POST'])
def evaluate():
    if request.method == "POST":
        text_area = request.form['text-content']
        model_name = request.form['model-name']
        print("text_area: ", text_area)
        model = Classify(model_name)

        model_name, text, probability, classification = model.evaluate(text_area)

        return render_template(
            "index.html",
            model_name=model_name,
            text=text,
            probability=probability,
            classification=classification
        )
    else:
        return render_template(
            "index.html",
            model_name=None,
            text=None,
            probability=None,
            classification=None
        )


if __name__ == '__main__':
    app.run()
