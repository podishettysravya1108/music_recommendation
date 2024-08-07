from flask import Flask, request, render_template
import re

import tensorflow as tf
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import numpy as np
app = Flask(__name__)

model = load_model("model.h5")
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()


ps = PorterStemmer()



# Define your NLP model integration here
# You would replace this with your actual model code
def process_text(textInput):
    textInput = re.sub('[^a-zA-Z]',' ',textInput) # Removing special char.
    textInput = textInput.lower() # Convert capital letters into small letters
    textInput = textInput.split() # Split the input
    textInput = [ps.stem(w) for w in textInput if w not in set(stopwords.words('english'))] # Stemming & Stopwords
    textInput = ' '.join(textInput)
    textInput = cv.fit_transform([textInput]).toarray()
    # join words
    # Replace this with your NLP model code
    # For this example, we'll just return the input text
    return textInput


@app.route("/")
def home():
    return render_template("input.html")

@app.route("/submit" ,methods = ["POST"])
def index():
    new_texts = request.form['textInput']
    X_new = vectorizer.transform(new_texts)
    predictions = model.predict(X_new)
    predictions[0]
    textInput = process_text(textInput)
    x = np.array(textInput)
    pred = model.predict(x)
    print(pred)
    return render_template("output.html", predict = pred)

if __name__ == "__main__":
    app.run(debug=True,port = 1111)
