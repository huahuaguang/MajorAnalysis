from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

# load the pre-trained KMeans model and TF-IDF vectorizer
kmeans = load('models/kmeans_model.pkl')
tfidf = load('models/tfidf_vectorizer.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',prediction_text="Please type something😄")

@app.route('/predict', methods=['POST'])
def predict():
    # get the input text from the form
    input_text = request.form['input_value']
    
    # check if the input text is empty
    if not input_text.strip():
        return render_template('index.html', prediction_text='Please type something🙂')

    # clean the input text
    cleaned_text = clean_text(input_text)
    
    # check if the cleaned text is empty
    if not cleaned_text:
        return render_template('index.html', prediction_text='Please type some word😶')

    # transform the input text using the loaded TF-IDF vectorizer
    X_new = tfidf.transform([cleaned_text])
    
    # predict the cluster
    cluster = kmeans.predict(X_new)
    
    # return the prediction according to the cluster
    prediction_text = 'You are talking about computer science!🥰' if cluster[0] == 1 else 'What\'s a pity, you are not talking about computer science.🤐'
    
    return render_template('index.html', prediction_text=prediction_text)

def clean_text(text):
    import re
    if isinstance(text, str):
        # remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # remove numbers
        text = re.sub(r'\d+', '', text)
        # remove code blocks 
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # remove indentation from each line
        text = re.sub(r'\n\s*\n', '\n', text)  # combine multiple newlines
        text = re.sub(r'`[^`]*`', '', text)  # remove single line code snippets,using ` 
        text = re.sub(r'```[\s\S]*?```', '', text)  # remove multi-line code snippets, using ```
        # remove extra spaces and newlines
        text = ' '.join(text.split())
        return text
    return ''

if __name__ == '__main__':
    app.run(port=8003, debug=True)