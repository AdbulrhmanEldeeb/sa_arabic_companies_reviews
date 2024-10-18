from flask import Flask,render_template,request 

app=Flask(__name__)


import joblib 

model=joblib.load('models\stacking_model.pkl')

vectorizer=joblib.load('models\count_vectorizer.pkl')

@app.route('/')
def main_func(): 
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_class(): 
    text = request.form['ali']
    text_vector=vectorizer.transform([text])

    class_value=model.predict(text_vector)[0]
    class_name= 'Positive' if class_value==1 else 'Negative' 
    return render_template('index.html',class_name=class_name )

if __name__=='__main__': 
    app.run(debug=True)


