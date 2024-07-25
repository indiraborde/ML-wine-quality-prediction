import pickle

from flask import Flask,request,app,url_for,render_template
import numpy as np

app=Flask(__name__)
## Load the model
model=pickle.load(open('model_wine.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]

    output=model.predict(np.array(data).reshape(1,-1))[0]
    
    if output == 1:
        return render_template("good.html",prediction_text="The wine quality is good")
    else:
        return render_template("bad.html",prediction_text="The wine quality is bad")



if __name__=="__main__":
    app.run(debug=True)
   
     
