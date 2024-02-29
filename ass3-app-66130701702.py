import pickle
import numpy as np
import streamlit as st
from sklearn.linear_model import Perceptron
# pickle.dump(model, open('per_model-xxx.sav', 'wb'))

## สร้างโปรแกรม ass3-app.py


model = pickle.load(open('per_model-66130701702.sav','rb'))

st.title("Iris Species Prediction using Perceptrin")

x1 = st.slider('Sepal Length', 0.0, 10.0, 0.1)
x2 = st.slider('Sepal Width', 0.0, 10.0, 0.1)
x3 = st.slider('Petal Length', 0.0, 10.0, 0.1)
x4 = st.slider('Petal Width', 0.0, 10.0, 0.1)

xnew = np.array([[x1,x2,x3,x4,]])#reshape(1,-1)

pred = model.predict(xnew)
st.write("## Prediction Result:")
st.write('Species:',pred[0])
