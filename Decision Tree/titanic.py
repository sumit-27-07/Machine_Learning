import numpy as np
import pickle
import streamlit as st
with open(r'C:\Users\sumit\Desktop\Code\python\Machine Learning\Decision Tree\titanic.sav','rb') as f:
    mp=pickle.load(f)

def titanic_predict(input):
    input_data=np.asarray(input)
    input_data_reshape=input_data.reshape(1,-1)
    prediction=mp.predict(input_data_reshape)
    if(prediction[0]==0):
        return 'The person is not survived'
    else:
        return 'The person is survived'

def main():
    
    st.title('Titanic Survival Prediction App')

    Pclass=st.text_input('Pclass')
    Sex=st.text_input('Sex (Male:1 / Female:0)')
    Age=st.text_input('Age')
    Fare=st.text_input('Fare')

    data=''
    if st.button("Test Result"):
        data=titanic_predict([Pclass,Sex,Age,Fare])
    
    st.success(data)

if __name__=='__main__':
    main()