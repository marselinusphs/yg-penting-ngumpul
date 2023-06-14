import numpy as np
import pickle
import streamlit as st

# creating a function for Prediction
def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return st.success("Anda tidak diabetes")
    else:
        return st.error("Anda diabetes")

if __name__ == '__main__':
    # loading the saved model
    loaded_model = pickle.load(open('./grid_model.sav', 'rb'))

    # giving a title
    st.title('Diabetes Prediction menggunakan Support Vector Machine')

    # getting the input data from the user
    pregnancies = st.text_input('Jumlah kehamilan')
    glucose = st.text_input('Glukosa')
    bloodPressure = st.text_input('Tekanan Darah')
    skinThickness = st.text_input('Ketebalan Kulit Trisep')
    insulin = st.text_input('Insulin')
    bmi = st.text_input('BMI')
    diabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    age = st.text_input('Usia')

    # creating a button for Prediction
    if st.button('Cek Hasil'):
        try:
            diagnosis = diabetes_prediction(
                [pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]
            )
        except:
            diagnosis = st.error("Inputan tidak valid")
