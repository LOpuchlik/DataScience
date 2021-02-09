import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


path = ''
# Web App Title
st.markdown('''
# **Exploratory Data Analysis App**
### App built by [Lidia J. Opuchlik](https://github.com/LOpuchlik)
''')
st.write(" ")

# Upload CSV data
with st.sidebar.header('Read in your dataset (.csv file)'):
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input CSV file", type=["csv"])

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv_file = pd.read_csv(uploaded_file)
        return csv_file
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Dataset**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)
else:
    #st.info('Awaiting for choosing an example dataset.')
    st.write(" ")
    option = st.selectbox(
        'Choose example dataset',
        ('iris', 'titanic', 'boston_housing'))
    if option == 'iris':
        path = 'https://raw.githubusercontent.com/venky14/Machine-Learning-with-Iris-Dataset/master/Iris.csv'
    elif option == 'titanic':
        # st.write('You chose titanic dataset')
        path = 'https://raw.githubusercontent.com/LOpuchlik/DataScience/master/streamlit/app_pandas_profiling/titanic_dataset.csv'
    elif option == 'boston_housing':
        path = 'https://raw.githubusercontent.com/LOpuchlik/DataScience/master/streamlit/app_pandas_profiling/bostonHousing_dataset.csv'

    @st.cache
    def load_default():
        csv_file = pd.read_csv(path)
        return csv_file
    df = load_default()
    pr = ProfileReport(df, explorative=True)
    st.write(" ")
    st.write(" ")
    st.write(option, ' DataFrame')
    st.write(df)
    st.write('---')
    st.write('**Pandas Profiling Report**')
    st_profile_report(pr)
