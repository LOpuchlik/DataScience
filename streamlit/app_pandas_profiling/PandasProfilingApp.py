import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.datasets import load_iris

# Web App Title
st.markdown('''
# **Exploratory Data Analysis App using Pandas Profiling**
**Credit:** App built by [Lidia J. Opuchlik](https://github.com/LOpuchlik)
---
''')

# Upload CSV data
with st.sidebar.header('1. Upload your dataset(.csv file)'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Pandas Profiling Report
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv_file = pd.read_csv(uploaded_file)
        return csv_file
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example IRIS Dataset'):
        # Example data
        @st.cache
        def load_default():
            path = 'https://raw.githubusercontent.com/venky14/Machine-Learning-with-Iris-Dataset/master/Iris.csv'
            iris_dataset = pd.read_csv(path)
            return iris_dataset
        df = load_default()
        pr = ProfileReport(df, explorative=True)
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
