import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
import sklearn
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title='ASHS-Pred', page_icon='🌐', layout="wide")

# Create title and subtitle
html_temp = """
        <div style="background-color:teal">
        <h1 style="font-family:arial;color:white;text-align:center;">ASHS-Pred</h1>
        <h4 style="font-family:arial;color:white;text-align:center;">Artificial Intelligence Based Bioactivity Prediction Web-App</h4>       
        </div>
        <br>
        """
st.markdown(html_temp, unsafe_allow_html=True)

# Logo image
image = Image.open('Logo.png')
st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Application Details

This app allows you to predict the bioactivity of any molecule towards inhibiting AKT1, STAT3, SRC, HRSP90AA genes.

**Credits**.
- App built in `Python` + `Streamlit` by [Ratul Bhowmik, Ajay Manaithiya, Dr. Satarupa Acharjee]
- Descriptor calculated using [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Read the Paper]](https://doi.org/10.1002/jcc.21707).
---
""")

# loading the saved models
bioactivity_first_model = pickle.load(open('akt_model.pkl', 'rb'))
bioactivity_second_model = pickle.load(open('stat3_model.pkl', 'rb'))
bioactivity_third_model = pickle.load(open('hrsp90_model.pkl', 'rb'))
bioactivity_fourth_model = pickle.load(open('src_model.pkl', 'rb'))


# For hiding streamlit messages
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

st.sidebar.title("ASHS-Pred")
st.sidebar.header("Menu")
CB = st.sidebar.checkbox("Web Application Information")

if CB:
    st.title('Application Description')
    st.success(
        " This module of [**ASHS-Pred**](https://github.com/RatulChemoinformatics/QSAR) has been built to predict bioactivity and identify potent inhibitors against AKT1, STAT3, SRC, HRSP90AA genes using robust machine learning algorithms."
    )

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Artificial Intelligence assisted Machine Learning Bioactivity Prediction Models',
                           ['AKT1 prediction model using pubchemfingerprints',
                            'STAT3 prediction model using pubchemfingerprints',
                            'HRSP90AA prediction model using pubchemfingerprints',
                            'SRC prediction model using pubchemfingerprints'],
                           icons=['activity', 'activity', 'activity', 'activity'],
                           default_index=0)

# AKT1 prediction model using pubchemfingerprints
if selected == 'AKT1 prediction model using pubchemfingerprints':
    # page title
    st.title('Predict bioactivity of molecules against AKT1 using pubchemfingerprints')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_first_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://github.com/RatulChemoinformatics/QSAR/blob/main/predict.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('akt_model.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')

# STAT3 prediction model using pubchemfingerprints
elif selected == 'STAT3 prediction model using pubchemfingerprints':
    # page title
    st.title('Predict bioactivity of molecules against STAT3 using pubchemfingerprints')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_second_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('stat3_model.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')
            
            
# HRSP90AA prediction model using pubchemfingerprints
if selected == 'HRSP90AA prediction model using pubchemfingerprints':
    # page title
    st.title('Predict bioactivity of molecules against HRSP90AA using pubchemfingerprints')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_third_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('hrsp90_model.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')

# SRC prediction model using pubchemfingerprints
elif selected == 'SRC prediction model using pubchemfingerprints':
    # page title
    st.title('Predict bioactivity of molecules against SRC using pubchemfingerprints')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_fourth_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('src_model.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')
