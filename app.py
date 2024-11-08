import streamlit as st
from Classify import Classify
st.set_page_config(layout="wide")
# Streamlit UI
st.title("Emotion Detection")
# Split the down part into three vertical columns
col1, col2 = st.columns(2)

with col1:
    if st.button('Load Data'):
        st.write("Data Loaded")
        c = Classify()
        Tweets,labels = c.loadData()
        st.write(len(Tweets),"record loaded")
        st.write(len(labels),"labels")
        st.write(labels)
    else:
        st.write("No Data Loaded")

with col2:
    st.write("Another Column")