import streamlit as st
from Classify import Classify
st.set_page_config(layout="wide")
# Streamlit UI
st.title("Emotion Detection")
# Split the down part into three vertical columns
col1, col2 = st.columns(2)
Tweets,labels=[],[]
embeddings = None
c = Classify()
with col1:
    st.write("**Loading + Preprocessing Data**")
    if st.button('Load Data'):
        st.write("Data Loaded")

        try:
            Tweets,labels = c.loadData()
            st.write(len(Tweets),"record loaded")
            st.write(len(labels),"labels are:",', '.join(labels))
            st.write(Tweets.head(2))
            Tweets["Cleaned"] = Tweets["Tweet"].apply(lambda x: c.PreprocessData(x))
            st.write(len(Tweets), "record Cleaned from URLs, Emojis, and Punctuation")
            st.write(Tweets["Cleaned"].head(2))
            Embeddings=c.Emdedding(Tweets["Cleaned"])
            st.write(len(Embeddings), "record Embedded")
            st.write(Embeddings[0])
        except Exception as e:
            st.error(e)
    else:
        st.write("No Data Loaded")

with col2:
    st.write("**Bert + CNN Model**")
    if(len(Embeddings)>0):
        test_size = st.number_input("Test Size", min_value=0.1, max_value=0.5, step=0.1)
        if st.button("Split Data"):
            train_loader,test_loader,labels_n = c.TrainPreparing(Embeddings,Tweets[labels],test_size)
            st.write("Data Splitted")
            model,criterion,optimizer = c.BuildModel(768,labels_n)
            st.write("Model Built")
            num_epochs = st.number_input("Epochs", min_value=10, max_value=50, step=5)
            model = c.TrainModel(model,criterion,optimizer,num_epochs, train_loader)
            st.write("Model Trained")
            st.write("Result of Testing Model:", c.TestModel(model,test_loader))
