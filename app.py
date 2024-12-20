import pandas as pd
import streamlit as st
from Classify import Classify

# Set page config
st.set_page_config(layout="wide")

# Streamlit UI
st.title("Emotion Detection")

# Split the down part into two vertical columns
col1, col2 = st.columns(2)

# Initialize session state variables
if "tweets" not in st.session_state:
    st.session_state.tweets = None
    st.session_state.labels = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# Initialize Classify instance
c = Classify()

with col1:
    st.write("**Loading + Preprocessing Data**")
    try:
        num = st.number_input("No. of Tweets", min_value=500, max_value=6000, step=500)
        if st.button('Load Data'):
            # Load data
            Tweets, labels = c.loadData()
            Tweets=Tweets.loc[0:num]
            st.write("Data Loaded")
            st.session_state.tweets = Tweets
            st.session_state.labels = labels
            st.write(len(st.session_state.tweets), "records loaded")
            st.write(len(st.session_state.labels), "labels are:", ', '.join(map(str, st.session_state.labels)))
            st.write(st.session_state.tweets["Tweet"].head(2))

            # Preprocess data
            Tweets["Cleaned"] = Tweets["Tweet"].apply(lambda x: c.PreprocessData(x))
            st.session_state.tweets = Tweets
            st.write(len(st.session_state.tweets), "records cleaned from URLs, emojis, and punctuation")
            st.write(st.session_state.tweets["Cleaned"].head(2))

            # Generate embeddings and save in session state
            tweets_embeddings = c.Bert_Emdedding(Tweets["Cleaned"].astype(str).tolist()).cpu().numpy()
            st.session_state.tweets_embeddings = tweets_embeddings  # Save embeddings in session state
            st.write("Embedding Size......", st.session_state.tweets_embeddings.shape)
            print(type(tweets_embeddings))

            st.write("Embedding Done...")


        elif st.session_state.tweets is not None:
            st.write("Data Loaded")
            st.write(len(st.session_state.tweets), "records loaded")
            st.write(len(st.session_state.labels), "labels are:", ', '.join(map(str, st.session_state.labels)))
            st.write(st.session_state.tweets["Tweet"].head(2))
            st.write(len(st.session_state.tweets), "records cleaned from URLs, emojis, and punctuation")
            st.write(st.session_state.tweets["Cleaned"].head(2))
            st.write("Embedding Size......", st.session_state.tweets_embeddings.shape)
            st.write("Embedding Done...")
        else:
            st.write("No Data Loaded")

    except Exception as e:
        st.write("Stack Trace:", e)
        st.error(f"Error loading data: {e}")
with col2:
    st.write("**Multi label Model Testing**")
    selected_website = st.selectbox("Select a model to classify after Bert Embedding", ['CNN', 'LSTM','Transformer'])
    test_size = st.number_input("Test Size", min_value=0.1, max_value=0.5, step=0.1)
    num_epochs = st.number_input("Epochs Size", min_value=2, max_value=50, step=2)
    batch_size = st.number_input("Batch Size", min_value=32, max_value=500, step=2)
    if st.button("Split, Build, Train & Test") and st.session_state.tweets_embeddings is not None:
        train_loader, test_loader, val_loader, labels_n = c.TrainPreparing(
            st.session_state.tweets_embeddings, st.session_state.tweets[st.session_state.labels].values, batch_size,
            test_size)
        st.write("Data split completed")
        model, criterion, optimizer = None,None,None
        if selected_website == 'CNN':
            model,criterion,optimizer = c.BertCNNBuildModel(st.session_state.tweets_embeddings.shape[1],len(st.session_state.labels))
        elif selected_website == 'Transformer':
            model, criterion, optimizer = c.TransformerBuildModel(st.session_state.tweets_embeddings.shape[1],len(st.session_state.labels))
        else:
            model, criterion, optimizer = c.LSTMBuildModel(st.session_state.tweets_embeddings.shape[1],128,
                                                                  len(st.session_state.labels))
        st.write("Model Built")
        st.write(model)
        model, train_loss, val_loss, train_acc, val_acc = c.TrainModel(model, criterion, optimizer, num_epochs, train_loader,
                                                    val_loader)
        st.write("Model Trained")
        st.write(c.TestModel(model, test_loader,st.session_state.labels))
        c.plot_curves(train_loss, val_loss, train_acc, val_acc)

