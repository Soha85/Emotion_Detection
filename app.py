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

    if st.button('Load Data'):
        st.write("Data Loaded")

        try:
            # Load data
            Tweets, labels = c.loadData()
            st.session_state.tweets = Tweets
            st.session_state.labels = labels
            st.write(len(Tweets), "records loaded")
            st.write(len(labels), "labels are:", ', '.join(map(str, labels)))
            st.write(Tweets.head(2))

            # Preprocess data
            Tweets["Cleaned"] = Tweets["Tweet"].apply(lambda x: c.PreprocessData(x))
            st.session_state.tweets = Tweets
            st.write(len(Tweets), "records cleaned from URLs, emojis, and punctuation")
            st.write(Tweets["Cleaned"].head(2))

            # Generate embeddings and save in session state
            embeddings = c.Bert_Emdedding(Tweets["Cleaned"].astype(str).tolist())
            st.session_state.embeddings = embeddings  # Save embeddings in session state

        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        st.write("No Data Loaded")

with col2:
    st.write("**Bert + CNN Model**")

    # Ensure embeddings and tweets are loaded before allowing split
    if st.button("Split Data") and st.session_state.embeddings is not None:
        test_size = st.number_input("Test Size", min_value=0.1, max_value=0.5, step=0.1)

        # Ensure embeddings are non-empty and tweets are loaded correctly
        if len(st.session_state.embeddings) > 0:
            try:
                # Ensure labels is accessible and used correctly
                if isinstance(st.session_state.labels, list):
                    train_loader, test_loader, labels_n = c.TrainPreparing(
                        st.session_state.embeddings, st.session_state.tweets[st.session_state.labels], test_size
                    )
                    st.write("Data split completed")
                else:
                    st.error("Labels must be a list of column names or indices for multi-label data.")

            except Exception as e:
                st.error(f"Error in data splitting: {e}")
        else:
            st.error("No embeddings available to split. Please load and preprocess data first.")
