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
        if st.button('Load Data'):
            st.write("Data Loaded")

            # Load data
            Tweets, labels = c.loadData()
            st.session_state.tweets = Tweets
            st.session_state.labels = labels
            st.write(len(st.session_state.tweets), "records loaded")
            st.write(len(st.session_state.labels), "labels are:", ', '.join(map(str, labels)))
            st.write(st.session_state.tweets.head(2))

            # Preprocess data
            Tweets["Cleaned"] = Tweets["Tweet"].apply(lambda x: c.PreprocessData(x))
            st.session_state.tweets = Tweets
            st.write(len(st.session_state.tweets), "records cleaned from URLs, emojis, and punctuation")
            st.write(st.session_state.tweets["Cleaned"].head(2))

            # Generate embeddings and save in session state
            embeddings = c.Bert_Emdedding(Tweets["Cleaned"].loc[0:1000].astype(str).tolist())
            st.session_state.embeddings = embeddings  # Save embeddings in session state
            st.write("Embedding Size......",st.session_state.embeddings.shape)


            st.write("Embedding Done...")


        else:
            st.write("No Data Loaded")

    except Exception as e:
        st.write("Stack Trace:", e)
        st.error(f"Error loading data: {e}")
with col2:
    st.write("**Bert + CNN Model**")
    try:
        # Ensure embeddings and tweets are loaded before allowing split
        test_size = st.number_input("Test Size", min_value=0.1, max_value=0.5, step=0.1)
        if st.button("Split Data") and st.session_state.embeddings is not None:
            # Ensure embeddings are non-empty and tweets are loaded correctly
            if len(st.session_state.embeddings) > 0:
                try:
                    # Ensure labels is accessible and used correctly
                    if isinstance(st.session_state.labels, list):
                        train_loader, test_loader, labels_n = c.TrainPreparing(
                            st.session_state.embeddings, st.session_state.tweets[st.session_state.labels].loc[0:1000], test_size
                        )
                        st.write("Data split completed")
                    else:
                        st.error("Labels must be a list of column names or indices for multi-label data.")

                except Exception as e:
                    st.error(f"Error in data splitting: {e}")
            else:

                st.error("No embeddings available to split. Please load and preprocess data first.")
    except Exception as e:
        st.write("Stack Trace:", e)
        st.error(f"Error Build, Training & Testing Model: {e}")