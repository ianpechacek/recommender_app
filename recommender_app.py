import streamlit as st
import pandas as pd
import numpy as np
# import opendatasets as od
import subprocess
import os
# import zipfile

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# user = st.secrets["user"]
# key = st.secrets["key"]

dataset_name = "arashnic/book-recommendation-dataset"
zip_path = 'book-recommendation-dataset.zip'



# path_rating = 'kaggle/Ratings.csv'
# paths_books = 'kaggle/Books.csv'
path_rating = 'Ratings.csv'
paths_books = 'Books.csv'

    
record_column = 'Book-Title'
user_column = 'User-ID'
ratings_count_threshold = 8

def download_dataset_from_kaggle(dataset_name, zip_path):
    # Check if the dataset already exists
    if not os.path.exists(zip_path):
        st.info("Downloading dataset from Kaggle...")
        # Run the Kaggle CLI command to download the dataset
        subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_name, '--unzip'], check=True)
        st.success("Dataset downloaded successfully!")
        st.session_state.books_dataset = True
    else:
        st.info("Dataset already exists. Skipping download.")

    # if 'zip_file' not in st.session_state:
    #     # Open and read the zip file into memory
    #     with open(zip_path, 'rb') as f:
    #         file_content = f.read()  # Read the content of the zip file into memory
        
    #     # Store the file content in session state
    #     st.session_state.zip_file = file_content


def load_data(path_rating, paths_books):
    ratings = pd.read_csv(path_rating)
    books = pd.read_csv(paths_books)
    dataset = pd.merge(ratings, books, on=['ISBN'])
    return dataset

def null_values_drop(df):
    for column in df.columns:
        nulls = df[column].isnull().sum()
        print(f'Column {column} has {nulls} null values')
    df.dropna(axis=1, how='any', inplace=True, subset=None)


def search_similar_names(df, record_column, user_column, target_name, threshold=90):
    matches = []
    user_ids = [] 
    for index, record in df[record_column].items():
        # Skip non-string (e.g., NaN, None, numbers)
        if isinstance(record, str) and fuzz.partial_ratio(record.lower(), target_name.lower()) >= threshold:
            matches.append(record)
            user_ids.append(df[user_column].iloc[index])  # Get corresponding User-ID for the match
    return np.unique(np.array(matches)), np.unique(np.array(user_ids))


def matrix_transform(dataset, users, ratings_count_threshold=8):
    books_to_compare = dataset[dataset['User-ID'].isin(users)] \
        .groupby(['Book-Title']) \
        .filter(lambda x: len(x) >= ratings_count_threshold) \
        ['Book-Title'] \
        .unique()
    ratings_matrix = dataset[dataset['Book-Title'].isin(books_to_compare)\
                                                & dataset['User-ID'].isin(users)] \
        .groupby(['User-ID', 'Book-Title'])['Book-Rating'] \
        .mean() \
        .unstack()

    return ratings_matrix.fillna(0)

# ========================================
# ML models functions:

def cosine_similarity_matrix(ratings_matrix):
    cosine_sim_matrix = cosine_similarity(ratings_matrix.T)
    st.session_state.cosine_sim_df = pd.DataFrame(
    np.round(cosine_sim_matrix,3), 
    index= ratings_matrix.columns,  
    columns= ratings_matrix.columns 
    )

def correlation_matrix(ratings_matrix):
    corr_matrix = ratings_matrix.corr()
    st.session_state.corr_df = pd.DataFrame(
    np.round(corr_matrix,3), 
    index=ratings_matrix.columns,  # Book IDs
    columns=ratings_matrix.columns  # Book IDs
    )

def knn_matrix(ratings_matrix, selected_book):
    model = NearestNeighbors(metric='cosine')
    model.fit(ratings_matrix.T.values)
    distance, indice = model.kneighbors(ratings_matrix.T.loc[selected_book].values.reshape(1,-1), n_neighbors=6)

    # knn_df = ratings_matrix.corr()
    st.session_state.knn_df = pd.DataFrame({
    'title'   : ratings_matrix.T.iloc[indice[0]].index.values,
    'distance': distance[0]
    # 'similarity': - np.log(distance[0])
    }) \
    .sort_values(by='distance', ascending=True)\
    .reset_index(drop=True)
    

def jaccard_similarity_matrix():
    pass

def svm_matrix():
    pass

# =========================================
# Streamlit callbacks:

def load_data_callback():
    with st.sidebar:
        with st.spinner('Looking up your book'):
            if st.session_state.text_input:
                st.session_state.target_name=st.session_state.text_input
            st.session_state.dataset = load_data(path_rating, paths_books)
            null_values_drop(st.session_state.dataset)
            st.session_state.matches, st.session_state.users = search_similar_names(st.session_state.dataset, record_column, user_column, st.session_state.target_name)
        st.success("Done!")

def run_model_callback():
    with st.spinner('Calculating recommendations...'):
        st.session_state.selected_book=st.session_state.select_book
        st.session_state.ratings_matrix = matrix_transform(st.session_state.dataset, st.session_state.users )

        # build cosine similarity matrix
        cosine_similarity_matrix(st.session_state.ratings_matrix)

        # build correlation matrix
        correlation_matrix(st.session_state.ratings_matrix)

        # build knn df
        knn_matrix(st.session_state.ratings_matrix, st.session_state.selected_book)

# ==============================================================
# STREAMLIT
# ==============================================================
st.set_page_config(
    page_title="Book Recommender",
    page_icon=" 📚",
    layout="wide",
    initial_sidebar_state="expanded")

if 'books_dataset' not in st.session_state:
    download_dataset_from_kaggle(dataset_name, zip_path)


with st.sidebar:
    st.title('📚 Book Recommender')
    
    target_name = st.text_input('What book have you read lately?',
                                on_change=load_data_callback, key='text_input'
                                  )
    # find = st.button('Find!', on_click=load_data_callback, args=target_name )

    
    if 'target_name' in st.session_state:
        selected_book = st.selectbox('Select the book: ', options=st.session_state.matches, on_change=run_model_callback, key='select_book')
                                 
if 'selected_book' in st.session_state:
    with st.sidebar:
        st.write('Reload page for new search') 

    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.header('You might enjoy these books: ')
        st.bar_chart(st.session_state.cosine_sim_df[selected_book]\
        [st.session_state.cosine_sim_df[selected_book] < 1]
        .sort_values(ascending=False)
        .head(5)
        ,horizontal=True, height=400) 
  
    with col2:
        st.header('You might dislike these books: ')
        st.bar_chart(st.session_state.cosine_sim_df[selected_book]\
        [st.session_state.cosine_sim_df[selected_book] > 0]
        .sort_values(ascending=False)
        .tail(5)
        ,horizontal=True, height=400, color=[255,0,0])

    st.markdown(
    '<p style="text-align: center; color: grey; font-size: 20px;">Cosine similarity model</p>',
    unsafe_allow_html=True
    )
    

    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.header('You might enjoy these books: ')
        st.bar_chart(st.session_state.corr_df[selected_book]\
        [st.session_state.corr_df[selected_book] < 1]
        .sort_values(ascending=False)
        .head(5)
        ,horizontal=True, height=400) 
  
    with col2:
        st.header('You might dislike these books: ')
        st.bar_chart(st.session_state.corr_df[selected_book]\
        [st.session_state.corr_df[selected_book] > -0.8]
        .sort_values(ascending=False)
        .tail(5)
        ,horizontal=True, height=400, color=[255,0,0])

    st.markdown(
    '<p style="text-align: center; color: grey; font-size: 20px;">Correlation model</p>',
    unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.header('You might enjoy these books: ')
        st.bar_chart(
        data = st.session_state.knn_df[1:6],
        x = 'title',
        y = 'distance',
        # .head(5)
        horizontal=True, height=400) 
  

    st.markdown(
    '<p style="text-align: center; color: grey; font-size: 20px;">KNN model</p>',
    unsafe_allow_html=True
    )
# ==============================================================



# st.session_state



