# Import necessary libraries
import pandas as pd
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Function to precompute and save embeddings
@st.cache_data  # Cache the dataset and embeddings to reduce recomputation
def load_data_and_compute_embeddings(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Extract questions and answers from the dataset
    questions = data['Question'].values
    answers = data['Answer'].values
    
    # Initialize Sentence-BERT model and precompute question embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=True)
    
    return questions, answers, question_embeddings

# Load dataset and compute embeddings (replace with correct path)
file_path = r"C:\Users\abhis\OneDrive\Desktop\final_year_project\medicalQnA\train.csv"
questions, answers, question_embeddings = load_data_and_compute_embeddings(file_path)

# Define a dictionary of generic responses for greetings and other conversational inputs
generic_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hi there! How can I help you?",
    "hey": "Hey! How can I assist?",
    "how are you": "I'm a chatbot, but I'm here to help you with any medical questions!",
    "good morning": "Good morning! How can I assist you today?",
    "good afternoon": "Good afternoon! What can I do for you today?",
    "good evening": "Good evening! How can I assist you?",
    "thanks": "You're welcome! Feel free to ask more questions.",
    "thank you": "You're welcome! How else can I assist you?",
}

# Function to handle generic inputs with predefined responses
def get_generic_response(user_input):
    # Convert the input to lowercase to match with predefined responses
    user_input_lower = user_input.lower()
    
    # Check if the user input matches any generic response
    for greeting, response in generic_responses.items():
        if greeting in user_input_lower:
            return response
    
    return None

def get_best_match(user_question, model, question_embeddings, questions, answers, top_n=5):
    # Encode the user question
    user_question_embedding = model.encode(user_question, convert_to_tensor=True)
    
    similarity_scores = util.pytorch_cos_sim(user_question_embedding, question_embeddings).flatten()
    
    top_n_indices = similarity_scores.topk(k=top_n).indices.cpu().numpy()
    
    best_question = questions[top_n_indices[0]]  # Get the best match
    best_answer = answers[top_n_indices[0]]
    
    if similarity_scores[top_n_indices[0]] < 0.5:
        return "I am sorry, can you describe your question more precisely?."
    
    return best_answer

# Streamlit Interface
st.title("Medical Chatbot (Optimized with Precomputed BERT Embeddings)")
st.write("Ask any medical-related questions, and the chatbot will find the best-matching question from the dataset and provide an answer.")

# Input from user
user_input = st.text_input("Type your question here:")

if user_input:
    generic_response = get_generic_response(user_input)
    if generic_response:
        st.write("**Answer**: ", generic_response)
    else:
        response = get_best_match(user_input, SentenceTransformer('all-MiniLM-L6-v2'), question_embeddings, questions, answers)
        
        # Display the response
        st.write("**Answer**: ", response)
