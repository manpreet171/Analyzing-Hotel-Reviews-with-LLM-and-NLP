import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import Agent
from langchain_groq.chat_models import ChatGroq
from feedback_processor import FeedbackProcessor
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(page_title="Hotel Reviews Analysis Tool", layout="wide")

# Load environment variables
load_dotenv()

# Rest of your imports and code...

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama3-70b-8192"

# Initialize the language model
@st.cache_resource
def initialize_model():
    return ChatGroq(api_key=GROQ_API_KEY, model=MODEL_NAME)

model = initialize_model()

# Set the title and display the image
st.title("Hotel Reviews Analysis Tool")

# Use a relative path for the image
image_path = r'C:\Users\hp\Desktop\Projects\Hotel reviews (AI analysis)\banner.png'
if os.path.exists(image_path):
    st.image(image_path, use_column_width=True)
else:
    st.warning("Banner image not found. Please ensure it's in the correct location.")

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def display_visualizations(report):
    st.header("Visualizations")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(report['hotel_name'], report['rating']['mean'], color='blue')
    ax.set_xlabel('Hotel Name')
    ax.set_ylabel('Average Rating')
    ax.set_title('Average Rating per Hotel')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

def process_query(prompt, processor):
    if "summary" in prompt:
        return processor.get_summary()
    elif "common themes" in prompt:
        return processor.get_most_common_themes()
    elif "hotel summary" in prompt:
        hotel_name = prompt.split("for ")[-1]
        return processor.get_hotel_summary(hotel_name)
    elif "compare" in prompt:
        hotels = prompt.split("compare ")[-1].split(" and ")
        return processor.compare_hotels(hotels[0], hotels[1])
    elif "reviews by nationality" in prompt:
        nationality = prompt.split("for ")[-1]
        return processor.get_reviews_by_nationality(nationality)
    elif "reviews with rating" in prompt:
        rating = int(prompt.split("rating ")[-1])
        return processor.get_reviews_by_rating(rating)
    elif "improvement suggestions for" in prompt:
        hotel_name = prompt.split("for ")[-1]
        return processor.get_suggestions_for_improvement(hotel_name)
    elif "appreciate the most about" in prompt:
        hotel_name = prompt.split("about ")[-1]
        return processor.get_positive_aspects(hotel_name)
    elif "key insights" in prompt:
        return processor.get_key_insights()
    elif "top 10 hotels by average rating" in prompt:
        fig, ax = processor.plot_top_hotels()
        st.pyplot(fig)
        return None
    else:
        return None

def validate_input(prompt):
    if not prompt.strip():
        st.warning("Please enter a valid query.")
        return False
    return True

uploaded_file = st.sidebar.file_uploader("Upload Hotel Reviews CSV", type=["csv"])

if uploaded_file is not None:
    try:
        data = load_data(uploaded_file)
        st.write(data.head(3))

        processor = FeedbackProcessor(data)
        collected_data = processor.collect_feedback()

        agent = Agent(collected_data, config={"llm": model})
        prompt = st.text_input("What analysis would you like to run:")

        if st.button("Generate"):
            if prompt and validate_input(prompt):
                with st.spinner("Generating response..."):
                    response = process_query(prompt, processor)
                    if response is None:
                        response = agent.chat(prompt)
                    st.write(response)

        if st.button("Generate Report"):
            report = processor.report_feedback(collected_data)
            st.write(report)
            display_visualizations(report)

    except Exception as e:
        st.error(f"An error occurred: {e}")