import streamlit as st
import requests

st.set_page_config(page_title="Movie Sentiment AI", page_icon="🍿")

st.title("🍿 IMDB Movie Review Classifier")
st.markdown("This app uses **Spark ML** for prediction and **FastAPI** as the backend.")

review_text = st.text_area("Type your movie review below:", placeholder="The movie was...")

if st.button("Analyze Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner('Asking Spark for the answer...'):
            try:
                # Send request to FastAPI
                response = requests.post(
                    "http://localhost:8000/predict", 
                    params={"review": review_text}
                )
                data = response.json()
                
                label = data["sentiment"]
                
                if label == "Positive":
                    st.success(f"Result: **{label}** 😊")
                else:
                    st.error(f"Result: **{label}** ☹️")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

st.divider()
st.caption("DATA 228 Demo 3 - Spring 2026")
