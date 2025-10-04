# # Load everything
# model = joblib.load("movie_recommender.pkl")
# encoders = joblib.load("encoders.pkl")
# columns = joblib.load("columns.pkl")

# # Example input (from Streamlit or manually)
# sample = pd.DataFrame([{
#     "directors": "Steven Spielberg",
#     "actors": "Tom Hanks",
#     "genres": "Action Adventure",
#     "year": 2005
# }])

# # Encode categorical columns
# for col in sample.columns:
#     if col in encoders:
#         le = encoders[col]
#         if sample[col][0] not in le.classes_:
#             # Add unknown safely
#             sample[col] = le.transform([le.classes_[0]])
#         else:
#             sample[col] = le.transform(sample[col])

# # Reorder & fill missing columns
# sample = sample.reindex(columns=columns, fill_value=0)

# # Now predict
# prediction = model.predict(sample)[0]
# app.py
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load model, encoders, and columns
# -------------------------------
model = joblib.load("movie_recommender.pkl")
encoders = joblib.load("encoders.pkl")  # LabelEncoders for categorical columns
columns = joblib.load("columns.pkl")    # Final column order used during training

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Movie Prediction App", page_icon=":clapper:")
st.title(" Movie Prediction App")

st.header("Enter Movie Details")

# User input
directors = st.text_input("Director")
actors = st.text_input("Actors")
genres = st.text_input("Genres")
year = st.number_input("Year", min_value=1900, max_value=2100, step=1)

# Predict button
if st.button("Predict"):
    # Create DataFrame from user input
    sample = pd.DataFrame([{
        "directors": directors,
        "actors": actors,
        "genres": genres,
        "year": year
    }])

    # Encode categorical columns safely
    for col in encoders:
        le = encoders[col]
        sample[col] = sample[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        sample[col] = le.transform(sample[col])

    # Reorder & fill missing columns
    sample = sample.reindex(columns=columns, fill_value=0)

    # Make prediction
    try:
        prediction = model.predict(sample)[0]
        st.success(f"The predicted output is: {prediction}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Show input data
st.subheader("Your Input")
st.write({
    "directors": directors,
    "actors": actors,
    "genres": genres,
    "year": year
})
