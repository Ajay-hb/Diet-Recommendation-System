
import streamlit as st
import pickle
import pandas as pd

# --- Load the saved objects ---
@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    with open('one_hot_encoder.pkl', 'rb') as file:
        loaded_encoder = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        loaded_label_encoder = pickle.load(file)
    return loaded_model, loaded_encoder, loaded_label_encoder

loaded_model, loaded_encoder, loaded_label_encoder = load_model()

# --- Define the feature categories (must match training features) ---
features = [
    "Age_category", "Gender", "Height_cat", "Weight_cat", "BMI_cat",
    "Activity_Level", "Sugar_Level_cat", "Cholesterol_cat", "Goal"
]

# --- Streamlit App Interface ---
st.set_page_config(page_title="Diet Recommendation App")
st.title("Diet Recommendation Predictor")
st.write("Enter your details to get a personalized diet recommendation.")

# Input fields for each feature

# Age Category
age_category_options = ['young', 'Adult', 'Senior']
selected_age_category = st.selectbox("Age Category", age_category_options)

# Gender
gender_options = {0: 'Female', 1: 'Male'}
selected_gender = st.selectbox("Gender", list(gender_options.keys()), format_func=lambda x: gender_options[x])

# Height Category
height_cat_options = ['Short', 'Average Height', 'Tall']
selected_height_cat = st.selectbox("Height Category", height_cat_options)

# Weight Category
weight_cat_options = ['Under_Weight', 'Normal_Weight', 'Over_Weight']
selected_weight_cat = st.selectbox("Weight Category", weight_cat_options)

# BMI Category
bmi_cat_options = ['Under_Weight', 'Normal_Weight', 'Over_Weight']
selected_bmi_cat = st.selectbox("BMI Category", bmi_cat_options)

# Activity Level
activity_level_options = {0: 'No Exercise', 1: 'Moderate Exercise', 2: 'Heavy Exercise'}
selected_activity_level = st.selectbox("Activity Level", list(activity_level_options.keys()), format_func=lambda x: activity_level_options[x])

# Sugar Level Category
sugar_level_cat_options = ['Low_Sugar', 'Normal_Sugar', 'High_Sugar']
selected_sugar_level_cat = st.selectbox("Sugar Level Category", sugar_level_cat_options)

# Cholesterol Category
cholesterol_cat_options = ['Desirable', 'High'] # Assuming these are the only categories present after preprocessing
selected_cholesterol_cat = st.selectbox("Cholesterol Category", cholesterol_cat_options)

# Goal
goal_options = {0: 'Weight Loss', 1: 'Weight Gain', 2: 'Maintain'}
selected_goal = st.selectbox("Goal", list(goal_options.keys()), format_func=lambda x: goal_options[x])

# Predict button
if st.button("Get Diet Recommendation"):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([{
        'Age_category': selected_age_category,
        'Gender': selected_gender,
        'Height_cat': selected_height_cat,
        'Weight_cat': selected_weight_cat,
        'BMI_cat': selected_bmi_cat,
        'Activity_Level': selected_activity_level,
        'Sugar_Level_cat': selected_sugar_level_cat,
        'Cholesterol_cat': selected_cholesterol_cat,
        'Goal': selected_goal
    }])

    # Preprocess the input data
    # Ensure input_data has all the original feature columns for the encoder
    # Create a dummy DataFrame with all possible categories to ensure the encoder transforms correctly
    # This assumes the original training data had these categories. A more robust solution
    # would involve storing the categories or fitting a new encoder with known categories.

    # Reconstruct a DataFrame with a single row for prediction, ensuring columns match the training data
    # This part is crucial to ensure the one-hot encoder gets all columns it expects
    dummy_df = pd.DataFrame(columns=features) # Create an empty df with correct columns
    # Append the input data, handling potential missing columns that the encoder expects
    processed_input = pd.concat([dummy_df, input_data], ignore_index=True)

    # Apply one-hot encoding
    new_data_encoded = loaded_encoder.transform(processed_input)
    new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=loaded_encoder.get_feature_names_out(features))

    # Make prediction
    prediction_encoded = loaded_model.predict(new_data_encoded_df)

    # Decode the prediction
    predicted_diet = loaded_label_encoder.inverse_transform(prediction_encoded)

    st.success(f"Based on your input, the recommended Diet is: **{predicted_diet[0]}**")
