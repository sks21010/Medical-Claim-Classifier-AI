import streamlit as st 
import joblib 
import pandas as pd 
import numpy as np 
from rf_classes import Node, DecisionTreeClassifier



st.title("Medical Claim Classifier AI")
st.subheader("Patient Intake Details")
st.markdown("Enter <b>medical claim details</b>, and have a <b>Random Forest Classifier</b> model predict whether it's an emergency visit or not!", unsafe_allow_html=True)
#==========LOAD MODEL JUST ONCE==========

MODEL_PATH = "emergency_rf_model.pkl"
COLUMNS_PATH = "feature_columns.pkl"

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        feature_list = joblib.load(COLUMNS_PATH)
        return model, feature_list 
    except FileNotFoundError:
        st.error(f"Required files not found. Ensure {MODEL_PATH} and {COLUMNS_PATH} are properly specified and present")
        return None, None
    

model, feature_columns = load_assets()

#==========UI Input==========

if model is not None:
    age_input = st.number_input("Age of Patient (13-90 yrs.)", min_value=13, max_value=90)
    
    # 3-column grid for selectboxes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender_input = st.selectbox("Gender", options=["Male", "Female"], index=None, placeholder="Select...")
        insurance_provider_input = st.selectbox("Insurance Provider", options=["Blue Cross", "Medicare", "Aetna", "UnitedHealthcare", "Cigna"], index=None, placeholder="Select...")
    
    with col2:
        blood_type_input = st.selectbox("Blood Type", options=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], index=None, placeholder="Select...")
        medication_input = st.selectbox("Medication", options=["Paracetamol", "Ibuprofen", "Aspirin", "Penicillin", "Lipitor"], index=None, placeholder="Select...")
    
    with col3:
        medical_condition_input = st.selectbox("Medical Condition", options=["Cancer", "Obesity", "Diabetes", "Asthma", "Hypertension", "Arthritis"], index=None, placeholder="Select...")
        test_results = st.selectbox("Test Results", options=["Normal", "Inconclusive", "Abnormal"], index=None, placeholder="Select...")
    
    los_input = st.number_input("Length of Stay (0-30 days)", min_value=0, max_value=30)

    input_data = pd.DataFrame({
        "Age": [age_input],
        "Gender": [gender_input],
        "Blood Type": [blood_type_input],
        "Medical Condition": [medical_condition_input],
        "Insurance Provider": [insurance_provider_input],
        "Length of Stay": [los_input],
        "Medication": [medication_input],
        "Test Results": [test_results]
    })

    # Approve selections

    input_data_transposed = input_data.T.reset_index()
    input_data_transposed.columns = ["Field", "Value"]
    input_data_transposed["Value"].astype(str)

    input_data_transposed["Value"] = input_data_transposed["Value"].apply(lambda x: "MISSING" if x is None else x)

    st.subheader("Review and Approve Your Selections")

    with st.popover("Review Your Selection"):
        st.caption("Please verify the entered details:")

        st.table(input_data_transposed)



#==========PREDICTION LOGIC==========

def run_prediction(input_data):

    # Step 2: Bin Age (same bins as training)
    age_bins = [13, 25, 40, 55, 70, 90]
    age_labels = [
        "Young (13-24)",
        "Adult (25-39)", 
        "Middle Age (40-54)",
        "Senior (55-69)",
        "Elderly (70-89)"
    ]
    input_data["Age_Binned"] = pd.cut(
        input_data["Age"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True
    )

    # Step 3: Bin Length of Stay (same bins as training)
    los_bins = [0, 6, 11, 16, 21, 31]
    los_labels = [
        "Very Short Stay (1-5 Days)",
        "Short Stay (6-10 Days)",
        "Medium Stay (11-15 Days)",
        "Long Stay (16-20 Days)",
        "Very Long Stay (21-30 Days)"
    ]
    input_data["LOS_Binned"] = pd.cut(
        input_data["Length of Stay"],
        bins=los_bins,
        labels=los_labels,
        right=False
    )

    # Drop original Age and Length of Stay
    input_data = input_data.drop(columns=['Age', 'Length of Stay'])

    # Step 3: One-hot encode categorical columns
    cat_cols = [col for col in input_data.columns if input_data[col].dtype == 'object' or input_data[col].dtype.name == 'category']
    input_data = pd.get_dummies(input_data, columns=cat_cols, drop_first=False)

    # Step 4: Ensure ALL expected columns are present (in the correct order)
    # Missing columns will be filled with 0 (feature not present)
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match training data
    input_data = input_data[feature_columns]

    # Step 4: Make prediction using the forest_predict function
    def forest_predict(X, forest):
        all_preds = np.array([tree.predict(X).to_numpy() for tree in forest])
        
        maj_vote = []
        for i in range(all_preds.shape[1]):
            values, counts = np.unique(all_preds[:, i], return_counts=True) 
            maj_vote.append(values[np.argmax(counts)])
        return pd.Series(maj_vote, index=X.index)

    # Make prediction
    prediction = forest_predict(input_data, model)

    # Display result
    result = "EMERGENCY VISIT" if prediction.iloc[0] == 1 else "NON-EMERGENCY VISIT"
    st.markdown(f"<h1 style='text-align: center; color: {'red' if prediction.iloc[0] == 1 else 'green'};'>{result}</h1>", unsafe_allow_html=True)



#==========SUBMISSION==========

has_missing = (input_data_transposed["Value"] == "MISSING").any()
if has_missing:
    st.warning("⚠️ Please fill in all required fields")

agree = st.checkbox("I have reviewed the above information")

# 2 conditions for being able to predict risk
if st.button("Predict Risk", disabled=not agree or has_missing):
    run_prediction(input_data)


