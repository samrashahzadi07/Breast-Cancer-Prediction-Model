import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load model
# -----------------------------
model = pickle.load(open("trained_model.sav", "rb"))

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Navigation")

st.sidebar.markdown("### 🔬 About This Project")
st.sidebar.info(
    "This app predicts whether a **breast tumor** is "
    "**Malignant** (cancerous) or **Benign** (non-cancerous) "
    "based on selected important medical features."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 About Me")
st.sidebar.write("**Mirza Yasir Abdullah Baig**")
st.sidebar.markdown(
    """
    - [🌐 Kaggle](https://www.kaggle.com/mirzayasirabdullah07)  
    - [💼 LinkedIn](https://www.linkedin.com/in/mirza-yasir-abdullah-baig/)  
    - [💻 GitHub](https://github.com/mirzayasirabdullahbaig07)  
    """
)

# -----------------------------
# Main App Title
# -----------------------------
st.markdown(
    """
    <div style="background-color:#f63366;padding:20px;border-radius:12px;margin-bottom:20px">
    <h1 style="color:white;text-align:center;">🎗️ Breast Cancer Prediction App</h1>
    <p style="color:white;text-align:center;">AI-powered tool to detect breast cancer risk</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Define all features (model expects 30)
# -----------------------------
all_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Important ones (user will see only these)
important_features = [
    'mean radius', 'mean texture', 'mean perimeter',
    'mean area', 'mean smoothness',
    'worst radius', 'worst perimeter', 'worst area'
]

# -----------------------------
# Input Form
# -----------------------------
st.markdown("### 📝 Enter Tumor Features")
with st.form("input_form"):
    user_inputs = {}
    cols = st.columns(2)
    for i, feature in enumerate(important_features):
        with cols[i % 2]:
            val = st.number_input(
                f"🔹 {feature.capitalize()}",
                min_value=0.0,
                value=1.0,
                step=0.01,
                key=feature
            )
            user_inputs[feature] = val

    submitted = st.form_submit_button("🔍 Predict")

# Fill other (unimportant) features with default = 0
final_inputs = []
for f in all_features:
    if f in user_inputs:
        final_inputs.append(user_inputs[f])
    else:
        final_inputs.append(0.0)

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    input_data = np.array(final_inputs).reshape(1, -1)
    prediction = model.predict(input_data)
    label = np.argmax(prediction)

    # Result card
    if label == 0:
        st.markdown(
            """
            <div style="background-color:#ff4d4d;padding:20px;border-radius:12px;text-align:center;">
                <h2 style="color:white;">⚠️ Prediction: Malignant</h2>
                <p style="color:white;">The tumor is likely **cancerous**. Please consult a doctor immediately.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color:#28a745;padding:20px;border-radius:12px;text-align:center;">
                <h2 style="color:white;">✅ Prediction: Benign</h2>
                <p style="color:white;">Good news! The tumor is likely **non-cancerous**.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit | Powered by Machine Learning</p>",
    unsafe_allow_html=True
)
