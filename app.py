import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt

# Load model
model = load('heart_disease_rf_model.pkl')
import shap
explainer = shap.TreeExplainer(model)

# Feature order
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

def predict_heart_disease_with_risk(input_data, threshold=0.5267):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    prob = model.predict_proba(input_df)[0][1]
    prediction = 1 if prob >= threshold else 0
    return prediction, prob

def get_local_feature_importance(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    explainer = shap.TreeExplainer(model)  # 🟡 Moved here to re-init every time
    shap_values = explainer.shap_values(input_df)

    # Safe access for binary classifiers
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    importance_dict = {feature: float(abs(val[0])) for feature, val in zip(feature_names, shap_vals[0])}
    print("SHAP values for input:", shap_vals[0])
    print("Feature Importance Dict:", importance_dict)

    return importance_dict


def add_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
            background-attachment: fixed;
        }
        .block-container {
            backdrop-filter: blur(8px);
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 2rem;
        }
        </style>
        """, unsafe_allow_html=True
    )

def plot_feature_importance(importance_dict, top_n=5):
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_importance = sorted_importance[:top_n]  # top 5
    features, importances = zip(*top_importance)
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(importances, labels=features, autopct='%1.1f%%', startangle=140,
           colors=plt.cm.Reds(np.linspace(0.4, 0.8, len(features))),
           textprops={'fontsize': 6})
    ax.set_title(f'Top {top_n} Contributing Features', fontsize=8)
    st.pyplot(fig)


def get_health_tips(prediction, input_data):
    tips = []
    if prediction == 1:
        tips.extend([
            "🚭 Quit smoking if you do.",
            "🥗 Adopt a heart-healthy diet.",
            "🏃‍♂️ Exercise regularly.",
            "🧘 Manage stress.",
            "💊 Consult a doctor for treatment."
        ])
    else:
        tips.extend([
            "✅ Great! Keep a healthy lifestyle.",
            "🥗 Eat balanced diet.",
            "🩺 Regular check-ups."
        ])
    if input_data['chol'] > 240:
        tips.append("⚠️ High cholesterol. Reduce fatty foods.")
    if input_data['trestbps'] > 130:
        tips.append("⚠️ High BP. Lower salt intake.")
    if input_data['fbs'] == 1:
        tips.append("⚠️ High fasting sugar. Avoid sugary foods.")
    if input_data['thalach'] < 100:
        tips.append("⚠️ Low heart rate. Consult cardiologist.")
    return tips
def generate_summary(input_data, risk_score, prediction):
    if prediction == 1:
        tone = "⚠️ Based on the values entered, there's an elevated risk of heart disease."
    else:
        tone = "✅ Your inputs suggest a low risk of heart disease. Keep maintaining a healthy lifestyle!"

    risk_note = f"Your calculated risk score is **{risk_score*100:.2f}%**, which helps understand the model's confidence in its prediction."

    chol_note = "High cholesterol may be a concern." if input_data['chol'] > 240 else "Your cholesterol level seems within normal limits."
    bp_note = "Your blood pressure is slightly high." if input_data['trestbps'] > 130 else "Your blood pressure seems fine."

    return f"""
    {tone}<br><br>
    {risk_note}<br>
    🧪 {chol_note}<br>
    🩸 {bp_note}
    """


def main():
    
    add_background()
    st.markdown("<h1 style='text-align: center; color: red;'>🫀 Heart Disease Prediction App </h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Enter your health details to check risk.</p>", unsafe_allow_html=True)

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("🧓 Age", min_value=5, max_value=99, value=50)
            sex = st.selectbox("🧬 Sex", ['Male', 'Female'])
            cp = st.selectbox("🫀 Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
            tres = st.number_input("🩸 Resting Blood Pressure")
            chol = st.number_input("🧪 Serum Cholesterol (mg/dl)")
            fbs = st.selectbox("🩺 Fasting Blood Sugar > 120 mg/dl?", ['Yes', 'No'])
            restecg = st.selectbox("🫀 ECG Results", ['Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy'])

        with col2:
            thalach = st.number_input("🫀 Max Heart Rate", min_value=50, max_value=220, value=150)
            exang = st.selectbox("🏃‍♂️ Exercise Induced Angina?", ['Yes', 'No'])
            oldpeak = st.number_input("📉 Oldpeak (ST Depression)")
            slope = st.selectbox("📈 Slope of ST", ['Upsloping', 'Flat', 'Downsloping'])
            ca = st.selectbox("🔍 Major Vessels (0–3)", [0, 1, 2, 3])
            thal = st.selectbox("🧬 Thalassemia", ['Normal', 'Fixed Defect', 'Reversible Defect'])

        threshold = st.slider("🛠️ Adjust Risk Threshold", 0.1, 0.9, 0.5267, step=0.01)
        submit = st.form_submit_button("💡 Predict Now")
        st.markdown("### ℹ️ Input Reference Guide")
    st.markdown("""
    <div style="background-color:#fff3cd;padding:15px;border-radius:10px;border-left:6px solid #ffeeba;">
    <table style="width:100%; font-size:14px; border-collapse: collapse;">
        <tr style="background-color:#f8d7da;">
            <th style="padding:8px; border: 1px solid #ccc;">🧪 Feature</th>
            <th style="padding:8px; border: 1px solid #ccc;">✅ Healthy / Normal Range</th>
            <th style="padding:8px; border: 1px solid #ccc;">📌 Notes</th>
        </tr>
        <tr>
            <td style="padding:8px; border: 1px solid #ccc;">Age</td>
            <td style="padding:8px; border: 1px solid #ccc;">25 - 65</td>
            <td style="padding:8px; border: 1px solid #ccc;">Heart risk increases with age</td>
        </tr>
        <tr>
            <td style="padding:8px; border: 1px solid #ccc;">Resting BP</td>
            <td style="padding:8px; border: 1px solid #ccc;">90 - 130 mmHg</td>
            <td style="padding:8px; border: 1px solid #ccc;">Above 130 = High BP</td>
        </tr>
        <tr>
            <td style="padding:8px; border: 1px solid #ccc;">Cholesterol</td>
            <td style="padding:8px; border: 1px solid #ccc;">125 - 200 mg/dL</td>
            <td style="padding:8px; border: 1px solid #ccc;">Above 240 = High Risk</td>
        </tr>
        <tr>
            <td style="padding:8px; border: 1px solid #ccc;">Max Heart Rate</td>
            <td style="padding:8px; border: 1px solid #ccc;">100 - 190 bpm</td>
            <td style="padding:8px; border: 1px solid #ccc;">Lower values may be concern</td>
        </tr>
        <tr>
            <td style="padding:8px; border: 1px solid #ccc;">Oldpeak</td>
            <td style="padding:8px; border: 1px solid #ccc;">0 - 2</td>
            <td style="padding:8px; border: 1px solid #ccc;">Higher indicates stress</td>
        </tr>
        <tr>
            <td style="padding:8px; border: 1px solid #ccc;">Fasting Sugar</td>
            <td style="padding:8px; border: 1px solid #ccc;">No (<= 120 mg/dL)</td>
            <td style="padding:8px; border: 1px solid #ccc;">"Yes" = high sugar</td>
        </tr>
    </table>
    <p style="margin-top:10px; font-size:12px;">⚠️ Always consult a healthcare provider for accurate medical advice.</p>
    </div>
    """, unsafe_allow_html=True)


    if submit:
        sex = 1 if sex == 'Male' else 0
        cp_dict = {'Typical Angina': 1, 'Atypical Angina': 2, 'Non-Anginal Pain': 3, 'Asymptomatic': 4}
        cp = cp_dict[cp]
        fbs = 1 if fbs == 'Yes' else 0
        restecg_dict = {'Normal': 0, 'Having ST-T wave abnormality': 1, 'Showing probable or definite left ventricular hypertrophy': 2}
        restecg = restecg_dict[restecg]
        exang = 1 if exang == 'Yes' else 0
        slope_dict = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
        slope = slope_dict[slope]
        thal_dict = {'Normal': 2, 'Fixed Defect': 1, 'Reversible Defect': 3}
        thal = thal_dict[thal]

        input_data = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': tres, 'chol': chol, 'fbs': fbs,
            'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
            'slope': slope, 'ca': ca, 'thal': thal
        }

        prediction, risk_score = predict_heart_disease_with_risk(input_data, threshold)

        confidence_level = "🚨 Very High Confidence" if risk_score >= 0.85 else "⚠️ Moderate Confidence" if risk_score >= 0.65 else "✅ Low Confidence"
        st.markdown(f"**🧠 Model Confidence Level:** {confidence_level}")
        summary_text = generate_summary(input_data, risk_score, prediction)
        st.markdown("### 🧾 Personalized Summary")
        st.markdown(f"<div style='background-color:#f0f8ff;padding:15px;border-radius:10px;'>{summary_text}</div>", unsafe_allow_html=True)


        result_text = "Likely to have heart disease" if prediction == 1 else "Unlikely to have heart disease"
        input_data_with_result = input_data.copy()
        input_data_with_result["Prediction"] = result_text
          # Heart Age Calculation
        heart_age = age + (chol - 200) * 0.05 + (tres - 120) * 0.05 + oldpeak * 2 - (thalach - 100) * 0.1
        heart_age = max(age, int(heart_age))
        heart_color = 'green' if heart_age <= age else 'orange' if heart_age - age <= 5 else 'red'
        st.markdown(f"<h4 style='color:{heart_color};'>🧓 Estimated Heart Age: <b>{heart_age} years</b></h4>", unsafe_allow_html=True)

        # 🧾 Dashboard-style Summary Panel
        st.markdown(
            f"""
            <div style="background-color:#f9f9f9; padding: 15px 20px; border-radius: 12px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1); font-family:Arial;">
                <h3 style="color:#d63384;">📊 Summary Dashboard</h3>
                <ul style="list-style-type:none; padding-left:0; font-size:16px;">
                    <li>✅ <b>Prediction:</b> <span style="color:{'red' if prediction == 1 else 'green'};">{result_text}</span></li>
                    <li>📉 <b>Risk Score:</b> {risk_score * 100:.2f}%</li>
                    <li>🧓 <b>Estimated Heart Age:</b> {heart_age} years</li>
                    <li>🧪 <b>Cholesterol:</b> {chol} mg/dL</li>
                    <li>🩸 <b>Resting Blood Pressure:</b> {tres} mmHg</li>
                    <li>🫀 <b>Maximum Heart Rate:</b> {thalach} bpm</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )


        if prediction == 1:
            st.markdown(f"""
                <div style="background-color:#ffe5e5;padding:15px;border-radius:10px;border-left:8px solid red;">
                    <h3>🧠 Prediction: <span style="color:red;">At Risk</span></h3>
                    <p>📊 <b>Risk Score:</b> {risk_score * 100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color:#e5ffe5;padding:15px;border-radius:10px;border-left:8px solid green;">
                    <h3>🧠 Prediction: <span style="color:green;">Healthy</span></h3>
                    <p>📊 <b>Risk Score:</b> {risk_score * 100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

       

        # Risk Bar
        fig2, ax2 = plt.subplots(figsize=(5, 0.5))
        ax2.barh(['Risk'], [risk_score], color='crimson' if risk_score > 0.5 else 'green')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Probability')
        st.pyplot(fig2)

        importance_dict = get_local_feature_importance(input_data)
        plot_feature_importance(importance_dict)

        st.markdown("### 🩺 Health Tips Based on Your Profile")
        for tip in get_health_tips(prediction, input_data):
            st.markdown(f"- {tip}")

        df_result = pd.DataFrame([input_data_with_result])
        csv = df_result.to_csv(index=False)
        st.download_button("📥 Download Report", csv, file_name="heart_disease_prediction.csv", mime="text/csv")

if __name__ == "__main__":
    main()
