import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import base64
import random
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Would You Survive the Titanic?",
    layout="wide"
)

# ---------------- BACKGROUND ----------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background: 
            linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.75)),
            url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
    }}

    .block-container {{
        background: rgba(0, 0, 0, 0.4);
        padding: 25px;
        border-radius: 15px;
    }}

    h1, h2, h3, h4, h5, h6, p, div {{
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("Ship_Pic.png")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    df = pd.read_csv("Titanic-Dataset-Cleaned.csv")
    df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    df['FamilySize'] = df['SibSp'] + df['Parch']

    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return df, model, le_sex, le_embarked, acc, X

df, model, le_sex, le_embarked, accuracy, X = load_model()

# ---------------- TITLE ----------------
st.title("Would You Survive the Titanic?")
st.caption("A data-driven survival simulation based on the Titanic disaster (1912).")

# ---------------- RANDOM ----------------
if st.button("🎲 Generate Random Passenger"):
    st.session_state.random = {
        "Pclass": random.choice([1,2,3]),
        "Sex": random.choice(["male","female"]),
        "Age": random.randint(1,80),
        "SibSp": random.randint(0,5),
        "Parch": random.randint(0,5),
        "Fare": random.randint(10,300),
        "Embarked": random.choice(["S","C","Q"])
    }

# ---------------- SIDEBAR ----------------
st.sidebar.header("Passenger Details")

def get_input():
    rand = st.session_state.get("random", {})

    pclass = st.sidebar.selectbox("Class", [1,2,3], index=[1,2,3].index(rand.get("Pclass",1)))
    sex = st.sidebar.selectbox("Sex", ["male","female"], index=["male","female"].index(rand.get("Sex","male")))
    age = st.sidebar.slider("Age", 0,80, rand.get("Age",25))
    sibsp = st.sidebar.slider("Siblings/Spouses", 0,8, rand.get("SibSp",0))
    parch = st.sidebar.slider("Parents/Children", 0,6, rand.get("Parch",0))
    fare = st.sidebar.slider("Fare", 0,500, rand.get("Fare",50))
    embarked = st.sidebar.selectbox("Port", ["S","C","Q"], index=["S","C","Q"].index(rand.get("Embarked","S")))

    family = sibsp + parch

    return pd.DataFrame([{
        "Pclass": pclass,
        "Sex": le_sex.transform([sex])[0],
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": le_embarked.transform([embarked])[0],
        "FamilySize": family
    }])

user_input = get_input()

# ---------------- PREDICTION ----------------
with st.spinner("Analyzing survival probability..."):
    prediction = model.predict(user_input)[0]
    proba = model.predict_proba(user_input)[0]
    prob = proba[1]*100

# ---------------- SUMMARY ----------------
st.markdown(f"""
### Summary
- Survival Chance: **{prob:.1f}%**
- Risk Level: **{'Low' if prob>70 else 'Medium' if prob>40 else 'High'}**
""")

# 🔥 CONFIDENCE INDICATOR (ADDED)
if prob > 80:
    st.success("High confidence prediction")
elif prob > 50:
    st.warning("Moderate confidence prediction")
else:
    st.error("Low confidence prediction")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Passenger Profile")
    st.write(user_input)

with col2:
    st.subheader("Outcome")

    st.markdown(f"""
    <div style="
    padding:20px;
    border-radius:10px;
    background: {'#14532d' if prediction==1 else '#7f1d1d'};
    font-size:18px;">
    {'Likely to survive' if prediction==1 else 'Unlikely to survive'}
    </div>
    """, unsafe_allow_html=True)

    st.write(f"Survival chance: {prob:.2f}%")
    st.progress(int(prob))
    st.write(f"Risk Score: {100-prob:.2f}")

# ---------------- SHAP ----------------
st.subheader("Model Explanation")

explainer = shap.Explainer(model, X)
shap_values = explainer(user_input)

plt.style.use('dark_background')
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

# ---------------- HUMAN EXPLANATION (IMPROVED) ----------------
st.subheader("Key Factors")

for i, val in enumerate(shap_values[0].values):
    if abs(val) > 0.3:
        feature = X.columns[i]

        if feature == "Sex":
            if val > 0:
                st.write("Being female significantly increased survival chances")
            else:
                st.write("Being male reduced survival chances")

        elif feature == "Pclass":
            if val > 0:
                st.write("Higher passenger class improved survival chances")
            else:
                st.write("Lower passenger class reduced survival chances")

        elif feature == "Fare":
            if val > 0:
                st.write("Higher fare contributed positively to survival")
        
        else:
            direction = "increased" if val > 0 else "decreased"
            st.write(f"{feature} {direction} your survival chances")

# ---------------- ALTERNATE SCENARIO ----------------
st.subheader("Alternate Scenario")

alt_input = user_input.copy()
alt_input['Sex'] = 1 - alt_input['Sex']

alt_prob = model.predict_proba(alt_input)[0][1]*100
st.write(f"If gender changed → survival chance: {alt_prob:.2f}%")

# ---------------- VISUAL ----------------
st.subheader("Dataset Insight")
st.write("Women had significantly higher survival rates than men.")

col3, col4 = st.columns(2)

with col3:
    fig1, ax1 = plt.subplots()
    df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=ax1)
    st.pyplot(fig1)

with col4:
    fig2, ax2 = plt.subplots()
    df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

# ---------------- MODEL INFO ----------------
with st.expander("Model Details"):

    st.write("Model: Logistic Regression")
    st.write("Type: Binary Classification")

    st.write(f"Accuracy: {accuracy*100:.2f}%")

    st.write("""
    Logistic Regression predicts survival probability using a sigmoid function.
    It combines features linearly and outputs a value between 0 and 1.
    """)

    st.subheader("Feature Importance")

    importance = pd.Series(model.coef_[0], index=X.columns)
    importance = importance.sort_values()

    st.bar_chart(importance)

    st.subheader("Top Influencing Features")

    top_features = importance.abs().sort_values(ascending=False).head(3)

    for feature in top_features.index:
        st.write(f"- {feature}")

    st.subheader("Why this model?")

    st.write("""
    - Interpretable and simple  
    - Works well for binary outcomes  
    - Provides probability-based predictions  
    """)

# ---------------- FOOTER ----------------
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<hr style='opacity:0.2'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:10px; margin-bottom:10px;">

<h4>About the Creator</h4>

<b>Arnav Singh</b><br>
Machine Learning Enthusiast | Aspiring Data Scientist <br><br>

<a href="mailto:itsarnav.singh80@gmail.com">
itsarnav.singh80@gmail.com
</a><br><br>

<a href="https://www.linkedin.com/in/arnav-singh-a87847351" target="_blank">LinkedIn</a> &nbsp; | &nbsp;
<a href="https://github.com/Arnav-Singh-5080" target="_blank">GitHub</a>

<br><br>
<small style="opacity:0.5;">© 2026 Arnav Singh | Titanic Survival Prediction</small>

</div>
""", unsafe_allow_html=True)
