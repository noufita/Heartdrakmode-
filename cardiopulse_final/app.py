"""
CardioPulse — Heart Risk Streamlit app
Run:  streamlit run app.py
Needs heart_disease_risk_dataset_earlymed.csv in the same folder.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from model import load_data, train_models, SYMPTOMS, RISKS, FEATURE_ORDER

st.set_page_config(page_title="CardioPulse", page_icon="heart", layout="wide")

# ---------------------------------------------------------------- theme / CSS
PRIMARY = "#E0524A"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');
html, body, [class*="css"] {{ font-family: 'Nunito', sans-serif; }}
.stApp {{ background: #FAF4EF; }}
section[data-testid="stSidebar"] {{ background: #FBF7F2; border-right: 1px solid #E8DED8; }}
.block-container {{ padding-top: 1.5rem; max-width: 1100px; }}
@keyframes pulse {{ 0%{{transform:scale(1)}}12%{{transform:scale(1.18)}}24%{{transform:scale(1)}}
  36%{{transform:scale(1.12)}}54%{{transform:scale(1)}}100%{{transform:scale(1)}} }}
.heart {{ display:inline-block; color:{PRIMARY}; animation: pulse 1.3s ease-in-out infinite; }}
.card {{ background:#fff; border:1px solid #E8DED8; border-radius:18px; padding:20px 22px; margin-bottom:18px; }}
.tile {{ background:#fff; border:1px solid #E8DED8; border-radius:12px; padding:9px 15px; text-align:center; }}
.tile b {{ font-size:20px; font-weight:900; }}
.tile span {{ font-size:11px; color:#857871; }}
h1, h2, h3 {{ color:#463A36; font-weight:900; }}
/* pills -> style multiselect / checkboxes lightly */
div[data-baseweb="tag"] {{ background:{PRIMARY} !important; border-radius:999px !important; }}
</style>
""", unsafe_allow_html=True)

HEART_SVG = ('<svg viewBox="0 0 24 24" width="{w}" height="{w}"><path d="M12 21s-1.45-1.32-3.6-3.3'
             'C5.4 15.36 2 12.28 2 8.5 2 6 4 4 6.5 4c1.74 0 3.04.81 3.91 2.09L12 7.6l1.59-1.51C14.46 '
             '4.81 15.76 4 17.5 4 20 4 22 6 22 8.5c0 3.78-3.4 6.86-6.4 9.2C13.45 19.68 12 21 12 21z" '
             'fill="currentColor"/></svg>')

# ---------------------------------------------------------------- data + models
df = load_data()
BUNDLE = train_models(df)          # {models, scaler, metrics, confusion, X_test, y_test}
MODELS = {"LogReg": "Logistic Regression", "Tree": "Decision Tree", "RF": "Random Forest"}

# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("page", "Predictor")
ss.setdefault("model", "RF")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f'<div style="display:flex;align-items:center;gap:11px;">'
                f'<span style="width:42px;height:42px;border-radius:13px;background:{PRIMARY};'
                f'color:#fff;display:flex;align-items:center;justify-content:center;" class="heart">'
                f'{HEART_SVG.format(w=22)}</span>'
                f'<div><div style="font-weight:900;font-size:19px;color:#463A36;">CardioPulse</div>'
                f'<div style="font-weight:700;font-size:11px;color:#857871;">Heart Risk Predictor</div>'
                f'</div></div><br>', unsafe_allow_html=True)

    ss.page = st.radio("Navigate", ["Predictor", "Dashboard"], label_visibility="collapsed")
    st.divider()

    st.caption("MODEL")
    ss.model = st.radio(
        "Model",
        list(MODELS.keys()),
        format_func=lambda k: f"{MODELS[k]}  ·  {BUNDLE['metrics'][k]['acc']:.2f}%",
        index=list(MODELS).index(ss.model),
        label_visibility="collapsed",
    )

    age = st.slider("Age", 18, 90, 58)
    gender_label = st.radio("Gender", ["Female", "Male"], index=1, horizontal=True)
    gender = 1 if gender_label == "Male" else 0

    st.markdown('<div class="card" style="margin-top:14px;font-size:11px;color:#857871;">'
                'Demo model trained on 70,000 records · for educational use, not medical advice.</div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------- helpers
def risk_level(p):
    if p < 0.35:  return "Low Risk", "#3FA86A", "No urgent indicators detected. Keep up healthy habits."
    if p < 0.65:  return "Moderate Risk", "#D79A36", "Some risk factors present — worth monitoring and a check-up."
    return "High Risk", PRIMARY, "Multiple indicators suggest elevated risk. Consult a clinician."

def age_group(a):  return "Young" if a < 40 else "Middle" if a < 60 else "Senior"

# ---------------------------------------------------------------- PREDICTOR
if ss.page == "Predictor":
    st.markdown("## Check your heart risk")
    st.caption("Select the symptoms and risk factors that apply, then read the live prediction.")

    st.markdown('<div class="card"><b>♥ Symptoms</b> · tap to toggle</div>', unsafe_allow_html=True)
    sym_on = st.multiselect("Symptoms", [s[1] for s in SYMPTOMS], default=["Chest Pain", "Shortness of Breath", "Pain in Arms/Jaw/Back"], label_visibility="collapsed")

    st.markdown('<div class="card"><b>◆ Risk factors</b> · tap to toggle</div>', unsafe_allow_html=True)
    risk_on = st.multiselect("Risk factors", [r[1] for r in RISKS], default=["High Blood Pressure", "High Cholesterol", "Smoking", "Sedentary Lifestyle"], label_visibility="collapsed")

    # build the 21-feature row
    sym_vals = {k: int(lbl in sym_on) for k, lbl in SYMPTOMS}
    risk_vals = {k: int(lbl in risk_on) for k, lbl in RISKS}
    symptom_count = sum(sym_vals.values())
    risk_factor_count = sum(risk_vals.values())
    ag = {"Young": 0, "Middle": 1, "Senior": 2}[age_group(age)]

    row = {**sym_vals, **risk_vals, "Gender": gender, "Age": age,
           "Symptom_Count": symptom_count, "Risk_Factor_Count": risk_factor_count, "Age_Group": ag}
    X = pd.DataFrame([row])[FEATURE_ORDER]

    model = BUNDLE["models"][ss.model]
    Xt = BUNDLE["scaler"].transform(X) if ss.model == "LogReg" else X
    prob = float(model.predict_proba(Xt)[0, 1])
    pct = round(prob * 100)
    label, color, sub = risk_level(prob)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f'<div style="width:158px;height:158px;border-radius:50%;'
            f'background:conic-gradient({color} {pct*3.6}deg, #ECE3DC 0deg);'
            f'display:flex;align-items:center;justify-content:center;">'
            f'<div style="width:118px;height:118px;border-radius:50%;background:#fff;display:flex;'
            f'flex-direction:column;align-items:center;justify-content:center;">'
            f'<div style="font-size:40px;font-weight:900;color:{color};line-height:1;">{pct}'
            f'<span style="font-size:19px;">%</span></div>'
            f'<div style="font-size:10px;font-weight:800;color:#857871;letter-spacing:.1em;">RISK SCORE</div>'
            f'</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="display:flex;align-items:center;gap:11px;">'
                    f'<span class="heart" style="color:{color}">{HEART_SVG.format(w=30)}</span>'
                    f'<span style="font-size:27px;font-weight:900;color:{color};">{label}</span></div>'
                    f'<div style="color:#857871;margin:7px 0 14px;">{sub}</div>', unsafe_allow_html=True)
        t = st.columns(4)
        for col, (b, s) in zip(t, [(symptom_count, "Symptom_Count"), (risk_factor_count, "Risk_Factor_Count"),
                                   (age_group(age), "Age_Group"), (f"{BUNDLE['metrics'][ss.model]['acc']:.2f}%", MODELS[ss.model])]):
            col.markdown(f'<div class="tile"><b>{b}</b><br><span>{s}</span></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- DASHBOARD
else:
    st.markdown("## Dataset & model analytics")
    st.caption("heart_disease_risk_dataset_earlymed.csv · exploratory analysis")

    k = st.columns(3)
    k[0].metric("Records", f"{len(df):,}")
    k[1].metric("Features", len(FEATURE_ORDER))
    k[2].metric("Missing", int(df.isna().sum().sum()))

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Heart Risk distribution**")
        st.bar_chart(df["Heart_Risk"].value_counts().rename({0: "No Risk", 1: "At Risk"}), color=PRIMARY)
    with g2:
        st.markdown("**Age distribution**")
        fig, ax = plt.subplots(); ax.hist(df["Age"], bins=20, color=PRIMARY); st.pyplot(fig)
    with g3:
        st.markdown("**Gender split**")
        gc = df["Gender"].value_counts().rename({0: "Female", 1: "Male"})
        fig, ax = plt.subplots(); ax.pie(gc, labels=gc.index, autopct="%1.1f%%",
                                         colors=[PRIMARY, "#F2D9CF"]); st.pyplot(fig)

    h1, h2 = st.columns([1, 1])
    with h1:
        st.markdown("**Correlation heatmap**")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=6)
        fig.colorbar(im, fraction=0.046); st.pyplot(fig)
    with h2:
        st.markdown(f"**Model performance — {MODELS[ss.model]}**")
        m = BUNDLE["metrics"][ss.model]
        st.markdown(f'<div style="font-size:52px;font-weight:900;color:{PRIMARY};line-height:1;">'
                    f'{m["acc"]:.2f}<span style="font-size:24px;">%</span></div>'
                    f'<span style="color:#857871;">accuracy · {len(BUNDLE["y_test"]):,} test</span>',
                    unsafe_allow_html=True)
        p = st.columns(3)
        p[0].metric("precision", f'{m["precision"]:.2f}')
        p[1].metric("recall", f'{m["recall"]:.2f}')
        p[2].metric("f1-score", f'{m["f1"]:.2f}')
        cm = BUNDLE["confusion"][ss.model]
        st.markdown("**Confusion matrix**")
        st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"],
                                  columns=["Pred 0", "Pred 1"]))
