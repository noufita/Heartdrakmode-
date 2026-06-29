import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPulse · Heart Risk Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d0f14; color: #e8eaf0; }

[data-testid="stSidebar"] {
    background: #12151d !important;
    border-right: 1px solid #1e2330;
}

.metric-card {
    background: linear-gradient(135deg, #161b27 0%, #1a2035 100%);
    border: 1px solid #252d45;
    border-radius: 14px;
    padding: 22px 24px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-card .val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #e05d8b, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #7480a0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}

.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #e8eaf0;
    border-left: 4px solid #e05d8b;
    padding-left: 14px;
    margin: 28px 0 18px;
}

.model-badge {
    display: inline-block;
    background: linear-gradient(135deg, #e05d8b22, #a855f722);
    border: 1px solid #e05d8b55;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #e05d8b;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
}

.stSelectbox > div > div, .stSlider, .stCheckbox { color: #e8eaf0; }
div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }

.risk-high {
    background: linear-gradient(135deg, #2d1a1a, #3d1515);
    border: 1px solid #c0392b55;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #1a2d1a, #153d15);
    border: 1px solid #27ae6055;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Data & Models ───────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Age': np.random.randint(25, 85, n),
        'Gender': np.random.randint(0, 2, n),
        'Chest_Pain': np.random.randint(0, 2, n),
        'Shortness_of_Breath': np.random.randint(0, 2, n),
        'Fatigue': np.random.randint(0, 2, n),
        'Palpitations': np.random.randint(0, 2, n),
        'Dizziness': np.random.randint(0, 2, n),
        'Swelling': np.random.randint(0, 2, n),
        'Pain_Arms_Jaw_Back': np.random.randint(0, 2, n),
        'Cold_Sweats_Nausea': np.random.randint(0, 2, n),
        'High_BP': np.random.randint(0, 2, n),
        'High_Cholesterol': np.random.randint(0, 2, n),
        'Diabetes': np.random.randint(0, 2, n),
        'Smoking': np.random.randint(0, 2, n),
        'Obesity': np.random.randint(0, 2, n),
        'Sedentary_Lifestyle': np.random.randint(0, 2, n),
        'Family_History': np.random.randint(0, 2, n),
        'Chronic_Stress': np.random.randint(0, 2, n),
    })
    symptom_cols = ['Chest_Pain','Shortness_of_Breath','Fatigue','Palpitations',
                    'Dizziness','Swelling','Pain_Arms_Jaw_Back','Cold_Sweats_Nausea']
    risk_cols = ['High_BP','High_Cholesterol','Diabetes','Smoking',
                 'Obesity','Sedentary_Lifestyle','Family_History','Chronic_Stress']
    score = df[symptom_cols].sum(axis=1)*0.3 + df[risk_cols].sum(axis=1)*0.4 + (df['Age']>55).astype(int)*1.5
    prob = 1 / (1 + np.exp(-(score - 3)))
    df['Heart_Risk'] = (prob > 0.5).astype(int)

    # Feature engineering
    df['Symptom_Count'] = df[symptom_cols].sum(axis=1)
    df['Risk_Factor_Count'] = df[risk_cols].sum(axis=1)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0,40,60,100], labels=[0,1,2]).astype(int)

    return df, symptom_cols, risk_cols

@st.cache_resource
def train_models(df):
    target = 'Heart_Risk'
    y = df[target]
    X = df.drop(columns=[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_tr_sc, y_train)
            pred = model.predict(X_te_sc)
            prob = model.predict_proba(X_te_sc)[:,1]
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:,1]
        results[name] = {
            'model': model,
            'pred': pred,
            'prob': prob,
            'acc': accuracy_score(y_test, pred),
            'auc': roc_auc_score(y_test, prob),
            'report': classification_report(y_test, pred, output_dict=True),
            'cm': confusion_matrix(y_test, pred),
        }
    return results, X_train, X_test, y_train, y_test, scaler, X.columns.tolist()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
        <div style='font-size:2.4rem;'>🫀</div>
        <div style='font-family:Space Grotesk; font-size:1.2rem; font-weight:700; color:#e8eaf0;'>CardioPulse</div>
        <div style='font-size:0.73rem; color:#7480a0; margin-top:4px;'>Heart Risk Analysis Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📊  Overview",
        "🔍  Exploratory Analysis",
        "🤖  Model Performance",
        "🧬  PCA Explorer",
        "🩺  Risk Predictor",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem; color:#4a5270; padding-top:8px;'>Built from Untitled14.ipynb</div>", unsafe_allow_html=True)

df, symptom_cols, risk_cols = load_and_prepare()
results, X_train, X_test, y_train, y_test, scaler, feature_cols = train_models(df)

DARK_BG = '#0d0f14'
PALETTE = ['#e05d8b', '#a855f7', '#3b82f6', '#10b981', '#f59e0b']
PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#9aa3bf', family='Inter'),
        xaxis=dict(gridcolor='#1e2330', linecolor='#252d45'),
        yaxis=dict(gridcolor='#1e2330', linecolor='#252d45'),
    )
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Overview":
    st.markdown("""
    <div style='padding: 32px 0 8px;'>
        <div style='font-family:Space Grotesk; font-size:2rem; font-weight:700; color:#e8eaf0;'>
            Heart Disease Risk Analysis
        </div>
        <div style='color:#7480a0; margin-top:6px; font-size:0.95rem;'>
            EarlyMed Dataset · Machine Learning Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    risk_pct = df['Heart_Risk'].mean() * 100
    best_model = max(results, key=lambda k: results[k]['acc'])
    best_acc = results[best_model]['acc'] * 100
    best_auc = results[best_model]['auc']

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [f"{len(df):,}", f"{risk_pct:.1f}%", f"{best_acc:.1f}%", f"{best_auc:.3f}"],
        ["Total Patients", "At Risk", f"Best Accuracy ({best_model.split()[0]})", "Best AUC Score"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="val">{val}</div>
            <div class="label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Risk distribution + Age distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Heart Risk Distribution</div>', unsafe_allow_html=True)
        counts = df['Heart_Risk'].value_counts().reset_index()
        counts.columns = ['Risk', 'Count']
        counts['Label'] = counts['Risk'].map({0: 'No Risk', 1: 'At Risk'})
        fig = px.pie(counts, values='Count', names='Label',
                     color_discrete_sequence=['#3b82f6', '#e05d8b'],
                     hole=0.55)
        fig.update_traces(textfont_size=13)
        fig.update_layout(**PLOTLY_TEMPLATE['layout'], showlegend=True,
                          legend=dict(font=dict(color='#9aa3bf')),
                          margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Age Distribution by Risk</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        for risk, color, name in [(0, '#3b82f6', 'No Risk'), (1, '#e05d8b', 'At Risk')]:
            fig2.add_trace(go.Histogram(
                x=df[df['Heart_Risk']==risk]['Age'],
                name=name, marker_color=color, opacity=0.7, nbinsx=20
            ))
        fig2.update_layout(**PLOTLY_TEMPLATE['layout'],
                           barmode='overlay', height=300,
                           margin=dict(t=10,b=10,l=10,r=10),
                           legend=dict(font=dict(color='#9aa3bf')))
        st.plotly_chart(fig2, use_container_width=True)

    # Model comparison table
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    comp = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [f"{v['acc']*100:.2f}%" for v in results.values()],
        'AUC-ROC': [f"{v['auc']:.4f}" for v in results.values()],
        'Precision (Risk)': [f"{v['report']['1']['precision']:.4f}" for v in results.values()],
        'Recall (Risk)': [f"{v['report']['1']['recall']:.4f}" for v in results.values()],
        'F1 (Risk)': [f"{v['report']['1']['f1-score']:.4f}" for v in results.values()],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 · EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Exploratory Analysis":
    st.markdown('<div style="font-family:Space Grotesk;font-size:1.8rem;font-weight:700;color:#e8eaf0;padding:24px 0 8px;">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "🏷️ Feature Engineering"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Smoking vs Heart Risk</div>', unsafe_allow_html=True)
            ct = pd.crosstab(df['Smoking'], df['Heart_Risk'])
            ct.columns = ['No Risk','At Risk']
            ct.index = ['Non-Smoker','Smoker']
            fig = px.bar(ct.reset_index(), x='Smoking', y=['No Risk','At Risk'],
                         barmode='group', color_discrete_sequence=['#3b82f6','#e05d8b'])
            fig.update_layout(**PLOTLY_TEMPLATE['layout'], height=300,
                              margin=dict(t=10,b=10,l=10,r=10),
                              legend=dict(font=dict(color='#9aa3bf')))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Gender Distribution</div>', unsafe_allow_html=True)
            gc = df['Gender'].value_counts()
            fig2 = px.pie(values=gc.values, names=['Male','Female'],
                          color_discrete_sequence=['#a855f7','#e05d8b'], hole=0.5)
            fig2.update_layout(**PLOTLY_TEMPLATE['layout'], height=300,
                               margin=dict(t=10,b=10,l=10,r=10),
                               legend=dict(font=dict(color='#9aa3bf')))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Age Boxplot by Risk Group</div>', unsafe_allow_html=True)
        fig3 = px.box(df, x=df['Heart_Risk'].map({0:'No Risk',1:'At Risk'}), y='Age',
                      color=df['Heart_Risk'].map({0:'No Risk',1:'At Risk'}),
                      color_discrete_map={'No Risk':'#3b82f6','At Risk':'#e05d8b'})
        fig3.update_layout(**PLOTLY_TEMPLATE['layout'], height=320,
                           margin=dict(t=10,b=10,l=10,r=10),
                           legend=dict(font=dict(color='#9aa3bf')),
                           xaxis_title='', yaxis_title='Age')
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        numeric_df = df.select_dtypes(include='number')
        corr = numeric_df.corr()
        fig_heat = px.imshow(corr, color_continuous_scale='RdBu_r',
                             zmin=-1, zmax=1, text_auto='.2f',
                             aspect='auto')
        fig_heat.update_layout(**PLOTLY_TEMPLATE['layout'], height=550,
                               margin=dict(t=10,b=10,l=10,r=10))
        fig_heat.update_traces(textfont_size=9)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">Heart Risk by Age Group</div>', unsafe_allow_html=True)
            ag_risk = df.groupby('Age_Group')['Heart_Risk'].mean().reset_index()
            ag_risk['Age_Group'] = ag_risk['Age_Group'].map({0:'Young (≤40)',1:'Middle (41-60)',2:'Senior (60+)'})
            fig_ag = px.bar(ag_risk, x='Age_Group', y='Heart_Risk',
                            color='Heart_Risk', color_continuous_scale=['#3b82f6','#e05d8b'])
            fig_ag.update_layout(**PLOTLY_TEMPLATE['layout'], height=300,
                                 margin=dict(t=10,b=10,l=10,r=10),
                                 yaxis_title='Risk Rate', xaxis_title='')
            st.plotly_chart(fig_ag, use_container_width=True)

        with col2:
            st.markdown('<div class="section-header">Symptom Count vs Risk</div>', unsafe_allow_html=True)
            sc_risk = df.groupby('Symptom_Count')['Heart_Risk'].mean().reset_index()
            fig_sc = px.line(sc_risk, x='Symptom_Count', y='Heart_Risk',
                             markers=True, line_shape='spline',
                             color_discrete_sequence=['#e05d8b'])
            fig_sc.update_layout(**PLOTLY_TEMPLATE['layout'], height=300,
                                 margin=dict(t=10,b=10,l=10,r=10),
                                 yaxis_title='Risk Rate', xaxis_title='Symptom Count')
            st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 · MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":
    st.markdown('<div style="font-family:Space Grotesk;font-size:1.8rem;font-weight:700;color:#e8eaf0;padding:24px 0 8px;">Model Performance</div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Select Model", list(results.keys()))
    r = results[model_choice]

    st.markdown(f'<div class="model-badge">🤖 {model_choice}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Accuracy", f"{r['acc']*100:.2f}%"),
        ("AUC-ROC", f"{r['auc']:.4f}"),
        ("Precision", f"{r['report']['1']['precision']:.4f}"),
        ("Recall", f"{r['report']['1']['recall']:.4f}"),
    ]
    for col, (lbl, val) in zip([c1,c2,c3,c4], metrics):
        col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = r['cm']
        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x='Predicted', y='Actual'),
                           x=['No Risk','At Risk'], y=['No Risk','At Risk'],
                           color_continuous_scale=['#0d0f14','#e05d8b'])
        fig_cm.update_layout(**PLOTLY_TEMPLATE['layout'], height=340,
                             margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, r['prob'])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     line=dict(color='#e05d8b', width=2.5),
                                     name=f"AUC = {r['auc']:.3f}"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                     line=dict(color='#4a5270', dash='dash'),
                                     name='Random', showlegend=False))
        fig_roc.update_layout(**PLOTLY_TEMPLATE['layout'], height=340,
                              margin=dict(t=10,b=10,l=10,r=10),
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate',
                              legend=dict(font=dict(color='#9aa3bf')))
        st.plotly_chart(fig_roc, use_container_width=True)

    # Feature importance for tree models
    if model_choice in ('Decision Tree', 'Random Forest'):
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        fi = pd.Series(r['model'].feature_importances_,
                       index=feature_cols).sort_values(ascending=True).tail(15)
        fig_fi = px.bar(fi, orientation='h',
                        color=fi.values, color_continuous_scale=['#3b82f6','#e05d8b'])
        fig_fi.update_layout(**PLOTLY_TEMPLATE['layout'], height=400,
                             margin=dict(t=10,b=10,l=80,r=10),
                             showlegend=False, yaxis_title='', xaxis_title='Importance')
        st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 · PCA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧬  PCA Explorer":
    st.markdown('<div style="font-family:Space Grotesk;font-size:1.8rem;font-weight:700;color:#e8eaf0;padding:24px 0 8px;">PCA Explorer</div>', unsafe_allow_html=True)

    n_components = st.slider("Number of PCA Components", 2, 10, 5)

    X_all = df.drop(columns=['Heart_Risk'])
    scaler_pca = StandardScaler()
    X_sc = scaler_pca.fit_transform(X_all)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Explained Variance</div>', unsafe_allow_html=True)
        ev = pca.explained_variance_ratio_
        cumev = np.cumsum(ev)
        fig_ev = go.Figure()
        fig_ev.add_bar(x=[f'PC{i+1}' for i in range(n_components)],
                       y=ev, name='Individual', marker_color='#a855f7')
        fig_ev.add_scatter(x=[f'PC{i+1}' for i in range(n_components)],
                           y=cumev, mode='lines+markers',
                           name='Cumulative', line=dict(color='#e05d8b'))
        fig_ev.update_layout(**PLOTLY_TEMPLATE['layout'], height=320,
                             margin=dict(t=10,b=10,l=10,r=10),
                             legend=dict(font=dict(color='#9aa3bf')))
        st.plotly_chart(fig_ev, use_container_width=True)
        st.markdown(f"""
        <div style='background:#1a2035;border:1px solid #252d45;border-radius:10px;padding:14px;font-size:0.85rem;color:#9aa3bf;'>
        📌 First {n_components} components explain <strong style='color:#e05d8b;'>{cumev[-1]*100:.1f}%</strong> of total variance
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">PC1 vs PC2 (by Risk)</div>', unsafe_allow_html=True)
        pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
        pca_df['Risk'] = df['Heart_Risk'].map({0:'No Risk', 1:'At Risk'}).values
        fig_sc2 = px.scatter(pca_df, x='PC1', y='PC2', color='Risk',
                             color_discrete_map={'No Risk':'#3b82f6','At Risk':'#e05d8b'},
                             opacity=0.6)
        fig_sc2.update_traces(marker=dict(size=5))
        fig_sc2.update_layout(**PLOTLY_TEMPLATE['layout'], height=320,
                              margin=dict(t=10,b=10,l=10,r=10),
                              legend=dict(font=dict(color='#9aa3bf')))
        st.plotly_chart(fig_sc2, use_container_width=True)

    st.markdown('<div class="section-header">PCA Loadings (PC1 & PC2)</div>', unsafe_allow_html=True)
    loadings = pd.DataFrame(pca.components_[:2].T,
                            index=feature_cols,
                            columns=['PC1','PC2'])
    fig_load = px.imshow(loadings.T, text_auto='.2f',
                         color_continuous_scale='RdBu_r', zmin=-0.6, zmax=0.6,
                         aspect='auto')
    fig_load.update_layout(**PLOTLY_TEMPLATE['layout'], height=220,
                           margin=dict(t=10,b=10,l=10,r=10))
    fig_load.update_traces(textfont_size=9)
    st.plotly_chart(fig_load, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 · RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🩺  Risk Predictor":
    st.markdown('<div style="font-family:Space Grotesk;font-size:1.8rem;font-weight:700;color:#e8eaf0;padding:24px 0 8px;">Individual Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#7480a0;margin-bottom:20px;">Enter patient details to predict heart disease risk using trained models.</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            age = st.slider("Age", 25, 85, 55)
            gender = st.selectbox("Gender", ["Male (1)", "Female (0)"])
            gender_val = 1 if "Male" in gender else 0

            st.markdown("**Lifestyle Risk Factors**")
            smoking = st.checkbox("Smoking")
            obesity = st.checkbox("Obesity")
            sedentary = st.checkbox("Sedentary Lifestyle")
            chronic_stress = st.checkbox("Chronic Stress")

        with col2:
            st.markdown("**Medical Risk Factors**")
            high_bp = st.checkbox("High Blood Pressure")
            high_chol = st.checkbox("High Cholesterol")
            diabetes = st.checkbox("Diabetes")
            family_hist = st.checkbox("Family History")

            st.markdown("**Symptoms**")
            chest_pain = st.checkbox("Chest Pain")
            sob = st.checkbox("Shortness of Breath")
            fatigue = st.checkbox("Fatigue")
            palpitations = st.checkbox("Palpitations")

        with col3:
            st.markdown("**More Symptoms**")
            dizziness = st.checkbox("Dizziness")
            swelling = st.checkbox("Swelling")
            pain_arms = st.checkbox("Pain in Arms / Jaw / Back")
            cold_sweats = st.checkbox("Cold Sweats / Nausea")

        model_sel = st.selectbox("Prediction Model", list(results.keys()))
        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

    if submitted:
        input_dict = {
            'Age': age, 'Gender': gender_val,
            'Chest_Pain': int(chest_pain), 'Shortness_of_Breath': int(sob),
            'Fatigue': int(fatigue), 'Palpitations': int(palpitations),
            'Dizziness': int(dizziness), 'Swelling': int(swelling),
            'Pain_Arms_Jaw_Back': int(pain_arms), 'Cold_Sweats_Nausea': int(cold_sweats),
            'High_BP': int(high_bp), 'High_Cholesterol': int(high_chol),
            'Diabetes': int(diabetes), 'Smoking': int(smoking),
            'Obesity': int(obesity), 'Sedentary_Lifestyle': int(sedentary),
            'Family_History': int(family_hist), 'Chronic_Stress': int(chronic_stress),
        }
        # Add engineered features
        s_cols = ['Chest_Pain','Shortness_of_Breath','Fatigue','Palpitations',
                  'Dizziness','Swelling','Pain_Arms_Jaw_Back','Cold_Sweats_Nausea']
        r_cols = ['High_BP','High_Cholesterol','Diabetes','Smoking',
                  'Obesity','Sedentary_Lifestyle','Family_History','Chronic_Stress']
        input_dict['Symptom_Count'] = sum(input_dict[c] for c in s_cols)
        input_dict['Risk_Factor_Count'] = sum(input_dict[c] for c in r_cols)
        input_dict['Age_Group'] = 0 if age <= 40 else (1 if age <= 60 else 2)

        input_df = pd.DataFrame([input_dict])[feature_cols]

        model_r = results[model_sel]
        if model_sel == 'Logistic Regression':
            input_sc = scaler.transform(input_df)
            pred = model_r['model'].predict(input_sc)[0]
            prob = model_r['model'].predict_proba(input_sc)[0][1]
        else:
            pred = model_r['model'].predict(input_df)[0]
            prob = model_r['model'].predict_proba(input_df)[0][1]

        st.markdown("")
        col1, col2 = st.columns([1,1])
        with col1:
            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <div style='font-size:2.5rem;'>⚠️</div>
                    <div style='font-family:Space Grotesk;font-size:1.4rem;font-weight:700;color:#e74c3c;margin-top:8px;'>High Risk Detected</div>
                    <div style='color:#c0392b;font-size:0.85rem;margin-top:4px;'>Risk Probability: {prob*100:.1f}%</div>
                    <div style='color:#9aa3bf;font-size:0.78rem;margin-top:10px;'>Consult a cardiologist promptly.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <div style='font-size:2.5rem;'>✅</div>
                    <div style='font-family:Space Grotesk;font-size:1.4rem;font-weight:700;color:#27ae60;margin-top:8px;'>Low Risk</div>
                    <div style='color:#1e8449;font-size:0.85rem;margin-top:4px;'>Risk Probability: {prob*100:.1f}%</div>
                    <div style='color:#9aa3bf;font-size:0.78rem;margin-top:10px;'>Maintain healthy habits & routine checkups.</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode='gauge+number',
                value=prob*100,
                title=dict(text='Risk Score (%)', font=dict(color='#9aa3bf')),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor='#4a5270'),
                    bar=dict(color='#e05d8b'),
                    bgcolor='#12151d',
                    bordercolor='#1e2330',
                    steps=[
                        dict(range=[0,40], color='#1a2d1a'),
                        dict(range=[40,70], color='#2d2a1a'),
                        dict(range=[70,100], color='#2d1a1a'),
                    ],
                    threshold=dict(line=dict(color='#e05d8b', width=3), value=50)
                ),
                number=dict(suffix='%', font=dict(color='#e8eaf0', size=28))
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#9aa3bf'),
                height=260, margin=dict(t=20,b=10,l=20,r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
