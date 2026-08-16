import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Lab 8: Naive Bayes Classifier | F1 Pit Stop Prediction",
    page_icon="🏎️",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    # Look for the dataset in data/ or current directory
    possible_paths = [
        "data/f1_strategy_dataset_v6.csv",
        "data/f1_dataset.csv",
        "f1_strategy_dataset_v6.csv",
        "f1_dataset.csv"
    ]
    data_path = None
    for p in possible_paths:
        if os.path.exists(p):
            data_path = p
            break
            
    if data_path:
        df = pd.read_csv(data_path)
    else:
        # Fallback synthetic dataset generation if file is missing
        np.random.seed(42)
        n_samples = 5000
        lap_num = np.random.randint(1, 70, size=n_samples)
        tyre_life = np.random.randint(1, 35, size=n_samples)
        norm_tyre_life = tyre_life / 35.0
        cum_deg = norm_tyre_life * 2.5 + np.random.normal(0, 0.2, size=n_samples)
        lap_time_delta = np.random.normal(0.2, 0.8, size=n_samples) + (norm_tyre_life * 1.5)
        race_progress = lap_num / 70.0
        
        # Pit decision logic
        pit_prob = 1 / (1 + np.exp(-(cum_deg * 2.0 + norm_tyre_life * 3.0 + lap_time_delta * 1.2 - 4.5)))
        pit_next_lap = (np.random.rand(n_samples) < pit_prob).astype(int)
        
        df = pd.DataFrame({
            'LapNumber': lap_num,
            'TyreLife': tyre_life,
            'Normalized_TyreLife': norm_tyre_life,
            'Cumulative_Degradation': cum_deg,
            'LapTime_Delta': lap_time_delta,
            'RaceProgress': race_progress,
            'PitNextLap': pit_next_lap
        })

    # Preprocessing: Forward-fill missing categorical/compound values if any
    df = df.ffill()
    return df

df = load_data()

# --- Sidebar Controls ---
st.sidebar.title("Experiment Controls")
st.sidebar.markdown("Configure hyperparameters and test splits for the **Gaussian Naive Bayes** model.")

# Feature selection
all_numeric_features = [
    'Normalized_TyreLife', 
    'Cumulative_Degradation', 
    'LapTime_Delta', 
    'RaceProgress', 
    'LapNumber',
    'TyreLife'
]
available_features = [col for col in all_numeric_features if col in df.columns]

selected_features = st.sidebar.multiselect(
    "Select Features for Training:",
    options=available_features,
    default=[f for f in ['Normalized_TyreLife', 'Cumulative_Degradation', 'LapTime_Delta', 'RaceProgress'] if f in available_features]
)

test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
random_state = st.sidebar.number_input("Random State Seed", value=42, step=1)

# Sampling slider to optimize speed if dataset is huge
sample_size = st.sidebar.slider("Training Sample Limit (Rows)", min_value=5000, max_value=len(df), value=min(25000, len(df)), step=5000)

# --- Model Training Function with Caching ---
@st.cache_resource
def train_naive_bayes(data_subset, features, test_ratio, seed):
    X = data_subset[features]
    y = data_subset['PitNextLap']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'conf_matrix': confusion_matrix(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    }
    
    return model, X_train, X_test, y_train, y_test, y_pred, metrics

# Prepare data sample
data_sampled = df.sample(n=sample_size, random_state=int(random_state)) if len(df) > sample_size else df

if len(selected_features) > 0:
    model, X_train, X_test, y_train, y_test, y_pred, metrics = train_naive_bayes(
        data_sampled, selected_features, test_size, int(random_state)
    )
else:
    st.error("Please select at least one feature from the sidebar to train the classifier.")
    st.stop()

# --- Main App Header ---
st.title("Lab Exercise 8: Naive Bayes Classification")
st.markdown("**Domain:** Formula 1 Strategy & Telemetry Analytics — Predicting `PitNextLap` (0 = Stay Out, 1 = Pit)")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Theory & Mathematical Working", "Model Evaluation & Metrics", "Unseen Instance Predictor"])

# ==========================================
# TAB 1: THEORY & IMPLEMENTATION
# ==========================================
with tab1:
    st.header("1. Mathematical Foundations of Naive Bayes")
    
    st.markdown(r"""
    The **Naive Bayes Classifier** is a probabilistic supervised learning algorithm rooted in **Bayes' Theorem**. 
    It is termed *'naive'* because it makes the simplifying assumption that all input features $x_1, x_2, \dots, x_n$ are mutually independent given the class label $y$.
    """)
    
    st.subheader("Bayes' Theorem Formula")
    st.latex(r"P(y \mid x_1, x_2, \dots, x_n) = \frac{P(y) \cdot P(x_1, x_2, \dots, x_n \mid y)}{P(x_1, x_2, \dots, x_n)}")
    
    st.markdown(r"""
    Applying the class-conditional independence assumption:
    """)
    st.latex(r"P(y \mid X) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)")
    
    st.markdown(r"""
    The final predicted class $\hat{y}$ corresponds to the Maximum A Posteriori (MAP) decision rule:
    """)
    st.latex(r"\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^{n} P(x_i \mid y)")
    
    st.markdown("---")
    
    st.header("2. Continuous Features & Gaussian Likelihood")
    st.markdown(r"""
    Because motorsport telemetry features (such as `Normalized_TyreLife`, `Cumulative_Degradation`, and `LapTime_Delta`) are continuous variables, we utilize **Gaussian Naive Bayes**.
    The continuous values associated with each class are assumed to follow a Gaussian (normal) distribution:
    """)
    st.latex(r"P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)")
    
    st.markdown(r"""
    Where:
    * $\mu_y$ is the calculated mean of feature $x_i$ for class $y$.
    * $\sigma_y^2$ is the calculated variance of feature $x_i$ for class $y$.
    """)
    
    st.markdown("---")
    
    st.header("3. Lab Execution Methodology")
    st.markdown(r"""
    The pipeline implemented in this dashboard follows standard machine learning workflows:
    1. **Data Ingestion & Preprocessing:** Loaded the lap-by-lap Formula 1 strategy dataset and executed forward-fill (`ffill`) imputation to handle any missing telemetry records.
    2. **Stratified Partitioning:** Split the processed dataset into Training and Testing partitions according to the configured test ratio while preserving the class distribution of `PitNextLap`.
    3. **Model Fitting:** Fitted `GaussianNB` across the selected continuous features to estimate class priors $P(y)$ and feature parameters $(\mu_y, \sigma_y^2)$.
    4. **Inference & Validation:** Generated discrete class predictions and calculated Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix.
    5. **New Instance Scoring:** Built an interactive simulation environment to evaluate real-time telemetry inputs against the trained probabilistic decision boundary.
    """)

# ==========================================
# TAB 2: MODEL EVALUATION & METRICS
# ==========================================
with tab2:
    st.header("Model Performance & Evaluation Metrics")
    
    col_split1, col_split2, col_split3 = st.columns(3)
    col_split1.metric("Total Records Analyzed", f"{len(data_sampled):,}")
    col_split2.metric("Training Set Size", f"{len(X_train):,}")
    col_split3.metric("Testing Set Size", f"{len(X_test):,}")
    
    st.markdown("---")
    
    # KPI Metric Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    kpi2.metric("Precision (Class 1)", f"{metrics['precision']:.4f}")
    kpi3.metric("Recall (Class 1)", f"{metrics['recall']:.4f}")
    kpi4.metric("F1-Score (Class 1)", f"{metrics['f1']:.4f}")
    
    st.markdown("---")
    
    col_cm, col_report = st.columns([1.2, 1])
    
    with col_cm:
        st.subheader("Confusion Matrix")
        cm = metrics['conf_matrix']
        cm_text = [[f"TN: {cm[0,0]}", f"FP: {cm[0,1]}"],
                   [f"FN: {cm[1,0]}", f"TP: {cm[1,1]}"]]
        
        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=["Predicted: Stay Out (0)", "Predicted: Pit (1)"],
            y=["Actual: Stay Out (0)", "Actual: Pit (1)"],
            annotation_text=cm_text,
            colorscale='Blues',
            showscale=True
        )
        fig_cm.update_layout(
            margin=dict(t=40, b=40, l=40, r=40),
            height=380
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_report:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(metrics['report']).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), height=380, use_container_width=True)
        
    st.markdown("---")
    
    # Feature Distributions by Class
    st.subheader("Feature Distributions by Class Condition (Gaussian Fit)")
    selected_dist_feat = st.selectbox("Select Feature to Inspect Class Separation:", options=selected_features)
    
    fig_dist = px.histogram(
        data_sampled,
        x=selected_dist_feat,
        color=data_sampled['PitNextLap'].astype(str),
        barmode="overlay",
        marginal="box",
        labels={'color': 'PitNextLap'},
        color_discrete_map={'0': '#1f77b4', '1': '#d62728'},
        title=f"Distribution of {selected_dist_feat} Conditioned on Pit Decision"
    )
    fig_dist.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

# ==========================================
# TAB 3: UNSEEN INSTANCE PREDICTOR
# ==========================================
with tab3:
    st.header("Predict Pit Stop for an Unseen Instance")
    st.markdown("Adjust the live telemetry attributes below to simulate an unseen car state during a Grand Prix.")
    
    input_cols = st.columns(2)
    user_inputs = {}
    
    for i, feature in enumerate(selected_features):
        col = input_cols[i % 2]
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        
        user_inputs[feature] = col.slider(
            f"{feature}",
            min_value=round(min_val, 2),
            max_value=round(max_val, 2),
            value=round(mean_val, 2),
            step=round((max_val - min_val) / 100.0, 3)
        )
        
    input_df = pd.DataFrame([user_inputs])
    
    st.markdown("---")
    st.subheader("Simulated Telemetry Vector")
    st.dataframe(input_df, use_container_width=True)
    
    # Inference on Unseen Instance
    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("Model Decision")
        if pred_class == 1:
            st.error("🚨 **PREDICTION: BOX, BOX! (Pit on Next Lap)**")
        else:
            st.success("🟢 **PREDICTION: STAY OUT (Continue Stint)**")
            
    with res_col2:
        st.subheader("Posterior Probabilities")
        prob_stay = pred_proba[0] * 100
        prob_pit = pred_proba[1] * 100
        
        st.write(f"**Stay Out Probability:** `{prob_stay:.2f}%`")
        st.progress(prob_stay / 100.0)
        
        st.write(f"**Pit Stop Probability:** `{prob_pit:.2f}%`")
        st.progress(prob_pit / 100.0)
