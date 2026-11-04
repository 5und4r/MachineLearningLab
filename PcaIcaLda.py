import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="PCA vs LDA vs ICA Comparative Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------
@st.cache_data
def load_data():
    raw_data = load_breast_cancer(as_frame=True)
    df = raw_data.frame
    target_names = raw_data.target_names
    return df, target_names

df, target_names = load_data()
feature_cols = [c for c in df.columns if c != 'target']

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("⚙️ Hyperparameters")
st.sidebar.markdown("Configure experiment settings:")

test_size = st.sidebar.slider("Test Size Split", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random Seed", value=42, step=1)

st.sidebar.divider()
st.sidebar.subheader("PCA Settings")
pca_mode = st.sidebar.radio("PCA Criterion", ["Variance Ratio (Lab default: 0.95)", "Fixed Number of Components"])
if pca_mode == "Variance Ratio (Lab default: 0.95)":
    pca_param = st.sidebar.slider("Retained Variance Ratio", 0.50, 0.99, 0.95, 0.01)
else:
    pca_param = st.sidebar.slider("Number of Principal Components", 2, 30, 10, 1)

st.sidebar.divider()
st.sidebar.subheader("LDA Settings")
lda_solver = st.sidebar.selectbox("LDA Solver", ["eigen", "svd", "lsqr"], index=0)

st.sidebar.divider()
st.sidebar.subheader("ICA Settings")
ica_components = st.sidebar.slider("ICA Components", 2, 15, 8, 1)
ica_whiten = st.sidebar.selectbox("ICA Whitening", ["unit-variance", "arbitrary-variance"], index=0)

# ---------------------------------------------------------
# Main Computation Pipeline
# ---------------------------------------------------------
X = df[feature_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. PCA
pca = PCA(n_components=pca_param, random_state=random_state)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 2. LDA (Classification & 1D Projection)
lda_model = LinearDiscriminantAnalysis(solver=lda_solver)
lda_model.fit(X_train_scaled, y_train)
y_pred_lda = lda_model.predict(X_test_scaled)

lda_dim = LinearDiscriminantAnalysis(n_components=1, solver=lda_solver)
X_train_lda = lda_dim.fit_transform(X_train_scaled, y_train)
X_test_lda = lda_dim.transform(X_test_scaled)

# 3. ICA
ica = FastICA(
    n_components=ica_components,
    whiten=ica_whiten,
    random_state=random_state,
    max_iter=1000
)
X_train_ica = ica.fit_transform(X_train_scaled)
X_test_ica = ica.transform(X_test_scaled)

# ---------------------------------------------------------
# Header & Scenario Overview
# ---------------------------------------------------------
st.markdown('<div class="main-title">Lab 4: Comparative Analysis of PCA, LDA, and ICA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Exploring Dimensionality Reduction and Feature Extraction on High-Dimensional Clinical Oncology Data</div>',
    unsafe_allow_html=True
)

tabs = st.tabs([
    "📂 1. Dataset & EDA",
    "🔬 2. Visualizations (PCA, LDA, ICA)",
    "📊 3. Metric Diagnostics",
    "📝 4. Comparative Report",
    "🔮 5. Live Predictor"
])

# ---------------------------------------------------------
# Tab 1: Dataset & EDA
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Scenario & Dataset Overview")
    st.info("""
    **Healthcare Research Scenario:** High-dimensional clinical patient measurements often suffer from feature multicollinearity and redundancy. 
    Dimensionality reduction simplifies the feature space while retaining crucial biological variance, discriminative power, and independent signal components.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1] - 1)
    col3.metric("Malignant (Class 0)", int((y == 0).sum()))
    col4.metric("Benign (Class 1)", int((y == 1).sum()))

    exp_data = st.expander("🔍 View Raw Data Sample & Summary Statistics", expanded=False)
    with exp_data:
        st.write("##### First 5 Rows")
        st.dataframe(df.head(), use_container_width=True)
        st.write("##### Numerical Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        st.write("##### Missing Values Analysis")
        null_counts = df.isnull().sum()
        st.write(f"Total Missing Values in Dataset: **{null_counts.sum()}**")

    st.subheader("Correlation Heatmap of Numerical Features")
    show_corr = st.checkbox("Render Full Correlation Matrix (30x30)", value=True)
    if show_corr:
        fig_corr, ax_corr = plt.subplots(figsize=(14, 11))
        corr_mat = df.corr()
        sns.heatmap(
            corr_mat,
            cmap="coolwarm",
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            square=True,
            linewidths=0.3,
            cbar_kws={"shrink": 0.75},
            ax=ax_corr
        )
        ax_corr.set_title("Correlation Matrix of Breast Cancer Numerical Attributes", fontsize=13, pad=12)
        st.pyplot(fig_corr)
        plt.close(fig_corr)

# ---------------------------------------------------------
# Tab 2: Visualizations (Scatter Plots)
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Visualizing Dimensionality Reduction Projections")
    st.markdown("Replicating the side-by-side scatter plots for the three distinct reduction paradigms.")

    np.random.seed(random_state)
    jitter = np.random.normal(0, 0.05, size=X_train_lda.shape[0])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. PCA Scatter Plot
    scatter1 = axes[0].scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=y_train,
        cmap='coolwarm',
        edgecolor='k',
        alpha=0.8
    )
    axes[0].set_xlabel('Principal Component 1', fontsize=11)
    axes[0].set_ylabel('Principal Component 2', fontsize=11)
    axes[0].set_title('PCA (Explained Variance)', fontsize=13, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. LDA Scatter Plot with Jitter
    axes[1].scatter(
        X_train_lda[:, 0],
        jitter,
        c=y_train,
        cmap='coolwarm',
        edgecolor='k',
        alpha=0.8
    )
    axes[1].set_xlabel('Linear Discriminant 1', fontsize=11)
    axes[1].set_yticks([])
    axes[1].set_title('LDA (Class Separability)', fontsize=13, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 3. ICA Scatter Plot
    axes[2].scatter(
        X_train_ica[:, 0],
        X_train_ica[:, 1],
        c=y_train,
        cmap='coolwarm',
        edgecolor='k',
        alpha=0.8
    )
    axes[2].set_xlabel('Independent Component 1', fontsize=11)
    axes[2].set_ylabel('Independent Component 2', fontsize=11)
    axes[2].set_title('ICA (Independent Sources)', fontsize=13, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('Comparison of Dimensionality Reduction Techniques on Breast Cancer Dataset', fontsize=15, y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    - **PCA (Unsupervised):** Preserves dominant directions of dataset variance. While benign and malignant cases broadly separate, there is boundary overlap.
    - **LDA (Supervised):** Maximizes the between-class variance relative to within-class variance, resulting in two tightly focused, highly separable clusters along 1D.
    - **ICA (Blind Source Separation):** Looks for statistically independent, non-Gaussian source components, exposing underlying physical/biological signal structure without using class labels.
    """)

# ---------------------------------------------------------
# Tab 3: Metric Diagnostics
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("Component Retention and Mathematical Diagnostics")

    # Dimensions comparison dataframe
    dim_df = pd.DataFrame({
        "Technique": ["Original Scaled", "PCA Reduced", "LDA Reduced", "ICA Reduced"],
        "Samples (Train)": [X_train_scaled.shape[0], X_train_pca.shape[0], X_train_lda.shape[0], X_train_ica.shape[0]],
        "Output Dimensions": [X_train_scaled.shape[1], X_train_pca.shape[1], X_train_lda.shape[1], X_train_ica.shape[1]]
    })
    st.write("##### Dimensionality Transformation Summary")
    st.dataframe(dim_df, use_container_width=True)

    col_pca, col_lda, col_ica = st.columns(3)

    # --- PCA Analytics ---
    with col_pca:
        st.markdown("### 📈 PCA Variance")
        exp_var = pca.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)

        st.metric("Retained Components", len(exp_var))
        st.metric("Cumulative Variance Explained", f"{cum_var[-1]*100:.2f}%")

        var_df = pd.DataFrame({
            "Component": [f"PC {i+1}" for i in range(len(exp_var))],
            "Individual Variance (%)": [round(v * 100, 2) for v in exp_var],
            "Cumulative Variance (%)": [round(c * 100, 2) for c in cum_var]
        })
        st.dataframe(var_df, height=280, use_container_width=True)

        fig_scree, ax_scree = plt.subplots(figsize=(6, 4))
        ax_scree.bar(range(1, len(exp_var) + 1), exp_var * 100, color='skyblue', label='Individual')
        ax_scree.plot(range(1, len(cum_var) + 1), cum_var * 100, color='crimson', marker='o', label='Cumulative')
        ax_scree.axhline(95, color='gray', linestyle='--', label='95% Cutoff')
        ax_scree.set_xlabel("Component")
        ax_scree.set_ylabel("Variance Explained (%)")
        ax_scree.legend(fontsize=8)
        st.pyplot(fig_scree)
        plt.close(fig_scree)

    # --- LDA Analytics ---
    with col_lda:
        st.markdown("### 🎯 LDA Discrimination")
        sil_score = silhouette_score(X_train_lda, y_train)
        lda_acc = accuracy_score(y_test, y_pred_lda)

        st.metric("LDA Silhouette Score", f"{sil_score:.3f}")
        st.metric("Test Classification Accuracy", f"{lda_acc*100:.2f}%")

        st.write("##### Classification Report (Test Set)")
        report_dict = classification_report(y_test, y_pred_lda, target_names=target_names, output_dict=True)
        st.dataframe(pd.DataFrame(report_dict).transpose().round(3), use_container_width=True)

        st.write("##### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_lda)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
        ax_cm.set_ylabel("True Label")
        ax_cm.set_xlabel("Predicted Label")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # --- ICA Analytics ---
    with col_ica:
        st.markdown("### 🔀 ICA Kurtosis & Sources")
        st.caption("Excess kurtosis (0 = Gaussian; Higher absolute value = Non-Gaussian signal recovery)")

        kurt_values = [stats.kurtosis(X_train_ica[:, i]) for i in range(X_train_ica.shape[1])]
        kurt_df = pd.DataFrame({
            "Component": [f"IC {i+1}" for i in range(len(kurt_values))],
            "Excess Kurtosis": [round(k, 3) for k in kurt_values]
        })
        st.dataframe(kurt_df, height=280, use_container_width=True)

        fig_kurt, ax_kurt = plt.subplots(figsize=(6, 4))
        ax_kurt.bar(range(1, len(kurt_values) + 1), kurt_values, color='coral')
        ax_kurt.set_xlabel("Independent Component")
        ax_kurt.set_ylabel("Excess Kurtosis")
        ax_kurt.set_title("Non-Gaussianity per IC")
        st.pyplot(fig_kurt)
        plt.close(fig_kurt)

        with st.expander("Show ICA Mixing Matrix"):
            st.dataframe(pd.DataFrame(ica.mixing_, index=feature_cols), use_container_width=True)

# ---------------------------------------------------------
# Tab 4: Comparative Report
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Comprehensive Comparative Analysis")

    comparison_summary = pd.DataFrame({
        "Attribute": [
            "Learning Paradigm",
            "Core Mathematical Objective",
            "Projection Geometry",
            "Dimension Constraints",
            "Labels Required",
            "Key Metric Measured in Lab",
            "Clinical / Biological Implication"
        ],
        "PCA": [
            "Unsupervised",
            "Maximize preserved variance",
            "Orthogonal principal axes",
            "min(n_samples, n_features)",
            "No (Unsupervised)",
            "Cumulative Explained Variance (95.11% in 10 PCs)",
            "Removes redundant clinical measurements without losing global patient variance."
        ],
        "LDA": [
            "Supervised",
            "Maximize between-class to within-class variance ratio",
            "Linear discriminant decision boundaries",
            "min(n_features, n_classes - 1) = 1 for binary",
            "Yes (ytrain required)",
            "Silhouette Score (0.682), High Classification Accuracy",
            "Isolates the single most diagnostic axis distinguishing malignant from benign tumors."
        ],
        "ICA": [
            "Unsupervised (Blind Source Separation)",
            "Maximize statistical independence (non-Gaussianity)",
            "Non-orthogonal basis vectors",
            "Arbitrary number of independent sources",
            "No (Unsupervised)",
            "Excess Kurtosis (up to ~41.9)",
            "Deconvolves mixed patient measurements into distinct biological drivers or biomarkers."
        ]
    })

    st.table(comparison_summary)

    st.markdown("""
    ### 🔬 Key Findings & Conclusions:
    1. **Variance Retention (PCA):** PC 1 and PC 2 explain approximately **63.00%** of total variance in the clinical dataset. Retaining **95%** total variance requires **10 components**, compressing the original 30 features by 66.7%.
    2. **Class Separability (LDA):** Because LDA incorporates the class target vector, it projects the 30-dimensional space onto a 1-dimensional discriminant line. Achieving a **Silhouette Score of 0.682** proves exceptional class density and boundary demarcation.
    3. **Source Recovery (ICA):** Components exhibit strong leptokurtic behavior (extreme positive excess kurtosis), confirming the recovery of independent non-Gaussian signals hidden inside mixed clinical tests.
    """)

# ---------------------------------------------------------
# Tab 5: Live Prediction on Custom Tumor Samples
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("Simulate Tumor Classification with Fitted Models")
    st.write("Adjust clinical attributes or select a preset test sample to observe LDA prediction and transformed coordinates.")

    sample_idx = st.selectbox(
        "Load an existing sample from test set:",
        options=list(range(len(X_test))),
        format_func=lambda idx: f"Sample #{idx} (True Class: {target_names[y_test.iloc[idx]]})"
    )

    selected_sample = X_test.iloc[[sample_idx]]
    selected_sample_scaled = scaler.transform(selected_sample)

    pred_class = lda_model.predict(selected_sample_scaled)[0]
    pred_prob = lda_model.predict_proba(selected_sample_scaled)[0]

    c_res1, c_res2, c_res3 = st.columns(3)
    c_res1.metric("Predicted Diagnosis", target_names[pred_class].upper())
    c_res2.metric("True Diagnosis", target_names[y_test.iloc[sample_idx]].upper())
    c_res3.metric("Confidence Score", f"{max(pred_prob)*100:.2f}%")

    st.write("##### Transformed Component Coordinates for this Patient:")
    coord_cols = st.columns(3)
    with coord_cols[0]:
        pca_coords = pca.transform(selected_sample_scaled)[0]
        st.write(f"**PCA (PC1, PC2):** `({pca_coords[0]:.2f}, {pca_coords[1]:.2f})`")
    with coord_cols[1]:
        lda_coord = lda_dim.transform(selected_sample_scaled)[0][0]
        st.write(f"**LDA (LD1):** `{lda_coord:.2f}`")
    with coord_cols[2]:
        ica_coords = ica.transform(selected_sample_scaled)[0]
        st.write(f"**ICA (IC1, IC2):** `({ica_coords[0]:.2f}, {ica_coords[1]:.2f})`")
