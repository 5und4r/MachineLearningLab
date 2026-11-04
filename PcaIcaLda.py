import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

# 1. CRITICAL: Set non-interactive Matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix

# Page configuration
st.set_page_config(
    page_title="Lab 4: PCA vs LDA vs ICA Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #4e73df;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CACHED DATA & MODEL PIPELINE (Prevents server hanging)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    raw_data = load_breast_cancer(as_frame=True)
    df = raw_data.frame
    X = df.drop(columns=['target'])
    y = df['target']
    return df, X, y, raw_data.target_names

@st.cache_data
def run_pipeline(test_size=0.2, random_state=42, pca_variance=0.95, ica_components=8):
    df, X, y, target_names = load_data()
    
    # Train-test split
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardization
    scaler = StandardScaler()
    XtrainScaled = scaler.fit_transform(Xtrain)
    XtestScaled = scaler.transform(Xtest)
    
    # 1. PCA
    pca = PCA(n_components=pca_variance)
    XtrainPCA = pca.fit_transform(XtrainScaled)
    XtestPCA = pca.transform(XtestScaled)
    
    # 2. LDA (Classifier & Dimensionality Reduction)
    lda_clf = LinearDiscriminantAnalysis(solver='eigen')
    lda_clf.fit(XtrainScaled, ytrain)
    y_pred_lda = lda_clf.predict(XtestScaled)
    
    lda_dim = LinearDiscriminantAnalysis(n_components=1)
    XtrainLDA = lda_dim.fit_transform(XtrainScaled, ytrain)
    lda_silhouette = silhouette_score(XtrainLDA, ytrain)
    
    # 3. ICA
    ica = FastICA(
        n_components=ica_components,
        whiten='unit-variance',
        random_state=random_state,
        max_iter=1000
    )
    XtrainICA = ica.fit_transform(XtrainScaled)
    
    # Kurtosis calculation
    kurtosis_values = [stats.kurtosis(XtrainICA[:, i]) for i in range(XtrainICA.shape[1])]
    
    return {
        "df": df,
        "X": X,
        "y": y,
        "target_names": target_names,
        "XtrainScaled": XtrainScaled,
        "XtestScaled": XtestScaled,
        "ytrain": ytrain,
        "ytest": ytest,
        "y_pred_lda": y_pred_lda,
        "pca": pca,
        "XtrainPCA": XtrainPCA,
        "XtestPCA": XtestPCA,
        "lda_clf": lda_clf,
        "lda_dim": lda_dim,
        "XtrainLDA": XtrainLDA,
        "lda_silhouette": lda_silhouette,
        "ica": ica,
        "XtrainICA": XtrainICA,
        "kurtosis_values": kurtosis_values
    }

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.title("Configuration")
st.sidebar.markdown("Tune experiment parameters below:")

test_size = st.sidebar.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State Seed", value=42, step=1)
pca_var = st.sidebar.slider("PCA Retained Variance Ratio", 0.80, 0.99, 0.95, 0.01)
ica_comp = st.sidebar.slider("ICA Components Count", 2, 15, 8, 1)

# Run cached pipeline
with st.spinner("Processing Dimensionality Reduction Pipeline..."):
    results = run_pipeline(test_size, random_state, pca_var, ica_comp)

# ---------------------------------------------------------
# MAIN INTERFACE TABS
# ---------------------------------------------------------
st.title("Comparative Analysis of PCA, LDA, and ICA")
st.markdown("**Lab 4 Implementation:** Dimensionality reduction and feature extraction on the Wisconsin Breast Cancer Dataset.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Dataset & EDA",
    "2. 2D Visual Projections",
    "3. Diagnostics & Statistical Metrics",
    "4. Comparative Evaluation"
])

# ---------------------------------------------------------
# TAB 1: DATASET & EDA
# ---------------------------------------------------------
with tab1:
    st.header("Dataset Overview & Exploratory Data Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", results["df"].shape[0])
    col2.metric("Total Columns", results["df"].shape[1])
    col3.metric("Missing Values", results["df"].isnull().sum().sum())
    col4.metric("Target Classes", f"{results['target_names'][0]} / {results['target_names'][1]}")
    
    st.subheader("Data Preview")
    st.dataframe(results["df"].head(10), use_container_width=True)
    
    with st.expander("Summary Statistics (describe)"):
        st.dataframe(results["df"].describe(), use_container_width=True)
        
    with st.expander("Full Correlation Matrix Heatmap (30x30 Features)"):
        corr_fig, ax = plt.subplots(figsize=(14, 11))
        corr_mat = results["X"].corr()
        sns.heatmap(
            corr_mat,
            cmap='coolwarm',
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=True,
            fmt='.2f',
            annot_kws={"size": 6},
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            ax=ax
        )
        ax.set_title('Correlation Matrix of Numerical Attributes', fontsize=12)
        st.pyplot(corr_fig)
        plt.close(corr_fig) # Close figure to free memory

# ---------------------------------------------------------
# TAB 2: VISUAL PROJECTIONS
# ---------------------------------------------------------
with tab2:
    st.header("Visual Comparison of Reduced Spaces")
    st.write("Replicating the lab's 3-panel scatter visualization comparing unsupervised variance, supervised separability, and non-Gaussian signal separation.")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. PCA Scatter Plot (PC1 vs PC2)
    scatter1 = axes[0].scatter(
        results["XtrainPCA"][:, 0],
        results["XtrainPCA"][:, 1],
        c=results["ytrain"],
        cmap='coolwarm',
        edgecolor='k',
        alpha=0.7
    )
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].set_title('PCA (Explained Variance)')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. LDA Scatter Plot (LD1 + Jitter)
    np.random.seed(random_state)
    jitter = np.random.normal(0, 0.05, size=results["XtrainLDA"].shape[0])
    axes[1].scatter(
        results["XtrainLDA"][:, 0],
        jitter,
        c=results["ytrain"],
        cmap='coolwarm',
        edgecolor='k',
        alpha=0.7
    )
    axes[1].set_xlabel('Linear Discriminant 1')
    axes[1].set_yticks([])
    axes[1].set_title('LDA (Class Separability)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # 3. ICA Scatter Plot (IC1 vs IC2)
    axes[2].scatter(
        results["XtrainICA"][:, 0],
        results["XtrainICA"][:, 1],
        c=results["ytrain"],
        cmap='coolwarm',
        edgecolor='k',
        alpha=0.7
    )
    axes[2].set_xlabel('Independent Component 1')
    axes[2].set_ylabel('Independent Component 2')
    axes[2].set_title('ICA (Independent Sources)')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle('Comparison of Dimensionality Reduction Techniques on Breast Cancer Dataset', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Extracted Components Shape Summary")
    comp_df = pd.DataFrame({
        "Technique": ["Original Scaled", "PCA Reduced", "LDA Reduced", "ICA Reduced"],
        "Samples": [
            results["XtrainScaled"].shape[0],
            results["XtrainPCA"].shape[0],
            results["XtrainLDA"].shape[0],
            results["XtrainICA"].shape[0]
        ],
        "Dimensions (Features)": [
            results["XtrainScaled"].shape[1],
            results["XtrainPCA"].shape[1],
            results["XtrainLDA"].shape[1],
            results["XtrainICA"].shape[1]
        ]
    })
    st.table(comp_df)

# ---------------------------------------------------------
# TAB 3: DIAGNOSTICS & STATISTICAL METRICS
# ---------------------------------------------------------
with tab3:
    st.header("Algorithmic Diagnostics")
    
    col_pca, col_lda, col_ica = st.columns(3)
    
    # PCA Metrics
    with col_pca:
        st.subheader("PCA Variance Analysis")
        exp_var = results["pca"].explained_variance_ratio_
        cum_var = np.cumsum(exp_var)
        
        st.info(f"Retained **{len(exp_var)}** Principal Components to explain **{cum_var[-1]*100:.2f}%** variance.")
        
        var_df = pd.DataFrame({
            "Component": [f"PC {i+1}" for i in range(len(exp_var))],
            "Individual Var (%)": [round(v * 100, 2) for v in exp_var],
            "Cumulative Var (%)": [round(c * 100, 2) for c in cum_var]
        })
        st.dataframe(var_df, use_container_width=True)

    # LDA Metrics
    with col_lda:
        st.subheader("LDA Discriminant Performance")
        st.success(f"Silhouette Score (1D Space): **{results['lda_silhouette']:.3f}**")
        st.write("Scores closer to 1.0 indicate strong cluster separation between benign and malignant classes.")
        
        # Test Set Classification Report
        st.markdown("**Test Set Evaluation:**")
        cm = confusion_matrix(results["ytest"], results["y_pred_lda"])
        cm_df = pd.DataFrame(
            cm,
            columns=[f"Pred {name}" for name in results["target_names"]],
            index=[f"Actual {name}" for name in results["target_names"]]
        )
        st.dataframe(cm_df)

    # ICA Metrics
    with col_ica:
        st.subheader("ICA Non-Gaussianity (Kurtosis)")
        st.write("High absolute excess kurtosis indicates strong recovery of non-Gaussian independent sources.")
        
        kurt_df = pd.DataFrame({
            "Component": [f"IC {i+1}" for i in range(len(results["kurtosis_values"]))],
            "Excess Kurtosis": [round(k, 3) for k in results["kurtosis_values"]]
        })
        st.dataframe(kurt_df, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: COMPARATIVE REPORT
# ---------------------------------------------------------
with tab4:
    st.header("Experimental Findings & Theoretical Synthesis")
    
    summary_data = {
        "Attribute": ["Type", "Primary Goal", "Projection Space", "Dimensions Retained in Lab"],
        "PCA": [
            "Unsupervised",
            "Maximize variance",
            "Orthogonal axes (Principal Components)",
            f"{results['XtrainPCA'].shape[1]} components (~{results['pca'].explained_variance_ratio_.sum()*100:.1f}%)"
        ],
        "LDA": [
            "Supervised",
            "Maximize class separability",
            "Linear Discriminant boundaries",
            "1 component (C - 1 = 1 for binary classification)"
        ],
        "ICA": [
            "Unsupervised",
            "Maximize statistical independence (non-Gaussianity)",
            "Independent sources (Non-orthogonal)",
            f"{results['XtrainICA'].shape[1]} independent components"
        ]
    }
    st.table(pd.DataFrame(summary_data))
    
    st.markdown("""
    ### Lab Analytical Observations:
    * **PCA (Variance Preservation):** PC 1 captures ~43.50% and PC 2 captures ~19.50% of the variance (~63% in 2D)[cite: 10, 12]. 10 components are needed to hit the 95% threshold[cite: 7, 12], meaning variance is spread broadly across feature dimensions[cite: 12].
    * **LDA (Supervised Separation):** Achieved a Silhouette Score of **0.682** on a single 1D projection[cite: 11, 12]. Utilizing ground-truth labels maximizes between-class variance over within-class variance, providing sharp boundaries[cite: 12].
    * **ICA (Source Separation):** Component 3 exhibits extreme excess Kurtosis (>40)[cite: 11, 12], demonstrating that FastICA successfully isolated heavy-tailed, non-Gaussian biological signals from the blended measurements[cite: 12].
    """)
