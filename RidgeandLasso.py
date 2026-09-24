import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page configuration
st.set_page_config(
    page_title="Regularized Regression Lab",
    page_icon="📊",
    layout="wide"
)

# App Title & Header
st.title("📊 Regularized Regression: Lasso & Ridge with Grid Search CV")
st.markdown("""
This application implements and compares **Linear Regression**, **Lasso (L1)**, and **Ridge (L2)** regression 
using the Scikit-Learn Diabetes dataset with 5-fold Cross-Validation and Grid Search to find optimal regularisation strengths ($\alpha$).
""")

# --- 1. Load Data with Caching ---
@st.cache_data
def load_data():
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame
    X = diabetes.data
    y = diabetes.target
    return df, X, y

df, X, y = load_data()

# --- 2. Train Models with Caching ---
@st.cache_resource
def train_and_tune_models(X, y):
    # Train-test split (80-20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Baseline Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_lin_pred = lin_reg.predict(X_test)
    mae_lin = mean_absolute_error(y_test, y_lin_pred)
    mse_lin = mean_squared_error(y_test, y_lin_pred)
    rmse_lin = np.sqrt(mse_lin)
    r2_lin = lin_reg.score(X_test, y_test)

    # 2. Lasso Grid Search
    param_grid_lasso = {'alpha': np.logspace(-4, 0, 100)}
    grid_lasso = GridSearchCV(
        estimator=Lasso(random_state=42),
        param_grid=param_grid_lasso,
        scoring='neg_mean_squared_error',
        cv=5
    )
    grid_lasso.fit(X_train, y_train)
    best_alpha_lasso = grid_lasso.best_params_['alpha']
    final_lasso = Lasso(alpha=best_alpha_lasso, random_state=42)
    final_lasso.fit(X_train, y_train)
    y_lasso_pred = final_lasso.predict(X_test)
    mae_lasso = mean_absolute_error(y_test, y_lasso_pred)
    mse_lasso = mean_squared_error(y_test, y_lasso_pred)
    rmse_lasso = np.sqrt(mse_lasso)
    r2_lasso = final_lasso.score(X_test, y_test)

    # 3. Ridge Grid Search
    param_grid_ridge = {'alpha': np.logspace(-4, 2, 100)}
    grid_ridge = GridSearchCV(
        estimator=Ridge(random_state=42),
        param_grid=param_grid_ridge,
        scoring='neg_mean_squared_error',
        cv=5
    )
    grid_ridge.fit(X_train, y_train)
    best_alpha_ridge = grid_ridge.best_params_['alpha']
    final_ridge = Ridge(alpha=best_alpha_ridge, random_state=42)
    final_ridge.fit(X_train, y_train)
    y_ridge_pred = final_ridge.predict(X_test)
    mae_ridge = mean_absolute_error(y_test, y_ridge_pred)
    mse_ridge = mean_squared_error(y_test, y_ridge_pred)
    rmse_ridge = np.sqrt(mse_ridge)
    r2_ridge = final_ridge.score(X_test, y_test)

    # Performance summary DataFrame
    perf_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Lasso Regression', 'Ridge Regression'],
        'MAE': [mae_lin, mae_lasso, mae_ridge],
        'MSE': [mse_lin, mse_lasso, mse_ridge],
        'RMSE': [rmse_lin, rmse_lasso, rmse_ridge],
        'R-squared': [r2_lin, r2_lasso, r2_ridge]
    })

    # Coefficients DataFrame
    coeff_df = pd.DataFrame({
        'Feature': X.columns,
        'Linear Regression Coeff': lin_reg.coef_,
        'Lasso Regression Coeff': final_lasso.coef_,
        'Ridge Regression Coeff': final_ridge.coef_
    })
    coeff_df['Abs_Linear_Coeff'] = np.abs(coeff_df['Linear Regression Coeff'])
    coeff_df = coeff_df.sort_values(by='Abs_Linear_Coeff', ascending=False).drop(columns=['Abs_Linear_Coeff'])

    return {
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'lin_reg': lin_reg, 'final_lasso': final_lasso, 'final_ridge': final_ridge,
        'best_alpha_lasso': best_alpha_lasso, 'best_alpha_ridge': best_alpha_ridge,
        'grid_lasso': grid_lasso, 'grid_ridge': grid_ridge,
        'y_lin_pred': y_lin_pred, 'y_lasso_pred': y_lasso_pred, 'y_ridge_pred': y_ridge_pred,
        'perf_df': perf_df, 'coeff_df': coeff_df
    }

results = train_and_tune_models(X, y)

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📁 Dataset & Exploration",
    "📈 Correlation Heatmap",
    "⚖️ Model Comparisons",
    "📉 Visualizations & Diagnostics",
    "🔮 Predict Progression"
])

# ----------------- TAB 1: DATA EXPLORATION -----------------
with tab1:
    st.subheader("Diabetes Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Training Samples", len(results['X_train']))
    col4.metric("Test Samples", len(results['X_test']))

    st.write("#### First 5 Rows")
    st.dataframe(df.head(), use_container_width=True)

    st.write("#### Descriptive Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.info("ℹ️ Note: In the Scikit-Learn Diabetes dataset, all 10 feature variables have already been mean-centered and scaled by the standard deviation times the square root of n_samples.")

# ----------------- TAB 2: CORRELATION HEATMAP -----------------
with tab2:
    st.subheader("Feature Correlation Matrix")
    corr_matrix = df.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax_corr)
    ax_corr.set_title("Correlation Matrix of Numerical Attributes", fontsize=14)
    st.pyplot(fig_corr)

# ----------------- TAB 3: MODEL COMPARISONS -----------------
with tab3:
    st.subheader("Model Performance Comparison")
    st.dataframe(results['perf_df'].style.format({
        'MAE': '{:.4f}',
        'MSE': '{:.2f}',
        'RMSE': '{:.4f}',
        'R-squared': '{:.4f}'
    }), use_container_width=True)

    st.write("#### Optimal Regularization Strengths (Alpha)")
    col_a1, col_a2 = st.columns(2)
    col_a1.success(f"**Optimal Alpha (Lasso):** `{results['best_alpha_lasso']:.4f}`")
    col_a2.success(f"**Optimal Alpha (Ridge):** `{results['best_alpha_ridge']:.4f}`")

    st.write("#### Comparison of Regression Coefficients")
    st.dataframe(results['coeff_df'].style.format({
        'Linear Regression Coeff': '{:.4f}',
        'Lasso Regression Coeff': '{:.4f}',
        'Ridge Regression Coeff': '{:.4f}'
    }), use_container_width=True)

    st.write("#### Feature Selection by Lasso (Coefficients set to 0.0)")
    zero_coeffs = results['coeff_df'][results['coeff_df']['Lasso Regression Coeff'] == 0.0]
    if not zero_coeffs.empty:
        st.warning(f"Lasso eliminated {len(zero_coeffs)} features by driving their coefficients to zero:")
        st.dataframe(zero_coeffs[['Feature', 'Lasso Regression Coeff']], use_container_width=True)
    else:
        st.info("No features had their coefficients set to zero by Lasso Regression.")

# ----------------- TAB 4: VISUALIZATIONS -----------------
with tab4:
    st.subheader("1. Actual vs. Predicted Values")
    fig_pred, axes_pred = plt.subplots(1, 2, figsize=(14, 5))
    y_test = results['y_test']

    # Lasso Scatter
    sns.scatterplot(x=y_test, y=results['y_lasso_pred'], ax=axes_pred[0], color='blue', alpha=0.8, edgecolor='w', s=70)
    axes_pred[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes_pred[0].set_xlabel('Actual Target Value')
    axes_pred[0].set_ylabel('Predicted Target Value (Lasso)')
    axes_pred[0].set_title('Lasso Regression: Actual vs. Predicted')
    axes_pred[0].grid(True, linestyle='--', alpha=0.6)

    # Ridge Scatter
    sns.scatterplot(x=y_test, y=results['y_ridge_pred'], ax=axes_pred[1], color='green', alpha=0.8, edgecolor='w', s=70)
    axes_pred[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes_pred[1].set_xlabel('Actual Target Value')
    axes_pred[1].set_ylabel('Predicted Target Value (Ridge)')
    axes_pred[1].set_title('Ridge Regression: Actual vs. Predicted')
    axes_pred[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig_pred)

    st.subheader("2. Comparison of Feature Coefficients Across Models")
    coeff_melted = results['coeff_df'].melt(id_vars='Feature', var_name='Model', value_name='Coefficient')
    fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Feature', y='Coefficient', hue='Model', data=coeff_melted, ax=ax_bar)
    plt.xticks(rotation=45, ha='right')
    ax_bar.set_title('Comparison of Feature Coefficients Across Models')
    ax_bar.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_bar)

    st.subheader("3. Cross-Validation Performance vs. Alpha Values")
    lasso_cv = pd.DataFrame(results['grid_lasso'].cv_results_)
    ridge_cv = pd.DataFrame(results['grid_ridge'].cv_results_)

    fig_cv, axes_cv = plt.subplots(1, 2, figsize=(14, 5))

    # Lasso CV Curve
    axes_cv[0].plot(lasso_cv['param_alpha'], -lasso_cv['mean_test_score'], marker='o', markersize=3, color='navy')
    axes_cv[0].set_xscale('log')
    axes_cv[0].set_xlabel('Alpha (Log Scale)')
    axes_cv[0].set_ylabel('Mean Squared Error (CV)')
    axes_cv[0].set_title('Lasso Regression: CV Performance vs. Alpha')
    axes_cv[0].axvline(results['best_alpha_lasso'], color='r', linestyle='--', label=f"Optimal Alpha: {results['best_alpha_lasso']:.4f}")
    axes_cv[0].grid(True, which="both", ls="--", alpha=0.5)
    axes_cv[0].legend()

    # Ridge CV Curve
    axes_cv[1].plot(ridge_cv['param_alpha'], -ridge_cv['mean_test_score'], marker='o', markersize=3, color='darkgreen')
    axes_cv[1].set_xscale('log')
    axes_cv[1].set_xlabel('Alpha (Log Scale)')
    axes_cv[1].set_ylabel('Mean Squared Error (CV)')
    axes_cv[1].set_title('Ridge Regression: CV Performance vs. Alpha')
    axes_cv[1].axvline(results['best_alpha_ridge'], color='r', linestyle='--', label=f"Optimal Alpha: {results['best_alpha_ridge']:.4f}")
    axes_cv[1].grid(True, which="both", ls="--", alpha=0.5)
    axes_cv[1].legend()

    plt.tight_layout()
    st.pyplot(fig_cv)

# ----------------- TAB 5: PREDICTION INTERFACE -----------------
with tab5:
    st.subheader("Predict Disease Progression on Custom Patient Features")
    st.write("Adjust the normalized features below to predict the continuous diabetes progression score:")

    cols = st.columns(5)
    input_data = {}
    for i, col_name in enumerate(X.columns):
        col_idx = i % 5
        input_data[col_name] = cols[col_idx].number_input(
            f"{col_name}",
            value=float(X.iloc[0][col_name]),
            step=0.01,
            format="%.4f"
        )

    sample_df = pd.DataFrame([input_data])

    if st.button("Calculate Predictions", type="primary"):
        pred_lin = results['lin_reg'].predict(sample_df)[0]
        pred_lasso = results['final_lasso'].predict(sample_df)[0]
        pred_ridge = results['final_ridge'].predict(sample_df)[0]

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Linear Regression", f"{pred_lin:.2f}")
        res_col2.metric("Lasso Regression", f"{pred_lasso:.2f}")
        res_col3.metric("Ridge Regression", f"{pred_ridge:.2f}")
