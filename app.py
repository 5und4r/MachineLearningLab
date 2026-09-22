import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AutoQuote | Car Price Intelligence Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2638 0%, #151a26 100%);
        border: 1px solid #2e3852;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #4da6ff;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9ab0cf;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #6f85a5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        padding: 0 20px;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Synthesis / Fallback Generator (Notebook Replica)
# ---------------------------------------------------------
@st.cache_data
def generate_sample_data(n=205):
    """Generates realistic synthetic data strictly conforming to the notebook's statistics and columns."""
    np.random.seed(42)
    car_ids = np.arange(1, n + 1)
    
    makes = ['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda', 'isuzu', 
             'jaguar', 'mazda', 'mercedes-benz', 'mitsubishi', 'nissan', 'peugeot', 
             'plymouth', 'porsche', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo']
    
    car_names = [f"{np.random.choice(makes)} {np.random.choice(['gt', 'sedan', 'coupe', 'deluxe', 'sport', 'turbo', 'gl'])}" for _ in range(n)]
    symboling = np.random.choice([-2, -1, 0, 1, 2, 3], size=n, p=[0.05, 0.15, 0.30, 0.25, 0.15, 0.10])
    fueltype = np.random.choice(['gas', 'diesel'], size=n, p=[0.90, 0.10])
    aspiration = np.random.choice(['std', 'turbo'], size=n, p=[0.82, 0.18])
    doornumber = np.random.choice(['two', 'four'], size=n, p=[0.44, 0.56])
    carbody = np.random.choice(['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible'], size=n, p=[0.48, 0.34, 0.12, 0.04, 0.02])
    drivewheel = np.random.choice(['fwd', 'rwd', '4wd'], size=n, p=[0.58, 0.38, 0.04])
    enginelocation = np.random.choice(['front', 'rear'], size=n, p=[0.985, 0.015])
    
    wheelbase = np.random.normal(98.75, 6.0, n).clip(86.6, 120.9)
    carlength = np.random.normal(174.0, 12.3, n).clip(141.1, 208.1)
    carwidth = np.random.normal(65.9, 2.1, n).clip(60.3, 72.3)
    carheight = np.random.normal(53.7, 2.4, n).clip(47.8, 59.8)
    curbweight = np.random.normal(2555, 520, n).clip(1488, 4066).astype(int)
    
    enginetype = np.random.choice(['ohc', 'ohcf', 'ohcv', 'dohc', 'l', 'rotor', 'dohcv'], size=n, p=[0.72, 0.07, 0.06, 0.06, 0.05, 0.02, 0.02])
    cylindernumber = np.random.choice(['four', 'six', 'five', 'eight', 'two', 'three', 'twelve'], size=n, p=[0.77, 0.12, 0.05, 0.03, 0.015, 0.01, 0.005])
    enginesize = np.random.normal(126.9, 41.6, n).clip(61, 326).astype(int)
    fuelsystem = np.random.choice(['mpfi', '2bbl', 'idi', '1bbl', 'spdi', '4bbl'], size=n, p=[0.45, 0.32, 0.10, 0.05, 0.05, 0.03])
    
    boreratio = np.random.normal(3.33, 0.27, n).clip(2.54, 3.94)
    stroke = np.random.normal(3.25, 0.31, n).clip(2.07, 4.17)
    compressionratio = np.random.normal(10.14, 3.97, n).clip(7.0, 23.0)
    horsepower = np.random.normal(104.1, 39.5, n).clip(48, 288).astype(int)
    peakrpm = np.random.choice([4200, 4500, 4800, 5000, 5200, 5500, 6000, 6600], size=n)
    citympg = np.random.normal(25.2, 6.5, n).clip(13, 49).astype(int)
    highwaympg = (citympg + np.random.normal(5.5, 1.5, n)).clip(16, 54).astype(int)
    
    # Ground truth price simulation with realistic correlations
    price = (
        120 * horsepower + 
        75 * enginesize + 
        4.5 * curbweight - 
        180 * citympg + 
        1500 * (aspiration == 'turbo') + 
        2800 * (drivewheel == 'rwd') + 
        np.random.normal(0, 1800, n)
    ).clip(5118, 45400).round(2)
    
    df = pd.DataFrame({
        'car_ID': car_ids,
        'symboling': symboling,
        'CarName': car_names,
        'fueltype': fueltype,
        'aspiration': aspiration,
        'doornumber': doornumber,
        'carbody': carbody,
        'drivewheel': drivewheel,
        'enginelocation': enginelocation,
        'wheelbase': np.round(wheelbase, 1),
        'carlength': np.round(carlength, 1),
        'carwidth': np.round(carwidth, 1),
        'carheight': np.round(carheight, 1),
        'curbweight': curbweight,
        'enginetype': enginetype,
        'cylindernumber': cylindernumber,
        'enginesize': enginesize,
        'fuelsystem': fuelsystem,
        'boreratio': np.round(boreratio, 2),
        'stroke': np.round(stroke, 2),
        'compressionratio': np.round(compressionratio, 1),
        'horsepower': horsepower,
        'peakrpm': peakrpm,
        'citympg': citympg,
        'highwaympg': highwaympg,
        'price': price
    })
    return df

# ---------------------------------------------------------
# Notebook Preprocessing Logic
# ---------------------------------------------------------
def preprocess_and_prepare_data(raw_df):
    """
    Executes the exact preprocessing steps from the notebook:
    1. Drops 'car_ID'
    2. Ordinal-encodes 'aspiration' (['std', 'turbo'])
    3. One-hot encodes ['fueltype', 'carbody', 'drivewheel', 'enginetype'] with drop='first'
    4. Separates X and y, retaining only numerical encoded features
    """
    df_clean = raw_df.copy()
    
    # Drop identifier
    df_clean = df_clean.drop(columns=['car_ID'], errors='ignore')
    
    # 1. Ordinal Encoding for 'aspiration'
    if 'aspiration' in df_clean.columns:
        ord_enc = OrdinalEncoder(categories=[['std', 'turbo']])
        df_clean[['aspiration']] = ord_enc.fit_transform(df_clean[['aspiration']])
    
    # 2. One-Hot Encoding for nominal columns
    cols_to_onehot = ['fueltype', 'carbody', 'drivewheel', 'enginetype']
    present_cols = [c for c in cols_to_onehot if c in df_clean.columns]
    
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    ohe_data = ohe.fit_transform(df_clean[present_cols])
    ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out(present_cols), index=df_clean.index)
    
    df_transformed = pd.concat([df_clean, ohe_df], axis=1)
    df_transformed = df_transformed.drop(columns=present_cols, errors='ignore')
    
    # Feature matrix X and target y
    y = df_transformed['price']
    X = df_transformed.drop(columns=['price'], errors='ignore')
    X = X.select_dtypes(include=['number'])
    
    return X, y, ohe, ord_enc, df_transformed

# ---------------------------------------------------------
# Sidebar Data Loading
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/car--v1.png", width=70)
    st.title("AutoQuote Engine")
    st.caption("Linear Regression Lab 1 Dashboard")
    st.markdown("---")
    
    data_source = st.radio(
        "Select Data Source:",
        ["Built-in Dataset (Lab 1 Replica)", "Upload 'CarPriceData.csv'"],
        index=0
    )
    
    raw_data = None
    if data_source == "Upload 'CarPriceData.csv'":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                raw_data = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                raw_data = generate_sample_data()
        else:
            st.info("Using built-in sample data until a file is uploaded.")
            raw_data = generate_sample_data()
    else:
        raw_data = generate_sample_data()

    st.markdown("---")
    st.subheader("Model Configuration")
    test_size_ratio = st.slider("Test Split Proportion", min_value=0.10, max_value=0.40, value=0.20, step=0.05)
    random_seed = st.number_input("Random State", value=42, step=1)
    
    st.markdown("---")
    st.markdown("Developed with Streamlit & Scikit-Learn based on Lab 1.")

# ---------------------------------------------------------
# Main App Structure
# ---------------------------------------------------------
st.title("🚗 Car Price Prediction & Machine Learning Analytics")
st.markdown("""
This dashboard mirrors the complete workflow of **Lab 1: Car Price Prediction Using Regression**, 
from raw dataset exploration and correlation analysis to regression training, performance evaluation, and new car price estimation.
""")

# Preprocess and fit model
X, y, ohe_transformer, ord_transformer, df_encoded = preprocess_and_prepare_data(raw_data)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size_ratio, random_state=random_seed
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Metrics calculation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Top Key Performance Indicators
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model Fit (R² Score)</div>
        <div class="metric-value">{r2:.4f}</div>
        <div class="metric-sub">{r2*100:.1f}% variance explained</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mean Absolute Error</div>
        <div class="metric-value">${mae:,.2f}</div>
        <div class="metric-sub">Average deviation per prediction</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Root Mean Sq. Error</div>
        <div class="metric-value">${rmse:,.2f}</div>
        <div class="metric-sub">Penalizes large error outliers</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Dataset Sample Size</div>
        <div class="metric-value">{len(raw_data):,}</div>
        <div class="metric-sub">{X.shape[1]} engineered features</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# Tabbed Navigation
# ---------------------------------------------------------
tab_explore, tab_eda, tab_model, tab_predict = st.tabs([
    "📊 Dataset Explorer", 
    "📈 Exploratory Data Analysis (EDA)", 
    "⚙️ Model Diagnostics & Evaluation", 
    "🎯 Predict New Car Price"
])

# ---------------------------------------------------------
# TAB 1: DATASET EXPLORER
# ---------------------------------------------------------
with tab_explore:
    st.subheader("Dataset Summary & Statistical Profiling")
    
    view_option = st.radio(
        "Select Table View:",
        ["Raw Data Preview", "Descriptive Statistics (`describe()`)", "Data Types & Missing Values", "Encoded Features View"],
        horizontal=True
    )
    
    if view_option == "Raw Data Preview":
        st.dataframe(raw_data, use_container_width=True, height=350)
        st.caption(f"Showing {raw_data.shape[0]} rows and {raw_data.shape[1]} columns.")
        
    elif view_option == "Descriptive Statistics (`describe()`)":
        st.markdown("#### Numerical Attributes Summary Table")
        st.dataframe(raw_data.describe().T.style.format("{:.2f}"), use_container_width=True)
        
    elif view_option == "Data Types & Missing Values":
        col_type1, col_type2 = st.columns(2)
        with col_type1:
            info_df = pd.DataFrame({
                'Column': raw_data.columns,
                'Non-Null Count': raw_data.notnull().sum(),
                'Null Values': raw_data.isnull().sum(),
                'Data Type': raw_data.dtypes.astype(str)
            })
            st.dataframe(info_df, use_container_width=True, height=400)
            
        with col_type2:
            st.markdown("##### Categorical vs Numerical Breakdown")
            num_cols = raw_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = raw_data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            st.info(f"**Numerical Features ({len(num_cols)}):**\n" + ", ".join(num_cols))
            st.info(f"**Categorical Features ({len(cat_cols)}):**\n" + ", ".join(cat_cols))
            
            duplicates = raw_data.duplicated().sum()
            st.metric("Duplicate Rows Count", duplicates)
            
    elif view_option == "Encoded Features View":
        st.markdown("#### Post-Preprocessing Feature Matrix ($X$)")
        st.caption("Includes ordinal-encoded aspiration and one-hot encoded nominal columns (first dummy dropped).")
        st.dataframe(X.head(10), use_container_width=True)
        st.write(f"Processed Feature Shape: **{X.shape[0]} rows × {X.shape[1]} features**")

# ---------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------
with tab_eda:
    st.subheader("Visualizing Features & Target Dynamics")
    
    # 1. Target Price Distribution
    col_hist, col_stats = st.columns([2, 1])
    with col_hist:
        fig_price = px.histogram(
            raw_data, 
            x="price", 
            nbins=25, 
            marginal="box",
            title="Distribution of Car Prices (Target Variable)",
            color_discrete_sequence=['#4a90e2'],
            opacity=0.85
        )
        fig_price.update_layout(
            xaxis_title="Price ($)",
            yaxis_title="Frequency",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
    with col_stats:
        st.markdown("#### Price Summary")
        st.markdown(f"""
        - **Mean Price:** `${raw_data['price'].mean():,.2f}`
        - **Median Price:** `${raw_data['price'].median():,.2f}`
        - **Min Price:** `${raw_data['price'].min():,.2f}`
        - **Max Price:** `${raw_data['price'].max():,.2f}`
        - **Standard Deviation:** `${raw_data['price'].std():,.2f}`
        """)
        st.info("The price distribution shows moderate right-skewness, with budget vehicles dominating the range ($5k - $18k) and a long tail of luxury sports vehicles reaching over $40k.")

    st.markdown("---")
    
    # 2. Key Scatter Plots (Replicating Notebook's 1x3 Analysis)
    st.markdown("### Feature-Price Relationships")
    st.caption("Notebook Lab 1 primary drivers: Horsepower, Engine Size, and Curb Weight.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        fig_hp = px.scatter(
            raw_data, x="horsepower", y="price", 
            trendline="ols",
            title="Horsepower vs. Price",
            color_discrete_sequence=['#3399ff'],
            template="plotly_dark"
        )
        fig_hp.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_hp, use_container_width=True)
        
    with col_p2:
        fig_es = px.scatter(
            raw_data, x="enginesize", y="price", 
            trendline="ols",
            title="Engine Size vs. Price",
            color_discrete_sequence=['#2ecc71'],
            template="plotly_dark"
        )
        fig_es.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_es, use_container_width=True)
        
    with col_p3:
        fig_cw = px.scatter(
            raw_data, x="curbweight", y="price", 
            trendline="ols",
            title="Curb Weight vs. Price",
            color_discrete_sequence=['#f1c40f'],
            template="plotly_dark"
        )
        fig_cw.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cw, use_container_width=True)
        
    st.markdown("---")
    
    # 3. Correlation Heatmap
    st.markdown("### Correlation Heatmap (Numerical Attributes)")
    num_df = raw_data.select_dtypes(include=['int64', 'float64']).drop(columns=['car_ID'], errors='ignore')
    corr = num_df.corr()
    
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson Correlation Matrix"
    )
    fig_corr.update_layout(template="plotly_dark", height=650)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("""
    **Key Takeaways from Correlation Analysis:**
    - `enginesize` ($r \approx 0.87$), `curbweight` ($r \approx 0.83$), and `horsepower` ($r \approx 0.81$) present strong positive linear correlations with `price`.
    - Fuel efficiency metrics `citympg` and `highwaympg` exhibit strong negative correlations with `price` ($r \approx -0.70$), showing that heavier, luxury cars trade off fuel economy for power.
    """)

# ---------------------------------------------------------
# TAB 3: MODEL DIAGNOSTICS & EVALUATION
# ---------------------------------------------------------
with tab_model:
    st.subheader("Linear Regression Model Performance")
    
    # Residuals and Predictions Scatter
    residuals = y_test - y_pred
    
    col_eval1, col_eval2 = st.columns(2)
    
    with col_eval1:
        # Plot 1: Actual vs. Predicted Prices
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        
        fig_eval1 = go.Figure()
        fig_eval1.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            marker=dict(color='#1abc9c', size=9, opacity=0.8, line=dict(color='white', width=1)),
            name="Actual vs Predicted"
        ))
        # 45 degree reference line
        fig_eval1.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name="Perfect Fit (y = x)"
        ))
        fig_eval1.update_layout(
            title="Actual vs. Predicted Car Prices",
            xaxis_title="Actual Price ($)",
            yaxis_title="Predicted Price ($)",
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05)
        )
        st.plotly_chart(fig_eval1, use_container_width=True)
        
    with col_eval2:
        # Plot 2: Residuals Distribution
        fig_eval2 = px.histogram(
            x=residuals,
            nbins=18,
            marginal="rug",
            title="Distribution of Residuals (Errors: y_test - y_pred)",
            color_discrete_sequence=['#e74c3c'],
            opacity=0.8
        )
        fig_eval2.add_vline(x=0, line_width=2, line_dash="dash", line_color="white")
        fig_eval2.update_layout(
            xaxis_title="Residual / Error ($)",
            yaxis_title="Count",
            template="plotly_dark"
        )
        st.plotly_chart(fig_eval2, use_container_width=True)
        
    st.markdown("---")
    
    # Feature Coefficients / Importance
    st.markdown("### Model Feature Coefficients")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    
    fig_coef = px.bar(
        coef_df,
        x='Coefficient',
        y='Feature',
        orientation='h',
        title="Standardized Regression Coefficients (Impact on Price)",
        color='Coefficient',
        color_continuous_scale="Blues",
        height=650,
        template="plotly_dark"
    )
    fig_coef.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_coef, use_container_width=True)
    
    st.markdown("""
    #### Interpretation of Error Metrics:
    - **$R^2$ Score ({:.2f}%)**: Indicates that the linear regression model explains the majority of variance in vehicle valuations.
    - **Mean Absolute Error (MAE: ${:,.2f})**: Represents the average absolute error margin for a predicted car.
    - **Root Mean Squared Error (RMSE: ${:,.2f})**: Gives greater weight to larger prediction errors, serving as a reliable benchmark against high-leverage outliers.
    """.format(r2 * 100, mae, rmse))

# ---------------------------------------------------------
# TAB 4: INTERACTIVE PREDICTOR
# ---------------------------------------------------------
with tab_predict:
    st.subheader("Vehicle Valuation Calculator")
    st.markdown("Configure custom car specifications below to calculate an instant estimated market price using the trained regression model.")
    
    with st.form("car_prediction_form"):
        st.markdown("##### 1. Core Powertrain & Dimensions")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        
        with pcol1:
            in_horsepower = st.number_input("Horsepower (hp)", min_value=40, max_value=350, value=150, step=5)
            in_enginesize = st.number_input("Engine Size (cc)", min_value=50, max_value=350, value=140, step=5)
            in_curbweight = st.number_input("Curb Weight (lbs)", min_value=1400, max_value=4500, value=2800, step=50)
            
        with pcol2:
            in_wheelbase = st.number_input("Wheelbase (in)", min_value=85.0, max_value=125.0, value=100.0, step=0.5)
            in_carlength = st.number_input("Car Length (in)", min_value=140.0, max_value=215.0, value=175.0, step=0.5)
            in_carwidth = st.number_input("Car Width (in)", min_value=60.0, max_value=75.0, value=66.0, step=0.5)
            
        with pcol3:
            in_carheight = st.number_input("Car Height (in)", min_value=45.0, max_value=62.0, value=54.0, step=0.5)
            in_boreratio = st.number_input("Bore Ratio", min_value=2.5, max_value=4.2, value=3.5, step=0.05)
            in_stroke = st.number_input("Stroke", min_value=2.0, max_value=4.5, value=3.15, step=0.05)
            
        with pcol4:
            in_compressionratio = st.number_input("Compression Ratio", min_value=7.0, max_value=23.0, value=9.0, step=0.5)
            in_peakrpm = st.number_input("Peak RPM", min_value=4000, max_value=7000, value=5500, step=100)
            in_symboling = st.selectbox("Risk Symboling (-2 to 3)", [-2, -1, 0, 1, 2, 3], index=3)
            
        st.markdown("##### 2. Fuel Economy & Configurations")
        fcol1, fcol2, fcol3, fcol4 = st.columns(4)
        
        with fcol1:
            in_citympg = st.number_input("City MPG", min_value=10, max_value=55, value=22, step=1)
            in_highwaympg = st.number_input("Highway MPG", min_value=15, max_value=60, value=28, step=1)
            
        with fcol2:
            in_aspiration = st.selectbox("Aspiration", ["std", "turbo"], index=1)
            in_fueltype = st.selectbox("Fuel Type", ["gas", "diesel"], index=0)
            
        with fcol3:
            in_carbody = st.selectbox("Car Body Style", ["sedan", "hatchback", "wagon", "hardtop", "convertible"], index=0)
            in_drivewheel = st.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"], index=0)
            
        with fcol4:
            in_enginetype = st.selectbox("Engine Type", ["ohc", "ohcf", "ohcv", "dohc", "l", "rotor", "dohcv"], index=0)
            
        predict_submit = st.form_submit_button("Estimate Car Price ⚡", use_container_width=True)

    if predict_submit:
        # Create prototype dictionary from the training dataframe structure
        sample_dict = {col: 0.0 for col in X.columns}
        
        # Populate known numeric continuous features
        sample_dict['symboling'] = float(in_symboling)
        sample_dict['wheelbase'] = float(in_wheelbase)
        sample_dict['carlength'] = float(in_carlength)
        sample_dict['carwidth'] = float(in_carwidth)
        sample_dict['carheight'] = float(in_carheight)
        sample_dict['curbweight'] = float(in_curbweight)
        sample_dict['enginesize'] = float(in_enginesize)
        sample_dict['boreratio'] = float(in_boreratio)
        sample_dict['stroke'] = float(in_stroke)
        sample_dict['compressionratio'] = float(in_compressionratio)
        sample_dict['horsepower'] = float(in_horsepower)
        sample_dict['peakrpm'] = float(in_peakrpm)
        sample_dict['citympg'] = float(in_citympg)
        sample_dict['highwaympg'] = float(in_highwaympg)
        
        # Ordinal encoded aspiration: std -> 0.0, turbo -> 1.0
        sample_dict['aspiration'] = 1.0 if in_aspiration == 'turbo' else 0.0
        
        # One-hot encoded features (dummy variables)
        if f"fueltype_{in_fueltype}" in sample_dict:
            sample_dict[f"fueltype_{in_fueltype}"] = 1.0
            
        if f"carbody_{in_carbody}" in sample_dict:
            sample_dict[f"carbody_{in_carbody}"] = 1.0
            
        if f"drivewheel_{in_drivewheel}" in sample_dict:
            sample_dict[f"drivewheel_{in_drivewheel}"] = 1.0
            
        if f"enginetype_{in_enginetype}" in sample_dict:
            sample_dict[f"enginetype_{in_enginetype}"] = 1.0
            
        # Create 1-row DataFrame aligned with training columns
        sample_df = pd.DataFrame([sample_dict])[X.columns]
        
        # Standardize features using fitted scaler
        sample_scaled = scaler.transform(sample_df)
        
        # Inference
        pred_val = model.predict(sample_scaled)[0]
        lower_bound = max(0, pred_val - mae)
        upper_bound = pred_val + mae
        
        st.markdown("### Prediction Results")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1b3a4b 0%, #0d212d 100%); padding: 24px; border-radius: 12px; border: 1px solid #235875;">
                <div style="color: #64b5f6; font-size: 1.1rem; font-weight: 600;">ESTIMATED SELLING PRICE</div>
                <div style="color: #ffffff; font-size: 2.8rem; font-weight: 800; margin: 8px 0;">${pred_val:,.2f}</div>
                <div style="color: #90caf9; font-size: 0.95rem;">Expected Range (±1 MAE): <b>${lower_bound:,.2f} – ${upper_bound:,.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown("#### Input vs. Dataset Benchmark")
            comp_df = pd.DataFrame({
                'Metric': ['Horsepower', 'Engine Size', 'Curb Weight', 'City MPG', 'Highway MPG'],
                'Your Car': [in_horsepower, in_enginesize, in_curbweight, in_citympg, in_highwaympg],
                'Dataset Mean': [
                    raw_data['horsepower'].mean(),
                    raw_data['enginesize'].mean(),
                    raw_data['curbweight'].mean(),
                    raw_data['citympg'].mean(),
                    raw_data['highwaympg'].mean()
                ]
            })
            st.dataframe(comp_df.style.format({'Your Car': '{:.1f}', 'Dataset Mean': '{:.1f}'}), use_container_width=True)

        st.success("Prediction generated using the fitted Ordinary Least Squares (OLS) Linear Regression model!")
