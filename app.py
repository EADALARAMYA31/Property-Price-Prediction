import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from db import init_db, register_user, login_user, save_prediction, get_history

init_db()
# ====================================
# DATABASE
# ====================================


# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Smart Property Prediction",
    page_icon="🏠",
    layout="wide"
)

# ====================================
# AUTH FUNCTIONS
# ====================================


# ====================================
# SESSION STATE
# ====================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# ====================================
# AUTH PAGE
# ====================================

def auth_page():

    st.title("🔐 Login / Register System")

    mode = st.radio("Choose Action", ["Login", "Register"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if mode == "Register":
        if st.button("Create Account"):
            if register_user(username, password):
                st.success("Account created!")
            else:
                st.error("User already exists")

    else:
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login success")
                st.rerun()
            else:
                st.error("Invalid login")

# ====================================
# LOGOUT
# ====================================

def logout():
    st.sidebar.write(f"👤 {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# ====================================
# STOP IF NOT LOGGED IN
# ====================================

if not st.session_state.logged_in:
    auth_page()
    st.stop()

# ====================================
# DATA
# ====================================

housing = fetch_california_housing(as_frame=True)
df = housing.frame

features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]

X = df[features]
y = df["MedHouseVal"] * 100000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====================================
# MODELS
# ====================================

lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ====================================
# SIDEBAR NAVIGATION (EXPANDED)
# ====================================

st.sidebar.title("🏠 Navigation")
logout()

page = st.sidebar.radio(
    "Go To",
    [
        "Home",
        "Dataset",
        "Visualization",
        "Prediction",
        "History",
        "Model Performance",
        "Market Insights",
        "Analytics Dashboard",
        "Project Info"
    ]
)

# ====================================
# HOME
# ====================================

if page == "Home":
    st.title("🏠 Smart Property Prediction System")

    st.success(f"Welcome {st.session_state.username} 👋")

    st.markdown("### 📊 Real Estate Dashboard Overview")

    # =========================
    # METRICS ROW 1
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Houses",
        len(df),
        "Dataset Size"
    )

    col2.metric(
        "Average Price",
        f"${df['MedHouseVal'].mean()*100000:,.0f}",
        "Market Avg"
    )

    col3.metric(
        "Maximum Price",
        f"${df['MedHouseVal'].max()*100000:,.0f}",
        "Luxury Segment"
    )

    st.markdown("---")

    # =========================
    # METRICS ROW 2 (EXPANSION)
    # =========================
    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Average Income",
        round(df["MedInc"].mean(), 2),
        "Income Level"
    )

    col5.metric(
        "Average Rooms",
        round(df["AveRooms"].mean(), 2),
        "Housing Structure"
    )

    col6.metric(
        "Average Population",
        round(df["Population"].mean(), 0),
        "Area Density"
    )

    st.markdown("---")

    # =========================
    # QUICK INSIGHTS SECTION
    # =========================

    st.subheader("📌 Quick Insights")

    st.info("💡 Higher Median Income areas generally have higher house prices.")
    st.warning("⚠ Some areas show extreme price variation due to location density.")
    st.success("📈 Random Forest model gives the most accurate predictions.")

    # =========================
    # MINI CHARTS
    # =========================

    st.subheader("📊 Quick Market View")

    col7, col8 = st.columns(2)

    with col7:
        fig1 = px.histogram(
            df,
            x="MedHouseVal",
            title="House Price Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col8:
        fig2 = px.scatter(
            df,
            x="MedInc",
            y="MedHouseVal",
            color="HouseAge",
            title="Income vs House Price"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # =========================
    # FOOTER INFO
    # =========================

    st.markdown("### 🚀 System Status")

    st.success("✔ Dataset Loaded Successfully")
    st.success("✔ Models Trained")
    st.success("✔ Prediction Engine Active")

# ====================================
# DATASET
# ====================================
elif page == "Dataset":
    st.header("📊 Dataset Overview")

    st.markdown("### 🔍 Quick Preview")
    st.dataframe(df.head(30), use_container_width=True)

    st.markdown("---")

    # =========================
    # DATASET SHAPE
    # =========================
    st.subheader("📦 Dataset Structure")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.markdown("---")

    # =========================
    # DATA INFO
    # =========================
    st.subheader("ℹ️ Data Information")

    buffer = df.info(buf=None)
    st.write(df.dtypes)

    st.markdown("---")

    # =========================
    # MISSING VALUES
    # =========================
    st.subheader("❗ Missing Values Check")

    missing = df.isnull().sum().to_frame()
    missing.columns = ["Missing Values"]

    st.dataframe(missing, use_container_width=True)

    st.markdown("---")

    # =========================
    # STATISTICS
    # =========================
    st.subheader("📊 Statistical Summary")

    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")

    # =========================
    # COLUMN INSIGHTS
    # =========================
    st.subheader("📌 Column Insights")

    selected_col = st.selectbox("Select Column", df.columns)

    col1, col2, col3 = st.columns(3)

    col1.metric("Mean", round(df[selected_col].mean(), 2))
    col2.metric("Min", round(df[selected_col].min(), 2))
    col3.metric("Max", round(df[selected_col].max(), 2))

    # Histogram
    st.markdown("### 📈 Distribution")
    fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
    st.plotly_chart(fig, use_container_width=True)

# ====================================
# VISUALIZATION
# ====================================

elif page == "Visualization":
    st.header("📈 Visual Analytics Dashboard")

    st.markdown("### 📊 Income vs House Price Relationship")

    st.info("This chart shows how Median Income affects House Prices in California.")

    # =========================
    # SCATTER CHART (ADDED)
    # =========================

    

    st.markdown("---")

    st.success("📌 Insight: Higher income areas generally have higher property values.")

    st.markdown("""
    ### 📈 Key Observations
    - Strong positive correlation between income and price  
    - Middle-income regions show high variability  
    - Luxury housing clusters at high income levels  
    """)
# ====================================
# PREDICTION
# ====================================

elif page == "Prediction":
    st.header("💰 Property Price Prediction")

    st.markdown("### 🧠 Select Model & Enter Property Details")

    # =========================
    # MODEL SELECTION
    # =========================
    model_name = st.selectbox(
        "Choose Model",
        ["Linear Regression", "Decision Tree", "Random Forest"]
    )

    model = {
        "Linear Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf
    }[model_name]

    st.markdown("---")

    st.subheader("🏡 Enter Property Features")

    # =========================
    # INPUTS (IMPROVED UI)
    # =========================

    col1, col2 = st.columns(2)

    with col1:
        medinc = st.number_input("Median Income", value=float(df["MedInc"].mean()))
        house_age = st.number_input("House Age", value=float(df["HouseAge"].mean()))
        rooms = st.number_input("Average Rooms", value=float(df["AveRooms"].mean()))

    with col2:
        bedrooms = st.number_input("Average Bedrooms", value=float(df["AveBedrms"].mean()))
        population = st.number_input("Population", value=float(df["Population"].mean()))
        occupancy = st.number_input("Average Occupancy", value=float(df["AveOccup"].mean()))

    inputs = [medinc, house_age, rooms, bedrooms, population, occupancy]

    st.markdown("---")

    # =========================
    # PREDICT BUTTON
    # =========================

    if st.button("🚀 Predict Price"):

        pred = model.predict([inputs])[0]

        st.success(f"🏠 Predicted Property Price: ${pred:,.2f}")

        # =========================
        # SAVE TO DATABASE
        # =========================
        save_prediction(
    st.session_state.username,
    model_name,
    float(pred)
)

        # =========================
        # EXTRA INSIGHT
        # =========================
        st.info("📌 Prediction saved to history database successfully!")

        st.markdown("### 📊 Quick Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Model Used", model_name)
        col2.metric("Predicted Price", f"${pred:,.0f}")
        col3.metric("User", st.session_state.username)

# ====================================
# HISTORY (DB)
# ====================================

elif page == "History":
    st.header("📜 Prediction History")

    st.markdown("### 🗄️ Your Saved Predictions from Database")

    # =========================
    # FETCH DATA
    # =========================
    rows = get_history(st.session_state.username)

    # =========================
    # EMPTY STATE
    # =========================
    if not rows:
        st.info("No history yet. Make a prediction first 🚀")

    else:
        df_hist = pd.DataFrame(rows, columns=["Model", "Prediction"])

        # =========================
        # TABLE VIEW
        # =========================
        st.subheader("📋 All Predictions")
        st.dataframe(df_hist, use_container_width=True)

        st.markdown("---")

        # =========================
        # ANALYTICS SECTION
        # =========================
        st.subheader("📊 Prediction Analytics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", len(df_hist))
        col2.metric("Avg Prediction", f"${df_hist['Prediction'].mean():,.2f}")
        col3.metric("Max Prediction", f"${df_hist['Prediction'].max():,.2f}")

        st.markdown("---")

        # =========================
        # MODEL USAGE DISTRIBUTION
        # =========================
        st.subheader("🤖 Model Usage Distribution")

        model_counts = df_hist["Model"].value_counts().reset_index()
        model_counts.columns = ["Model", "Count"]

        fig = px.bar(
            model_counts,
            x="Model",
            y="Count",
            title="Most Used Models"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # =========================
        # DOWNLOAD BUTTON
        # =========================
        csv = df_hist.to_csv(index=False)

        st.download_button(
            "📥 Download History CSV",
            csv,
            "prediction_history.csv",
            "text/csv"
        )

# ====================================
# MODEL PERFORMANCE
# ====================================

elif page == "Model Performance":
    st.header("📊 Model Performance Dashboard")

    st.markdown("### 🤖 Compare ML Models on Accuracy & Error Metrics")

    # =========================
    # MODELS
    # =========================
    models = {
        "Linear Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf
    }

    data = []

    for name, model in models.items():
        pred = model.predict(X_test)
        data.append([
            name,
            r2_score(y_test, pred),
            mean_squared_error(y_test, pred)
        ])

    perf_df = pd.DataFrame(data, columns=["Model", "R2 Score", "MSE"])

    # =========================
    # TABLE VIEW
    # =========================
    st.subheader("📋 Performance Table")
    st.dataframe(perf_df, use_container_width=True)

    st.markdown("---")

    # =========================
    # METRICS HIGHLIGHT
    # =========================
    st.subheader("🏆 Best Model Insights")

    best_r2_model = perf_df.loc[perf_df["R2 Score"].idxmax(), "Model"]
    best_mse_model = perf_df.loc[perf_df["MSE"].idxmin(), "Model"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Best R² Model", best_r2_model)
    col2.metric("Lowest Error Model", best_mse_model)
    col3.metric("Total Models", len(perf_df))

    st.markdown("---")

    # =========================
    # R2 SCORE VISUALIZATION
    # =========================
    st.subheader("📈 R² Score Comparison")

    fig1 = px.bar(
        perf_df,
        x="Model",
        y="R2 Score",
        title="Model Accuracy Comparison (R² Score)"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # =========================
    # ERROR VISUALIZATION
    # =========================
    st.subheader("📉 Error Comparison (MSE)")

    fig2 = px.bar(
        perf_df,
        x="Model",
        y="MSE",
        title="Model Error Comparison (MSE)"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # =========================
    # INSIGHTS
    # =========================
    st.success(f"📌 Best performing model based on R² Score: {best_r2_model}")
    st.info("💡 Random Forest usually performs best for complex housing datasets due to non-linear patterns.")

# ====================================
# MARKET INSIGHTS
# ====================================

elif page == "Market Insights":
    st.header("📊 Market Insights Dashboard")

    st.markdown("### 🏘️ Key Real Estate Market Indicators")

    # =========================
    # METRICS (YOUR ORIGINAL + CLEAN UI)
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Income", round(df["MedInc"].mean(), 2))
    col2.metric("Avg Population", round(df["Population"].mean()))
    col3.metric("Avg Occupancy", round(df["AveOccup"].mean(), 2))

    st.markdown("---")

    # =========================
    # EXTRA MARKET METRICS
    # =========================
    st.subheader("📈 Extended Market KPIs")

    col4, col5, col6 = st.columns(3)

    col4.metric("Max Income", round(df["MedInc"].max(), 2))
    col5.metric("Min Income", round(df["MedInc"].min(), 2))
    col6.metric("Avg House Value", f"${df['MedHouseVal'].mean()*100000:,.0f}")

    st.markdown("---")

    # =========================
    # INCOME DISTRIBUTION
    # =========================
    st.subheader("💰 Income Distribution")

    fig1 = px.histogram(
        df,
        x="MedInc",
        title="Distribution of Median Income"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # =========================
    # POPULATION VS PRICE
    # =========================
    st.subheader("🏡 Population vs House Price")

    fig2 = px.scatter(
        df,
        x="Population",
        y="MedHouseVal",
        color="HouseAge",
        title="Population Impact on House Prices"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # =========================
    # INSIGHTS
    # =========================
    st.success("📌 Insight: Areas with moderate population density often show balanced pricing.")
    st.info("📊 Income level is the strongest driver of housing price variations.")
    st.warning("⚠ Extremely high population regions may not always mean higher prices.")

# ====================================
# ANALYTICS DASHBOARD (NEW PAGE)
# ====================================

elif page == "Analytics Dashboard":
    st.header("📊 Advanced Analytics Dashboard")

    st.markdown("### 🔬 Deep Data Insights for Housing Market")

    # =========================
    # CORRELATION HEATMAP (YOUR ORIGINAL)
    # =========================
    st.subheader("📌 Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig1 = px.imshow(
        corr,
        text_auto=True,
        title="Feature Correlation Matrix"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # =========================
    # STRONG CORRELATIONS
    # =========================
    st.subheader("📈 Strong Relationships with House Price")

    corr_target = corr["MedHouseVal"].drop("MedHouseVal").sort_values(ascending=False)

    fig2 = px.bar(
        corr_target,
        title="Correlation with House Price",
        labels={"value": "Correlation", "index": "Features"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # =========================
    # INCOME VS PRICE (KEY ANALYTICS)
    # =========================
    st.subheader("💰 Income vs House Price Trend")

    fig3 = px.scatter(
        df,
        x="MedInc",
        y="MedHouseVal",
        color="HouseAge",
        title="Income Impact on House Prices"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # =========================
    # PRICE DISTRIBUTION
    # =========================
    st.subheader("🏡 House Price Distribution")

    fig4 = px.histogram(
        df,
        x="MedHouseVal",
        title="Distribution of House Prices"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # =========================
    # INSIGHTS
    # =========================
    st.success("📌 Insight: Median Income is the strongest predictor of house prices.")
    st.info("📊 Features like Rooms and Occupancy have moderate influence.")
    st.warning("⚠ Population has weaker correlation compared to income factors.")

# ====================================
# PROJECT INFO
# ====================================

elif page == "Project Info":
    st.header("📌 Project Information")

    st.markdown("### 🏠 Smart Property Price Prediction System")

    st.markdown("---")

    # =========================
    # PROJECT SUMMARY
    # =========================
    st.subheader("📖 Project Overview")

    st.info("""
    This project is a machine learning-based web application that predicts
    property prices using California housing dataset features such as income,
    house age, population, and occupancy.
    """)

    st.markdown("---")

    # =========================
    # TECH STACK
    # =========================
    st.subheader("🛠️ Technology Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - Python 🐍  
        - Streamlit 🌐  
        - Pandas 📊  
        - Scikit-Learn 🤖  
        """)

    with col2:
        st.markdown("""
        - Plotly 📈  
        - SQLite 🗄️  
        - NumPy 🔢  
        - Machine Learning Models  
        """)

    st.markdown("---")

    # =========================
    # ML MODELS
    # =========================
    st.subheader("🤖 Machine Learning Models Used")

    st.success("""
    ✔ Linear Regression – Baseline model for prediction  
    ✔ Decision Tree Regressor – Captures non-linear patterns  
    ✔ Random Forest Regressor – Best performing ensemble model  
    """)

    st.markdown("---")

    # =========================
    # SYSTEM FEATURES
    # =========================
    st.subheader("⚙️ System Features")

    st.markdown("""
    - 🔐 User Authentication (Login/Register)  
    - 🗄️ SQLite Database Integration  
    - 💰 Real-time Property Price Prediction  
    - 📊 Data Visualization Dashboard  
    - 📜 Prediction History Tracking  
    - 📈 Model Performance Comparison  
    - 🧠 Market Insights Analytics  
    """)

    st.markdown("---")

    # =========================
    # FUTURE ENHANCEMENTS
    # =========================
    st.subheader("🚀 Future Enhancements")

    st.warning("""
    - Cloud Database (PostgreSQL / Firebase)  
    - User Role Management (Admin/User)  
    - AI-based Recommendation System  
    - Real-time Housing Market Data Integration  
    - Deployment on AWS / Render / Streamlit Cloud  
    """)

    st.markdown("---")

    # =========================
    # FOOTER
    # =========================
    st.success("💡 This project demonstrates end-to-end Machine Learning + Full Stack Web App development.")