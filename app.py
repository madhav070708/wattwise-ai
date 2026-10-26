import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="WattWise AI | Energy Predictor", page_icon="⚡", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.stApp {background: linear-gradient(180deg, #071426 0%, #0b1830 100%); color: #eef5ff;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {background: #0c1b31;}
h1, h2, h3 {color: #f4f8ff;}
p, label, .stMarkdown {color: #d7e4f5;}
div[data-testid="stMetric"] {background:#112641; border:1px solid #25415e; padding:16px; border-radius:14px;}
div[data-testid="stMetricLabel"] {color:#a9c2dc;}
div[data-testid="stMetricValue"] {color:#69e5b1;}
.stButton button {background:#3bd69f; color:#062019; border:0; border-radius:10px; font-weight:700;}
.small-note {color:#9db5cf; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Demo dataset + model ----------
@st.cache_data
def make_demo_data():
    rng = np.random.default_rng(42)
    n = 24 * 730  # about two years of hourly sample data
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    hour = dates.hour.to_numpy()
    dow = dates.dayofweek.to_numpy()
    month = dates.month.to_numpy()
    weekend = (dow >= 5).astype(int)
    # Synthetic weather and occupancy values, for demonstration only
    temp = 25 + 7*np.sin((month-3)*2*np.pi/12) + 4*np.sin((hour-8)*2*np.pi/24) + rng.normal(0, 1.8, n)
    occupancy = np.clip(1.5 + 1.5*np.sin((hour-7)*np.pi/12) + 0.7*weekend + rng.normal(0, 0.35, n), 0, 5)
    evening_peak = np.exp(-((hour-20)/3.2)**2)
    morning_peak = np.exp(-((hour-8)/2.3)**2)
    cooling = np.maximum(temp-24, 0) * 0.10
    kwh = (0.35 + 0.24*occupancy + 0.48*evening_peak + 0.28*morning_peak
           + cooling + 0.10*weekend + rng.normal(0, 0.10, n))
    kwh = np.clip(kwh, 0.08, None)
    df = pd.DataFrame({
        "timestamp": dates, "hour": hour, "day_of_week": dow, "month": month,
        "temperature_c": temp.round(1), "occupancy_index": occupancy.round(2),
        "previous_hour_kwh": np.r_[kwh[0], kwh[:-1]].round(3), "consumption_kwh": kwh.round(3)
    })
    return df

@st.cache_resource
def train_model():
    df = make_demo_data()
    features = ["hour", "day_of_week", "month", "temperature_c", "occupancy_index", "previous_hour_kwh"]
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]
    model = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(train[features], train["consumption_kwh"])
    preds = model.predict(test[features])
    mae = mean_absolute_error(test["consumption_kwh"], preds)
    rmse = np.sqrt(mean_squared_error(test["consumption_kwh"], preds))
    r2 = r2_score(test["consumption_kwh"], preds)
    results = test[["timestamp", "consumption_kwh"]].copy()
    results["predicted_kwh"] = preds
    return model, features, mae, rmse, r2, results

df = make_demo_data()
model, features, mae, rmse, r2, test_results = train_model()

# ---------- Sidebar navigation ----------
st.sidebar.markdown("# ⚡ WattWise AI")
st.sidebar.caption("Smart Energy Consumption Predictor")
page = st.sidebar.radio("Navigate", ["Overview", "Appliance Calculator", "AI Prediction", "Model Performance", "About Project"])
st.sidebar.divider()
st.sidebar.caption("Educational demo • Uses synthetic sample data")

# ---------- Header ----------
st.markdown("<div style='color:#69e5b1;font-weight:700;letter-spacing:2px;'>SMART ENERGY ANALYTICS</div>", unsafe_allow_html=True)
st.title("WattWise AI")
st.markdown("Understand your energy use. Estimate your bill. Explore machine-learning forecasts.")
st.write("")

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample hourly average", f"{df['consumption_kwh'].mean():.2f} kWh")
    c2.metric("Sample daily average", f"{df['consumption_kwh'].mean()*24:.1f} kWh")
    c3.metric("Model MAE", f"{mae:.3f} kWh")
    c4.metric("Model R²", f"{r2:.2f}")
    st.info("Important: the included dataset is synthetic demo data, not readings from a real home or live smart meter. Metrics describe this demonstration dataset only.")
    left, right = st.columns([1.4, 1])
    with left:
        daily = df.set_index("timestamp")["consumption_kwh"].resample("D").sum().tail(30).reset_index()
        fig = px.line(daily, x="timestamp", y="consumption_kwh", title="Sample daily energy use · last 30 days",
                      labels={"timestamp":"Date", "consumption_kwh":"Energy (kWh)"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        by_hour = df.groupby("hour", as_index=False)["consumption_kwh"].mean()
        fig2 = px.bar(by_hour, x="hour", y="consumption_kwh", title="Average sample use by hour",
                      labels={"hour":"Hour of day", "consumption_kwh":"Average kWh"})
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### What you can do")
    a, b, c = st.columns(3)
    a.markdown("**🔌 Calculate**\n\nEstimate usage from appliance power ratings and daily operating hours.")
    b.markdown("**🤖 Predict**\n\nUse a trained Random Forest model to estimate one future hour.")
    c.markdown("**📊 Evaluate**\n\nReview model errors on a held-out later time period.")

elif page == "Appliance Calculator":
    st.subheader("🔌 Appliance-based energy calculator")
    st.write("Enter the appliances you use. The estimate uses power × quantity × hours × days.")
    tariff = st.number_input("Electricity tariff (₹ per kWh)", min_value=0.0, max_value=100.0, value=7.5, step=0.5)
    days = st.slider("Number of days", 1, 31, 30)
    appliances = [
        ("LED bulb", 9, 6), ("Ceiling fan", 75, 8), ("Television", 100, 4),
        ("Refrigerator", 150, 10), ("Air conditioner", 1500, 6), ("Laptop", 65, 5),
        ("Washing machine", 500, 1), ("Water heater", 2000, 0.5)
    ]
    st.caption("Power values are examples. Change them to match the label on your actual appliance.")
    rows = []
    cols = st.columns(2)
    for i, (name, watts, hours) in enumerate(appliances):
        with cols[i % 2]:
            st.markdown(f"**{name}**")
            qcol, wcol, hcol = st.columns(3)
            qty = qcol.number_input("Qty", min_value=0, max_value=30, value=1 if i < 4 else 0, key=f"qty_{i}")
            power = wcol.number_input("Watts", min_value=0, max_value=20000, value=watts, step=10, key=f"watts_{i}")
            hrs = hcol.number_input("Hrs/day", min_value=0.0, max_value=24.0, value=float(hours), step=0.5, key=f"hours_{i}")
            rows.append({"Appliance": name, "Quantity": qty, "Power (W)": power, "Hours/day": hrs,
                         "Daily kWh": qty * power * hrs / 1000})
    usage_df = pd.DataFrame(rows)
    daily_kwh = usage_df["Daily kWh"].sum()
    total_kwh = daily_kwh * days
    bill = total_kwh * tariff
    m1, m2, m3 = st.columns(3)
    m1.metric("Daily consumption", f"{daily_kwh:.2f} kWh")
    m2.metric(f"Consumption in {days} days", f"{total_kwh:.2f} kWh")
    m3.metric("Estimated energy cost", f"₹{bill:,.2f}")
    st.markdown("### Appliance breakdown")
    st.dataframe(usage_df, use_container_width=True, hide_index=True)
    active = usage_df[usage_df["Daily kWh"] > 0]
    if not active.empty:
        fig = px.pie(active, names="Appliance", values="Daily kWh", hole=0.45, title="Share of daily appliance consumption")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download appliance estimate (CSV)", usage_df.to_csv(index=False).encode("utf-8"), "appliance_energy_estimate.csv", "text/csv")
    st.caption("Bill estimate excludes fixed charges, taxes, slabs, subsidies and other utility fees.")

elif page == "AI Prediction":
    st.subheader("🤖 Machine-learning energy prediction")
    st.write("The Random Forest model predicts an estimated energy value for one hour based on time, temperature, occupancy index and the previous hour's usage.")
    st.warning("This starter model is trained on synthetic data. Use real historical meter readings to make meaningful predictions for a real building.")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            date = st.date_input("Forecast date", value=pd.Timestamp.today().date())
            hour = st.slider("Hour of day", 0, 23, 19)
        with col2:
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=30.0, step=0.5)
            occupancy = st.slider("Occupancy index (0–5)", 0.0, 5.0, 3.0, 0.5)
        with col3:
            previous = st.number_input("Previous hour usage (kWh)", min_value=0.0, max_value=1000.0, value=0.8, step=0.1)
            tariff = st.number_input("Tariff (₹ per kWh)", min_value=0.0, max_value=100.0, value=7.5, step=0.5, key="pred_tariff")
        submitted = st.form_submit_button("Predict energy consumption")
    if submitted:
        features_row = pd.DataFrame([{
            "hour": hour, "day_of_week": pd.Timestamp(date).dayofweek, "month": pd.Timestamp(date).month,
            "temperature_c": temperature, "occupancy_index": occupancy, "previous_hour_kwh": previous
        }])[features]
        prediction = max(0.0, float(model.predict(features_row)[0]))
        p1, p2, p3 = st.columns(3)
        p1.metric("Predicted next-hour use", f"{prediction:.3f} kWh")
        p2.metric("Estimated cost for that hour", f"₹{prediction*tariff:.2f}")
        p3.metric("Equivalent average load", f"{prediction*1000:.0f} W")
        st.success("Prediction generated. This is a model estimate, not a live meter reading or a guaranteed result.")
        st.markdown("### How to interpret this")
        st.write("The result is an estimate for the selected hour. Accuracy depends on the quality and relevance of the training data and the input values.")
        export = pd.DataFrame([{"date": str(date), "hour": hour, "temperature_c": temperature, "occupancy_index": occupancy,
                                "previous_hour_kwh": previous, "predicted_kwh": prediction, "tariff_inr_per_kwh": tariff,
                                "estimated_cost_inr": prediction*tariff}])
        st.download_button("Download prediction (CSV)", export.to_csv(index=False).encode("utf-8"), "energy_prediction.csv", "text/csv")

elif page == "Model Performance":
    st.subheader("📊 Model evaluation")
    st.write("A Random Forest regression model is trained on the first 80% of the synthetic timeline and evaluated on the later 20%.")
    st.info("These are demo-only metrics from synthetic data. They should not be presented as verified performance on real electricity usage.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean Absolute Error (MAE)", f"{mae:.3f} kWh")
    m2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.3f} kWh")
    m3.metric("R² score", f"{r2:.3f}")
    sample = test_results.tail(240)
    fig = px.line(sample, x="timestamp", y=["consumption_kwh", "predicted_kwh"],
                  labels={"timestamp":"Time", "value":"Energy (kWh)", "variable":"Series"},
                  title="Actual vs predicted · sample of held-out hours")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download test predictions (CSV)", test_results.to_csv(index=False).encode("utf-8"), "model_test_predictions.csv", "text/csv")
    st.markdown("**Metrics guide**")
    st.write("- MAE: average absolute difference between actual and predicted consumption; lower is better.")
    st.write("- RMSE: penalizes larger prediction errors more strongly; lower is better.")
    st.write("- R²: compares model performance with a simple mean baseline; values closer to 1 generally indicate a better fit.")

elif page == "About Project":
    st.subheader("About the project")
    st.markdown("""
**Problem statement**  
Households and buildings often cannot easily estimate future electricity use or understand which appliances contribute most to their bills.

**Objectives**
- Calculate energy use from appliance power and operating time.
- Estimate energy cost using an editable tariff.
- Demonstrate machine-learning forecasting.
- Visualize energy patterns and model evaluation.

**Methodology**
1. Generate a sample hourly dataset for demonstration.
2. Prepare time, temperature, occupancy and previous-hour usage features.
3. Train a Random Forest regression model on the earlier time period.
4. Evaluate it on a later, held-out time period using MAE, RMSE and R².
5. Use the trained model to estimate consumption for a user-selected hour.

**Limitations**
- The bundled dataset is synthetic and does not represent an actual household.
- Predictions are educational estimates, not live measurements.
- The bill calculator does not model local utility billing slabs or fixed fees.

**Future scope**
- Train on real smart-meter data.
- Connect an IoT energy meter.
- Add household profiles, monthly reports and anomaly alerts.
- Compare Random Forest with other forecasting models.
""")
    st.markdown("**Technology stack:** Python · Streamlit · Pandas · NumPy · Scikit-learn · Plotly")
    st.caption("WattWise AI — student project demonstration")
