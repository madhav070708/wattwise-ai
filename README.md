# WattWise AI — Smart Energy Consumption Predictor

A beginner-friendly educational project with:
- An appliance energy and bill calculator
- A Random Forest model that predicts one future hour of energy use
- Interactive energy charts
- Model evaluation (MAE, RMSE, R²)
- CSV downloads

## Important note about the sample data
The app generates synthetic data so it can run immediately. The model metrics are only a demonstration and do not represent verified accuracy for a real building. For meaningful forecasts, train and test the model on real historical meter data.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create a virtual environment (optional but recommended).
4. Install dependencies:
   `pip install -r requirements.txt`
5. Run:
   `streamlit run app.py`
6. Open the local URL shown in the terminal.

## Deploy online
1. Create a GitHub account and a new repository named `wattwise-ai`.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Open https://share.streamlit.io and sign in with GitHub.
4. Click **Create app**, select your repository, branch (`main`), and main file (`app.py`).
5. Click **Deploy** and wait for the app to launch.

## Suggested next step: use real data
Use a public electricity-consumption dataset with a timestamp and measured kWh column. Update the feature engineering and target column to match the dataset, and evaluate on a later time period than the training period. Never claim model accuracy from synthetic data as real-world performance.
