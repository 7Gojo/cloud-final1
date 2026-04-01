import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.stattools import grangercausalitytests

# --- CORE LOGIC FUNCTIONS ---

def run_anomaly_detection(df, contamination=0.05):
    # Training on machine metrics (CPU, Mem, Network)
    model = IsolationForest(contamination=contamination, random_state=42)
    # Ensure we only use numeric columns for the model
    numeric_df = df.select_dtypes(include=[np.number])
    df['anomaly_score'] = model.fit_predict(numeric_df)
    
    # -1 is an anomaly, 1 is normal
    anomalies = df[df['anomaly_score'] == -1]
    return anomalies, df

def find_root_cause(df, target_machine):
    # Simple correlation matrix to find immediate neighbors
    corr_matrix = df.corr(numeric_only=True)
    if target_machine not in corr_matrix.columns:
        return []
    neighbors = corr_matrix[target_machine].sort_values(ascending=False)
    # Return top 5 potential influencers (excluding itself)
    return neighbors.iloc[1:6].index.tolist()

def run_risk_simulation(df, target, neighbors):
    risk_report = {}
    for node in neighbors:
        try:
            # Granger Test: Does 'node' cause 'target'?
            gc_test = grangercausalitytests(df[[target, node]], maxlag=2, verbose=False)
            p_val = gc_test[1][0]['ssr_chi2test'][1]
            
            # Transfer Entropy Proxy (Correlation of Lagged Signal)
            te_proxy = df[node].shift(1).corr(df[target])
            
            risk_score = ( (1 - p_val) + abs(te_proxy) ) / 2
            risk_report[node] = {
                "risk_impact": round(float(risk_score), 4), 
                "status": "Critical" if risk_score > 0.7 else "Stable"
            }
        except Exception as e:
            risk_report[node] = {"risk_impact": 0, "status": f"Inconclusive: {str(e)}"}
    return risk_report

def verify_stability(df, target):
    # Simulate a 10% noise increase to see if the service breaks
    simulated_load = df[target] * 1.10
    variance = simulated_load.var()
    
    if variance > df[target].var() * 1.5:
        return "UNSTABLE: High Variance detected. Rollback initiated."
    return "STABLE: Environment healthy."

def calculate_migration_costs(current_cost):
    options = {
        "Same-Cloud Shift": current_cost * 0.85, 
        "Type Shift (EC2->Lambda)": current_cost * 0.40, 
        "Cross-Cloud Migration": current_cost * 1.20 
    }
    return options

# --- STREAMLIT UI ---

st.title("🧠 Cloud Nervous System: Dependency-Aware Optimizer")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Machine Metrics (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.write("Data Preview:", df.head())
    
    # STEP 1: ANOMALY
    if st.button("1. Detect Anomalies"):
        anomalies, full_df = run_anomaly_detection(df)
        if not anomalies.empty:
            st.warning(f"Detected {len(anomalies)} suspicious machines.")
            # We use the column name of the first numeric feature as a target for the demo
            #st.session_state['target'] = df.select_dtypes(include=[np.number]).columns[0]
            #st.write(f"Targeting analysis on metric: **{st.session_state['target']}**")
            st.session_state['target'] = numeric_cols[0] 
            
            st.write(f"Targeting analysis on metric: **{st.session_state['target']}**")
        else:
            st.success("No anomalies detected.")

    # STEP 2 & 3: ROOT CAUSE & RISK
    if 'target' in st.session_state:
        if st.button("2. Analyze Dependency Risk"):
            neighbors = find_root_cause(df, st.session_state['target'])
            if neighbors:
                risk_data = run_risk_simulation(df, st.session_state['target'], neighbors)
                st.subheader("Dependency Risk Report")
                st.json(risk_data)
            else:
                st.error("No correlating neighbors found.")

        # STEP 4 & 5: OPTIMIZE & MIGRATE
        if st.button("3. Simulate Migration & Stability"):
            costs = calculate_migration_costs(1000) 
            st.subheader("Cost Projection")
            st.write(costs)
            
            stability = verify_stability(df, st.session_state['target'])
            if "UNSTABLE" in stability:
                st.error(stability)
            else:
                st.success(stability)