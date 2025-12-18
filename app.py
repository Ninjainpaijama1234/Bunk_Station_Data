import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import os

# --- Page Config ---
st.set_page_config(page_title="Bunk Station Operational Dashboard", layout="wide")

# --- Title ---
st.title("📊 Bunk Station: Operational Command Center")
st.markdown("""
> **Goal:** Move from "Data Viewing" to "Decision Making". Use the interactive controls below to simulate outcomes.
""")

# --- 1. Data Loading ---
@st.cache_data
def load_data():
    file_path = "Bunk_Station_Daily_Sales_Full_Year.csv"
    if not os.path.exists(file_path):
        st.error(f"❌ File {file_path} not found.")
        return None
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Feature Engineering
        df['Day_of_Week'] = df['Date'].dt.day_name()
        df['Month'] = df['Date'].dt.month_name()
        df['Month_Num'] = df['Date'].dt.month
        df['Day_Index'] = df['Date'].dt.dayofweek
        df['Day_of_Year'] = df['Date'].dt.dayofyear # Consistent naming
        df['Is_Weekend'] = df['Day_Index'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Safe Conversion Rate
        df['Conversion_Rate'] = df.apply(
            lambda row: (row['Orders'] / row['Footfall'] * 100) if row['Footfall'] > 0 else 0, 
            axis=1
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # Sidebar Filters
    st.sidebar.header("🗓️ Timeframe Selector")
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    date_range = st.sidebar.date_input("Filter Data", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) == 2:
        mask = (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))
        df_filtered = df.loc[mask]
    else:
        df_filtered = df

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Descriptive & Trends", "🤖 AI Operational Insights", "💰 Financial Simulator"])

    # ==========================================
    # TAB 1: DESCRIPTIVE (INTERACTIVE)
    # ==========================================
    with tab1:
        st.subheader("Operational Pulse")
        
        # 1. KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Revenue", f"AED {df_filtered['Revenue_AED'].sum():,.0f}")
        kpi2.metric("Footfall", f"{df_filtered['Footfall'].sum():,.0f}")
        kpi3.metric("Conversion", f"{df_filtered['Conversion_Rate'].mean():.1f}%")
        kpi4.metric("Avg Ticket", f"AED {df_filtered['Avg_Ticket_AED'].mean():.1f}")

        st.markdown("---")

        # 2. Interactive Heatmap
        col_desc1, col_desc2 = st.columns([2, 1])
        
        with col_desc1:
            st.markdown("### 🗓️ Daily Performance Heatmap")
            # Metric Selector
            hm_metric = st.radio("Select Metric to Visualize:", ["Revenue_AED", "Footfall", "Avg_Ticket_AED", "Conversion_Rate"], horizontal=True)
            
            pivot = df_filtered.pivot_table(index='Day_of_Week', columns='Month', values=hm_metric, aggfunc='mean')
            # Ordering
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            pivot = pivot.reindex(days).reindex(columns=[m for m in months if m in pivot.columns])
            
            fig_hm = px.imshow(pivot, text_auto=".0f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(fig_hm, use_container_width=True)

        with col_desc2:
            st.markdown("### 📊 Consistency Check")
            st.caption("Are Fridays consistently good, or risky? (Wider box = More volatile)")
            fig_box = px.box(df_filtered, x="Day_of_Week", y="Revenue_AED", 
                             color="Day_of_Week", 
                             category_orders={"Day_of_Week": days})
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    # ==========================================
    # TAB 2: AI OPERATIONAL INSIGHTS
    # ==========================================
    with tab2:
        st.subheader("🤖 AI Command Center")
        
        col_ai1, col_ai2 = st.columns([1, 2])

        # 1. "What-If" Calculator (Left Column)
        with col_ai1:
            st.markdown("### 🧮 Revenue Simulator")
            st.info("Train the AI on history, then predict the impact of changes.")
            
            # Train Model on the fly
            features = ['Footfall', 'Avg_Ticket_AED', 'Conversion_Rate', 'Day_Index', 'Is_Weekend']
            X = df_filtered[features].fillna(0)
            y = df_filtered['Revenue_AED']
            model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
            
            # Inputs
            base_foot = df_filtered['Footfall'].mean()
            base_ticket = df_filtered['Avg_Ticket_AED'].mean()
            
            new_foot = st.number_input("Target Daily Footfall", value=int(base_foot), step=10)
            new_ticket = st.number_input("Target Ticket Size (AED)", value=float(base_ticket), step=1.0)
            
            # Prediction
            # Assume weekday average (Day 2 = Wed) for generic prediction
            pred_rev = model.predict([[new_foot, new_ticket, (new_foot/base_foot)*df_filtered['Conversion_Rate'].mean(), 2, 0]])[0]
            
            st.metric("Predicted Daily Revenue", f"AED {pred_rev:,.0f}", 
                      delta=f"{pred_rev - df_filtered['Revenue_AED'].mean():,.0f} vs Avg")
            
            st.markdown("**Driver Importance:**")
            imp = pd.DataFrame({'Feat': features, 'Imp': model.feature_importances_}).sort_values('Imp', ascending=False).head(3)
            st.table(imp.style.format({'Imp': '{:.1%}'}))

        # 2. Anomaly Detection (Right Column)
        with col_ai2:
            st.markdown("### 🚨 Missed Opportunities (Anomaly Detection)")
            st.caption("Dates where Footfall was High, but Revenue was unexpectedly Low (Possible Service Failure).")
            
            # Isolation Forest to find weird days
            iso_df = df_filtered[['Date', 'Footfall', 'Revenue_AED', 'Conversion_Rate']].copy()
            model_iso = IsolationForest(contamination=0.05, random_state=42)
            iso_df['Anomaly'] = model_iso.fit_predict(iso_df[['Footfall', 'Revenue_AED']])
            
            # Filter for "High Footfall / Low Revenue" anomalies
            avg_foot = iso_df['Footfall'].mean()
            avg_rev = iso_df['Revenue_AED'].mean()
            
            # Logic: Anomaly (-1) AND Footfall > Avg AND Revenue < Avg
            problem_days = iso_df[
                (iso_df['Anomaly'] == -1) & 
                (iso_df['Footfall'] > avg_foot) & 
                (iso_df['Revenue_AED'] < avg_rev)
            ].sort_values('Date')
            
            if not problem_days.empty:
                st.error(f"⚠️ Found {len(problem_days)} days requiring investigation:")
                st.dataframe(problem_days.style.format({
                    'Date': '{:%Y-%m-%d}', 
                    'Revenue_AED': 'AED {:,.0f}',
                    'Conversion_Rate': '{:.1f}%'
                }), use_container_width=True)
                
                # Chart
                fig_anom = px.scatter(iso_df, x="Footfall", y="Revenue_AED", 
                                      color=iso_df['Anomaly'].astype(str),
                                      color_discrete_map={'-1': 'red', '1': 'lightgray'},
                                      title="Red Dots = Anomalies (Check Operations)",
                                      hover_data=['Date'])
                st.plotly_chart(fig_anom, use_container_width=True)
            else:
                st.success("✅ No major 'High Traffic / Low Revenue' anomalies detected in this range.")

    # ==========================================
    # TAB 3: FINANCIAL SIMULATOR (INTERACTIVE)
    # ==========================================
    with tab3:
        st.subheader("💰 Profitability Sandbox")
        
        # 1. Controls
        with st.expander("🛠️ Control Panel: Adjust Your Cost Structure", expanded=True):
            c1, c2, c3 = st.columns(3)
            fixed_cost = c1.slider("Monthly Fixed Costs (Rent/Staff)", 30000, 100000, 60000, step=5000)
            var_cost_pct = c2.slider("Variable Cost % (COGS)", 10, 70, 30, step=5) / 100
            target_margin = c3.slider("Target Net Profit %", 5, 50, 20) / 100

        # 2. Dynamic Analysis
        col_fin1, col_fin2 = st.columns(2)
        
        with col_fin1:
            st.markdown("### 📉 Break-Even Calculator")
            # Calculation
            be_revenue = fixed_cost / (1 - var_cost_pct)
            current_avg_monthly_rev = df_filtered['Revenue_AED'].mean() * 30
            
            st.metric("Monthly Break-Even Revenue", f"AED {be_revenue:,.0f}")
            
            if current_avg_monthly_rev > be_revenue:
                st.success(f"✅ Safe! Current Monthly Avg (AED {current_avg_monthly_rev:,.0f}) is **{current_avg_monthly_rev/be_revenue:.1f}x** higher than Break-Even.")
            else:
                st.error(f"⚠️ Danger! Current Monthly Avg (AED {current_avg_monthly_rev:,.0f}) is below Break-Even.")

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = current_avg_monthly_rev,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Current vs Break-Even Revenue"},
                delta = {'reference': be_revenue},
                gauge = {
                    'axis': {'range': [0, max(be_revenue*1.5, current_avg_monthly_rev*1.2)]},
                    'bar': {'color': "#00CC96" if current_avg_monthly_rev > be_revenue else "#EF553B"},
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': be_revenue}
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_fin2:
            st.markdown("### 🎯 Path to Profit Target")
            # How much revenue needed to hit target margin?
            # Revenue - VarCost - Fixed = Revenue * TargetMargin
            # Revenue * (1 - VarCost% - TargetMargin) = Fixed
            # Revenue = Fixed / (1 - VarCost% - TargetMargin)
            
            denom = (1 - var_cost_pct - target_margin)
            if denom > 0:
                target_rev = fixed_cost / denom
                gap = target_rev - current_avg_monthly_rev
                
                st.write(f"To achieve a **{target_margin:.0%} Net Profit Margin**, you need:")
                st.metric("Target Monthly Revenue", f"AED {target_rev:,.0f}", delta=f"{gap:,.0f} gap")
                
                # Visualizing the gap
                df_gap = pd.DataFrame({
                    "Stage": ["Current Revenue", "Gap to Target", "Target Revenue"],
                    "Amount": [current_avg_monthly_rev, max(0, gap), target_rev]
                })
                fig_waterfall = go.Figure(go.Waterfall(
                    measure = ["relative", "relative", "total"],
                    x = ["Current", "Gap Needed", "Target"],
                    y = [current_avg_monthly_rev, max(0, gap), 0],
                    connector = {"line":{"color":"rgb(63, 63, 63)"}},
                ))
                fig_waterfall.update_layout(title="Revenue Gap Analysis")
                st.plotly_chart(fig_waterfall, use_container_width=True)
            else:
                st.warning("Impossible Target! Variable Costs + Target Margin exceed 100%.")
