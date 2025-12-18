import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# --- Page Config ---
st.set_page_config(page_title="Bunk Station Analytics", layout="wide")

# --- Title & Context ---
st.title("📊 Bunk Station: Strategic Analytics Dashboard")
st.markdown("""
> **Strategic Context:** Leveraging front-loaded fixed investments (Q1 2021). 
> This dashboard focuses on optimizing variable returns, analyzing conversion efficiency, and stress-testing financial resilience.
""")

# --- 1. Data Loading ---
@st.cache_data
def load_data():
    file_path = "Bunk_Station_Synthetic_Data.xlsx"
    
    if not os.path.exists(file_path):
        st.error(f"❌ File not found: {file_path}. Please ensure the Excel file is in the root directory.")
        return None
        
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Ensure date format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Feature Engineering for ML
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Is_Weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Calculate Conversion Rate if not present, handle division by zero
        if 'Orders' in df.columns and 'Footfall' in df.columns:
            df['Conversion_Rate'] = (df['Orders'] / df['Footfall'] * 100).fillna(0)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data automatically
df = load_data()

# --- Main Dashboard Logic ---
if df is not None:
    
    # Sidebar Filters
    st.sidebar.header("Filter Settings")
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Filter Data based on selection
    if len(date_range) == 2:
        mask = (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))
        df_filtered = df.loc[mask]
    else:
        df_filtered = df

    # Tabs for Organization
    tab1, tab2, tab3 = st.tabs(["📈 Descriptive Analytics", "🤖 AI/ML Insights", "💰 Financial Impact"])

    # ==========================================
    # TAB 1: DESCRIPTIVE ANALYTICS
    # ==========================================
    with tab1:
        st.subheader("Operational Overview")
        
        # High-level KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"AED {df_filtered['Revenue_AED'].sum():,.0f}")
        col2.metric("Total Footfall", f"{df_filtered['Footfall'].sum():,.0f}")
        avg_conv = df_filtered['Conversion_Rate'].mean()
        col3.metric("Avg Conversion Rate", f"{avg_conv:.2f}%")
        col4.metric("Avg Ticket Size", f"AED {df_filtered['Avg_Ticket_AED'].mean():.2f}")

        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### Revenue vs. Footfall Trend")
            fig_trend = px.line(df_filtered, x='Date', y=['Revenue_AED', 'Footfall'], 
                                title="Daily Trends (scaled)")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col_chart2:
            st.markdown("### Conversion Rate Efficiency")
            fig_conv = px.bar(df_filtered, x='Date', y='Conversion_Rate', 
                              color='Conversion_Rate', title="Daily Conversion Rate (%)",
                              color_continuous_scale='Blues')
            st.plotly_chart(fig_conv, use_container_width=True)

    # ==========================================
    # TAB 2: AI/ML INSIGHTS
    # ==========================================
    with tab2:
        st.subheader("Machine Learning: Revenue Drivers & Prediction")
        
        # ML Preprocessing
        features = ['Footfall', 'Avg_Ticket_AED', 'Day_of_Week', 'Is_Weekend']
        target = 'Revenue_AED'
        
        # Ensure we have enough data
        if len(df_filtered) > 10:
            X = df_filtered[features]
            y = df_filtered[target]
            
            # Train Model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            st.info(f"**Model Accuracy (R² Score):** {score:.2f} (Explains {score*100:.0f}% of revenue variance)")

            # Feature Importance
            col_ml1, col_ml2 = st.columns(2)
            
            with col_ml1:
                st.markdown("### Key Revenue Drivers")
                importances = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h',
                                 title="What drives your Revenue most?")
                st.plotly_chart(fig_imp, use_container_width=True)
                
            with col_ml2:
                st.markdown("### 🔮 AI 'What-If' Simulator")
                st.write("Adjust sliders to predict Daily Revenue:")
                
                sim_footfall = st.slider("Simulated Footfall", int(df['Footfall'].min()), int(df['Footfall'].max() * 1.5), int(df['Footfall'].mean()))
                sim_ticket = st.slider("Simulated Avg Ticket (AED)", float(df['Avg_Ticket_AED'].min()), float(df['Avg_Ticket_AED'].max() * 1.2), float(df['Avg_Ticket_AED'].mean()))
                sim_weekend = st.selectbox("Is it a Weekend?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                
                # Prediction
                # We use mean Day_of_Week (4 = Friday approx) for generic prediction
                sim_input = np.array([[sim_footfall, sim_ticket, 4, sim_weekend]]) 
                predicted_rev = model.predict(sim_input)[0]
                
                st.metric(label="Predicted Daily Revenue", value=f"AED {predicted_rev:,.2f}")
                
                # Comparison logic
                current_avg_rev = df_filtered['Revenue_AED'].mean()
                diff = predicted_rev - current_avg_rev
                color = "green" if diff > 0 else "red"
                st.markdown(f"This is <span style='color:{color}'>**AED {diff:,.0f}**</span> vs your selected period average.", unsafe_allow_html=True)
        else:
            st.warning("Not enough data points selected for Machine Learning. Please expand your date range.")

    # ==========================================
    # TAB 3: FINANCIAL IMPACT
    # ==========================================
    with tab3:
        st.subheader("Financial Impact & Operating Leverage")
        st.markdown("Since you **front-loaded fixed investments in 2021**, your Fixed Costs should be stable. This tool calculates your 'Safety Margin'.")

        # Inputs for Financials
        col_fin1, col_fin2 = st.columns([1, 2])
        
        with col_fin1:
            st.markdown("#### Cost Inputs (Monthly)")
            fixed_cost = st.number_input("Est. Monthly Fixed Cost (Rent/Salaries)", value=50000, step=1000)
            cogs_pct = st.slider("Variable Cost % (COGS)", 0, 100, 30) / 100
        
        with col_fin2:
            # Aggregating data to Monthly for this view
            df_monthly = df_filtered.copy()
            df_monthly['YearMonth'] = df_monthly['Date'].dt.to_period('M')
            monthly_data = df_monthly.groupby('YearMonth')['Revenue_AED'].sum().reset_index()
            monthly_data['YearMonth'] = monthly_data['YearMonth'].astype(str)
            
            # Calculating Financials
            monthly_data['Variable_Cost'] = monthly_data['Revenue_AED'] * cogs_pct
            monthly_data['Gross_Profit'] = monthly_data['Revenue_AED'] - monthly_data['Variable_Cost']
            monthly_data['Net_Profit'] = monthly_data['Gross_Profit'] - fixed_cost
            
            # Charting Profitability
            fig_fin = go.Figure()
            fig_fin.add_trace(go.Bar(x=monthly_data['YearMonth'], y=monthly_data['Net_Profit'], name='Net Profit/Loss', marker_color=np.where(monthly_data['Net_Profit']<0, 'red', 'green')))
            fig_fin.add_trace(go.Scatter(x=monthly_data['YearMonth'], y=[0]*len(monthly_data), mode='lines', name='Break-even Line', line=dict(color='black', dash='dash')))
            
            fig_fin.update_layout(title="Monthly Net Profit Simulation (Post-Fixed Cost)", barmode='relative')
            st.plotly_chart(fig_fin, use_container_width=True)

        # Leverage Analysis
        st.markdown("---")
        st.markdown("### ⚖️ Operating Leverage Analysis")
        
        # Calculate Degree of Operating Leverage (DOL) for the most recent complete month
        if len(monthly_data) > 0:
            last_month = monthly_data.iloc[-1]
            contribution_margin = last_month['Gross_Profit']
            operating_income = last_month['Net_Profit']
            
            if operating_income > 0:
                dol = contribution_margin / operating_income
                st.write(f"**Degree of Operating Leverage (DOL): {dol:.2f}**")
                st.info(f"For every **1% increase in Revenue**, your Operating Profit increases by **{dol:.2f}%**. This confirms the benefit of your high fixed-cost, low variable-cost structure.")
            else:
                st.warning("Currently operating at a loss in the most recent month; DOL calculation requires positive operating income.")
