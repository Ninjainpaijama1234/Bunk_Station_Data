import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import os

# --- Page Config ---
st.set_page_config(page_title="Bunk Station Analytics Pro", layout="wide")

# --- Title & Context ---
st.title("📊 Bunk Station: Strategic Analytics Dashboard (Pro)")
st.markdown("""
> **Strategic Context:** Leveraging front-loaded fixed investments (Q1 2021). 
> This advanced dashboard includes **Interactive Scenario Planning** to model your 2026 ROI.
""")

# --- 1. Data Loading ---
@st.cache_data
def load_data():
    file_path = "Bunk_Station_Daily_Sales_Full_Year.csv"
    
    if not os.path.exists(file_path):
        st.error(f"❌ File not found: {file_path}. Please ensure the CSV file is in the root directory.")
        return None
        
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Feature Engineering
        df['Day_of_Week'] = df['Date'].dt.day_name()
        df['Month'] = df['Date'].dt.month_name()
        df['Month_Num'] = df['Date'].dt.month
        df['Day_Index'] = df['Date'].dt.dayofweek
        # Standardized column name (Lowercase 'o')
        df['Day_of_Year'] = df['Date'].dt.dayofyear 
        
        df['Is_Weekend'] = df['Day_Index'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Trend Feature (Days since start)
        start_date = df['Date'].min()
        df['Days_Since_Start'] = (df['Date'] - start_date).dt.days
        
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
    # Sidebar
    st.sidebar.header("Filter Settings")
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    if len(date_range) == 2:
        mask = (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))
        df_filtered = df.loc[mask]
    else:
        df_filtered = df

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Descriptive", "🤖 AI/ML Insights", "💰 Financial Impact", "🔮 Future Forecasting"])

    # ==========================================
    # TAB 1: DESCRIPTIVE ANALYTICS
    # ==========================================
    with tab1:
        st.subheader("Operational Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue", f"AED {df_filtered['Revenue_AED'].sum():,.0f}")
        c2.metric("Total Footfall", f"{df_filtered['Footfall'].sum():,.0f}")
        c3.metric("Avg Conversion", f"{df_filtered['Conversion_Rate'].mean():.2f}%")
        c4.metric("Avg Ticket", f"AED {df_filtered['Avg_Ticket_AED'].mean():.2f}")

        st.markdown("---")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("### 🗓️ Seasonal Heatmap")
            pivot_table = df_filtered.pivot_table(index='Day_of_Week', columns='Month', values='Revenue_AED', aggfunc='mean')
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            pivot_table = pivot_table.reindex(days_order)
            pivot_table = pivot_table.reindex(columns=[m for m in months_order if m in pivot_table.columns])
            
            fig_heat = px.imshow(pivot_table, text_auto=".0f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_d2:
            st.markdown("### 📈 Revenue Trend")
            df_trend = df_filtered.sort_values('Date').copy()
            df_trend['7_Day_MA'] = df_trend['Revenue_AED'].rolling(window=7).mean()
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['Revenue_AED'], name='Daily Revenue', line=dict(color='lightgray', width=1)))
            fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['7_Day_MA'], name='7-Day Trend', line=dict(color='blue', width=3)))
            st.plotly_chart(fig_trend, use_container_width=True)

    # ==========================================
    # TAB 2: AI/ML INSIGHTS
    # ==========================================
    with tab2:
        st.subheader("Deep Dive Intelligence")
        col_ai1, col_ai2 = st.columns(2)

        with col_ai1:
            st.markdown("### 🧠 Key Revenue Drivers")
            features = ['Footfall', 'Avg_Ticket_AED', 'Conversion_Rate', 'Day_Index', 'Is_Weekend']
            X = df_filtered[features].fillna(0)
            y = df_filtered['Revenue_AED']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_ai2:
            st.markdown("### 🏷️ Price Elasticity Proxy")
            try:
                # Basic scatter without OLS to avoid statsmodels dependency error if not installed
                fig_elas = px.scatter(df_filtered, x="Avg_Ticket_AED", y="Conversion_Rate", 
                                      color="Is_Weekend", title="Ticket Price vs. Conversion Rate")
                st.plotly_chart(fig_elas, use_container_width=True)
            except:
                st.info("Scatter plot requires valid data.")

    # ==========================================
    # TAB 3: FINANCIAL IMPACT
    # ==========================================
    with tab3:
        st.subheader("Financial Stress Testing")
        with st.expander("⚙️ Financial Assumptions (Monthly)", expanded=True):
            col_f1, col_f2 = st.columns(2)
            fixed_cost = col_f1.number_input("Monthly Fixed Cost (AED)", value=60000)
            cogs_pct = col_f2.slider("COGS % (Variable Cost)", 10, 80, 30) / 100
        
        col_fin1, col_fin2 = st.columns(2)
        with col_fin1:
            st.markdown("### ⚙️ Operating Leverage")
            total_rev = df_filtered['Revenue_AED'].sum()
            contribution_margin = total_rev - (total_rev * cogs_pct)
            num_days = (df_filtered['Date'].max() - df_filtered['Date'].min()).days + 1
            est_months = max(1, num_days / 30)
            operating_income = contribution_margin - (fixed_cost * est_months)
            
            if operating_income > 0:
                dol = contribution_margin / operating_income
                st.metric("Degree of Operating Leverage", f"{dol:.2f}x")
                st.success(f"For every 1% sales increase, Profit grows by {dol:.2f}%")
            else:
                st.error("Operating below Break-Even.")

    # ==========================================
    # TAB 4: STRATEGIC FORECASTING (REBUILT)
    # ==========================================
    with tab4:
        st.subheader("🔮 2026 Strategic Scenario Planner")
        st.markdown("Control the **Business Drivers** below to simulate your future performance.")
        
        # --- 1. SCENARIO CONTROLS (Top Section) ---
        with st.container():
            st.markdown("#### 🛠️ Simulation Variables")
            col_var1, col_var2, col_var3 = st.columns(3)
            
            with col_var1:
                footfall_growth = st.slider("📢 Marketing Impact (Footfall Growth)", -20, 50, 0, format="%+d%%") / 100
                st.caption("Simulates increasing ad spend or brand awareness.")
                
            with col_var2:
                price_change = st.slider("🏷️ Pricing Strategy (Ticket Change)", -10, 20, 0, format="%+d%%") / 100
                st.caption("Simulates price hikes or discounting strategies.")
                
            with col_var3:
                weekend_boost = st.slider("🎉 Weekend Promo Boost", 0, 30, 0, format="%+d%%") / 100
                st.caption("Extra footfall specifically on Sat/Sun.")

        st.markdown("---")

        # --- 2. MODELING ENGINE (Hidden) ---
        # We train TWO models: One for Footfall, One for Ticket Size.
        # This allows us to adjust them independently.
        
        # Prepare Data
        train_df = df[['Day_Index', 'Month_Num', 'Day_of_Year', 'Is_Weekend', 'Days_Since_Start', 'Footfall', 'Avg_Ticket_AED']].dropna()
        X = train_df[['Day_Index', 'Month_Num', 'Day_of_Year', 'Is_Weekend', 'Days_Since_Start']]
        y_footfall = train_df['Footfall']
        y_ticket = train_df['Avg_Ticket_AED']
        
        # Train Models
        model_footfall = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_footfall)
        model_ticket = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_ticket)
        
        # Create Future Dates (Next 365 Days)
        last_date = df['Date'].max()
        future_dates = [last_date + timedelta(days=x) for x in range(1, 366)]
        future_df = pd.DataFrame({'Date': future_dates})
        
        # Future Features
        future_df['Day_Index'] = future_df['Date'].dt.dayofweek
        future_df['Month_Num'] = future_df['Date'].dt.month
        future_df['Day_of_Year'] = future_df['Date'].dt.dayofyear
        future_df['Is_Weekend'] = future_df['Day_Index'].apply(lambda x: 1 if x >= 5 else 0)
        start_date = df['Date'].min()
        future_df['Days_Since_Start'] = (future_df['Date'] - start_date).dt.days
        
        # Predict Base Case
        X_future = future_df[['Day_Index', 'Month_Num', 'Day_of_Year', 'Is_Weekend', 'Days_Since_Start']]
        future_df['Base_Footfall'] = model_footfall.predict(X_future)
        future_df['Base_Ticket'] = model_ticket.predict(X_future)
        
        # --- 3. APPLY SCENARIOS ---
        # Apply Footfall Growth + Weekend Boost
        future_df['Adj_Footfall'] = future_df['Base_Footfall'] * (1 + footfall_growth)
        future_df['Adj_Footfall'] = np.where(future_df['Is_Weekend'] == 1, 
                                             future_df['Adj_Footfall'] * (1 + weekend_boost), 
                                             future_df['Adj_Footfall'])
        
        # Apply Price Strategy
        future_df['Adj_Ticket'] = future_df['Base_Ticket'] * (1 + price_change)
        
        # Calculate Final Revenue
        future_df['Forecast_Revenue'] = future_df['Adj_Footfall'] * future_df['Adj_Ticket']

        # --- 4. RESULTS & DOWNLOAD (Middle Section) ---
        total_forecast = future_df['Forecast_Revenue'].sum()
        hist_revenue = df['Revenue_AED'].sum() # Compare to historical period sum (approx)
        
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            st.metric("💰 Total Projected Revenue (Next 365 Days)", f"AED {total_forecast:,.0f}",
                      delta=f"{(total_forecast - hist_revenue)/hist_revenue:.1%} vs Previous Year")
        
        with col_res2:
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            csv = future_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name="Bunk_Station_2026_Forecast.csv",
                mime="text/csv",
                use_container_width=True
            )

        # --- 5. ALIGNED CHARTS (Grid Layout) ---
        
        # Row A: Main Revenue Timeline
        st.markdown("### 📈 Revenue Trajectory")
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(x=df['Date'], y=df['Revenue_AED'], name='Historical', line=dict(color='gray', width=1)))
        fig_main.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Forecast_Revenue'], name='Forecast (Scenario)', line=dict(color='#00CC96', width=2)))
        fig_main.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Row B: The Drivers (Side by Side)
        st.markdown("### 🚦 Driver Breakdown")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("**Expected Footfall Pattern**")
            fig_foot = px.line(future_df, x='Date', y='Adj_Footfall', 
                               title="Projected Daily Footfall", color_discrete_sequence=['#636EFA'])
            fig_foot.update_layout(height=300)
            st.plotly_chart(fig_foot, use_container_width=True)
            
        with col_chart2:
            st.markdown("**Expected Ticket Size Strategy**")
            fig_tick = px.line(future_df, x='Date', y='Adj_Ticket', 
                               title="Projected Avg Ticket Size", color_discrete_sequence=['#EF553B'])
            fig_tick.update_layout(height=300)
            st.plotly_chart(fig_tick, use_container_width=True)
