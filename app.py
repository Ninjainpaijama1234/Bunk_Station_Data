import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import os

# --- Page Config ---
st.set_page_config(page_title="Bunk Station Analytics Pro", layout="wide")

# --- Title & Context ---
st.title("📊 Bunk Station: Strategic Analytics Dashboard (Pro)")
st.markdown("""
> **Strategic Context:** Leveraging front-loaded fixed investments (Q1 2021). 
> This advanced dashboard now includes **AI-powered Annual Forecasting** to project ROI for the coming year.
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
        
        # --- FIXED TYPO HERE (Capital 'O') ---
        df['Day_Of_Year'] = df['Date'].dt.dayofyear 
        
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
        
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue", f"AED {df_filtered['Revenue_AED'].sum():,.0f}")
        c2.metric("Total Footfall", f"{df_filtered['Footfall'].sum():,.0f}")
        c3.metric("Avg Conversion", f"{df_filtered['Conversion_Rate'].mean():.2f}%")
        c4.metric("Avg Ticket", f"AED {df_filtered['Avg_Ticket_AED'].mean():.2f}")

        st.markdown("---")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("### 🗓️ Seasonal Heatmap (Day vs Month)")
            pivot_table = df_filtered.pivot_table(index='Day_of_Week', columns='Month', values='Revenue_AED', aggfunc='mean')
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            pivot_table = pivot_table.reindex(days_order)
            pivot_table = pivot_table.reindex(columns=[m for m in months_order if m in pivot_table.columns])
            
            fig_heat = px.imshow(pivot_table, text_auto=".0f", color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_d2:
            st.markdown("### 📈 Revenue Trend (7-Day MA)")
            df_trend = df_filtered.sort_values('Date').copy()
            df_trend['7_Day_MA'] = df_trend['Revenue_AED'].rolling(window=7).mean()
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['Revenue_AED'], name='Daily Revenue', line=dict(color='lightgray', width=1)))
            fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['7_Day_MA'], name='7-Day Trend', line=dict(color='blue', width=3)))
            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("### 🔗 Correlation Matrix")
        corr_cols = ['Footfall', 'Revenue_AED', 'Avg_Ticket_AED', 'Conversion_Rate', 'Orders']
        corr_matrix = df_filtered[corr_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ==========================================
    # TAB 2: AI/ML INSIGHTS
    # ==========================================
    with tab2:
        st.subheader("Deep Dive Intelligence")
        
        col_ai1, col_ai2 = st.columns(2)

        # ML 1: Feature Importance
        with col_ai1:
            st.markdown("### 🧠 Key Revenue Drivers")
            features = ['Footfall', 'Avg_Ticket_AED', 'Conversion_Rate', 'Day_Index', 'Is_Weekend']
            X = df_filtered[features].fillna(0)
            y = df_filtered['Revenue_AED']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Feature Importance Analysis")
            st.plotly_chart(fig_imp, use_container_width=True)

        # NEW: Price Elasticity Proxy
        with col_ai2:
            st.markdown("### 🏷️ Price Elasticity Proxy")
            st.caption("Does a higher Avg Ticket size hurt Conversion Rate?")
            fig_elas = px.scatter(df_filtered, x="Avg_Ticket_AED", y="Conversion_Rate", 
                                  trendline="ols", color="Is_Weekend",
                                  title="Ticket Price vs. Conversion Rate")
            st.plotly_chart(fig_elas, use_container_width=True)

        st.markdown("---")
        
        # NEW: 3D Visualization
        st.markdown("### 🧊 The 'Revenue Cube'")
        st.caption("Visualize the sweet spot between Footfall (X), Ticket Size (Y), and Revenue (Z)")
        fig_3d = px.scatter_3d(df_filtered, x='Footfall', y='Avg_Ticket_AED', z='Revenue_AED',
                               color='Revenue_AED', color_continuous_scale='Viridis', opacity=0.7)
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
        st.plotly_chart(fig_3d, use_container_width=True)

    # ==========================================
    # TAB 3: FINANCIAL IMPACT
    # ==========================================
    with tab3:
        st.subheader("Financial Stress Testing & Planning")
        
        # Inputs
        with st.expander("⚙️ Financial Assumptions (Monthly)", expanded=True):
            col_f1, col_f2 = st.columns(2)
            fixed_cost = col_f1.number_input("Monthly Fixed Cost (AED)", value=60000)
            cogs_pct = col_f2.slider("COGS % (Variable Cost)", 10, 80, 30) / 100
        
        col_fin1, col_fin2 = st.columns(2)
        
        # 1. Operating Leverage
        with col_fin1:
            st.markdown("### ⚙️ Operating Leverage")
            
            # Simple DOL Calc (Contribution Margin / Operating Income)
            total_rev = df_filtered['Revenue_AED'].sum()
            total_var_cost = total_rev * cogs_pct
            contribution_margin = total_rev - total_var_cost
            # Estimate months in selection
            num_days = (df_filtered['Date'].max() - df_filtered['Date'].min()).days + 1
            est_months = max(1, num_days / 30)
            total_fixed = fixed_cost * est_months
            operating_income = contribution_margin - total_fixed
            
            if operating_income > 0:
                dol = contribution_margin / operating_income
                st.metric("Degree of Operating Leverage", f"{dol:.2f}x", 
                         help="For every 1% increase in Sales, Profit increases by this multiplier.")
                st.success(f"🚀 **High Leverage:** A 10% sales boost adds **{dol*10:.1f}%** to your profits!")
            else:
                st.metric("Degree of Operating Leverage", "N/A (Loss Making)")
                st.error("Currently operating below Break-Even for this period.")

        # 2. Profit Scenario
        with col_fin2:
            st.markdown("### 🔮 Next Month Profit Scenario")
            avg_mth_rev = df_filtered['Revenue_AED'].mean() * 30
            
            scenarios = [-0.2, -0.1, 0, 0.1, 0.2]
            res = []
            for s in scenarios:
                r = avg_mth_rev * (1 + s)
                p = r - (r * cogs_pct) - fixed_cost
                res.append({"Scenario": f"{s:+.0%}", "Revenue": r, "Net Profit": p})
            
            df_res = pd.DataFrame(res)
            fig_scen = px.bar(df_res, x='Scenario', y='Net Profit', color='Net Profit',
                              color_continuous_scale='RdBu', title="Projected Monthly Profit")
            st.plotly_chart(fig_scen, use_container_width=True)

    # ==========================================
    # TAB 4: REVENUE FORECASTING (NEW)
    # ==========================================
    with tab4:
        st.subheader("🔮 2025/2026 Revenue Forecast")
        st.markdown("Predicting daily revenue for the next 365 days based on historical patterns.")
        
        col_fore1, col_fore2 = st.columns([1, 3])
        
        with col_fore1:
            st.markdown("#### Model Settings")
            model_type = st.radio("Choose Algorithm", ["Random Forest (Seasonality)", "Linear Regression (Trend)"])
            growth_adjustment = st.slider("Manual Growth Adjust (%)", -20, 50, 0, help="Manually adjust the AI forecast up or down.") / 100
        
        with col_fore2:
            # 1. Prepare Training Data
            train_df = df[['Day_Index', 'Month_Num', 'Day_Of_Year', 'Is_Weekend', 'Days_Since_Start', 'Revenue_AED']].dropna()
            X_train = train_df.drop(columns=['Revenue_AED'])
            y_train = train_df['Revenue_AED']
            
            # 2. Select Model
            if "Random Forest" in model_type:
                model = RandomForestRegressor(n_estimators=200, random_state=42)
            else:
                model = LinearRegression()
                
            model.fit(X_train, y_train)
            
            # 3. Create Future Dataframe (Next 365 days)
            last_date = df['Date'].max()
            future_dates = [last_date + timedelta(days=x) for x in range(1, 366)]
            future_df = pd.DataFrame({'Date': future_dates})
            
            # Feature Engineering for Future
            future_df['Day_Index'] = future_df['Date'].dt.dayofweek
            future_df['Month_Num'] = future_df['Date'].dt.month
            
            # --- FIXED TYPO HERE (Capital 'O') ---
            future_df['Day_Of_Year'] = future_df['Date'].dt.dayofyear
            
            future_df['Is_Weekend'] = future_df['Day_Index'].apply(lambda x: 1 if x >= 5 else 0)
            start_date = df['Date'].min()
            future_df['Days_Since_Start'] = (future_df['Date'] - start_date).dt.days
            
            # 4. Predict
            X_future = future_df[['Day_Index', 'Month_Num', 'Day_Of_Year', 'Is_Weekend', 'Days_Since_Start']]
            future_df['Predicted_Revenue'] = model.predict(X_future)
            
            # Apply Manual Adjustment
            future_df['Predicted_Revenue'] = future_df['Predicted_Revenue'] * (1 + growth_adjustment)
            
            # 5. Visuals
            st.metric("Projected Annual Revenue (Next 365 Days)", f"AED {future_df['Predicted_Revenue'].sum():,.0f}")
            
            # Combined Chart
            fig_forecast = go.Figure()
            # Historical
            fig_forecast.add_trace(go.Scatter(x=df['Date'], y=df['Revenue_AED'], name='Historical Data', line=dict(color='gray', width=1)))
            # Forecast
            fig_forecast.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Revenue'], name='AI Forecast', line=dict(color='blue', width=2)))
            
            fig_forecast.update_layout(title="Historical vs. Forecasted Revenue", xaxis_title="Date", yaxis_title="Revenue (AED)")
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # 6. Forecast Breakdown
            future_df['Month'] = future_df['Date'].dt.month_name()
            monthly_forecast = future_df.groupby('Month')['Predicted_Revenue'].sum().reset_index()
            # Sort by month order
            monthly_forecast['Month_Num'] = pd.to_datetime(monthly_forecast['Month'], format='%B').dt.month
            monthly_forecast = monthly_forecast.sort_values('Month_Num')
            
            st.markdown("#### 📅 Monthly Breakdown")
            fig_mth_fore = px.bar(monthly_forecast, x='Month', y='Predicted_Revenue', title="Projected Revenue by Month", text_auto='.2s')
            st.plotly_chart(fig_mth_fore, use_container_width=True)
