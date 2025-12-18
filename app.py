import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# --- Page Config ---
st.set_page_config(page_title="Bunk Station Analytics Pro", layout="wide")

# --- Title & Context ---
st.title("📊 Bunk Station: Strategic Analytics Dashboard (Pro)")
st.markdown("""
> **Strategic Context:** Leveraging front-loaded fixed investments (Q1 2021). 
> This enhanced dashboard focuses on seasonality, driver analysis, and financial scenario planning.
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

    tab1, tab2, tab3 = st.tabs(["📈 Descriptive Analytics", "🤖 AI/ML Insights", "💰 Financial Impact"])

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

        # Row 1: Seasonality & Trends
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("### 🗓️ Seasonal Heatmap (Day vs Month)")
            # Pivot for Heatmap
            pivot_table = df_filtered.pivot_table(index='Day_of_Week', columns='Month', values='Revenue_AED', aggfunc='mean')
            # Sort days/months correctly
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            pivot_table = pivot_table.reindex(days_order)
            pivot_table = pivot_table.reindex(columns=[m for m in months_order if m in pivot_table.columns])
            
            fig_heat = px.imshow(pivot_table, text_auto=".0f", color_continuous_scale="RdBu_r", aspect="auto",
                                 title="Avg Revenue: Spotting the 'Golden' Days")
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_d2:
            st.markdown("### 📈 Revenue Trend (with 7-Day Moving Avg)")
            # Rolling Average
            df_trend = df_filtered.sort_values('Date').copy()
            df_trend['7_Day_MA'] = df_trend['Revenue_AED'].rolling(window=7).mean()
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['Revenue_AED'], name='Daily Revenue', line=dict(color='lightgray', width=1)))
            fig_trend.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['7_Day_MA'], name='7-Day Trend', line=dict(color='blue', width=3)))
            fig_trend.update_layout(title="Daily Volatility vs. Underlying Trend")
            st.plotly_chart(fig_trend, use_container_width=True)

        # Row 2: Correlations
        st.markdown("### 🔗 Correlation Matrix")
        st.caption("Do high Footfall days actually lead to lower Ticket sizes? (Negative correlation)")
        corr_cols = ['Footfall', 'Revenue_AED', 'Avg_Ticket_AED', 'Conversion_Rate', 'Orders']
        corr_matrix = df_filtered[corr_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu", zmid=0,
                             title="Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)

    # ==========================================
    # TAB 2: AI/ML INSIGHTS
    # ==========================================
    with tab2:
        st.subheader("Deep Dive Intelligence")
        
        col_ai1, col_ai2 = st.columns(2)

        # ML 1: Feature Importance
        with col_ai1:
            st.markdown("### 🧠 What drives Revenue most?")
            # Train model just for importance
            features = ['Footfall', 'Avg_Ticket_AED', 'Conversion_Rate', 'Day_Index', 'Is_Weekend']
            X = df_filtered[features].fillna(0)
            y = df_filtered['Revenue_AED']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
            
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Feature Importance Analysis")
            st.plotly_chart(fig_imp, use_container_width=True)
            st.info("💡 **Insight:** Focus your operational efforts on the top bar. If 'Footfall' is #1, marketing is key. If 'Avg Ticket' is #1, upselling is key.")

        # ML 2: Clustering
        with col_ai2:
            st.markdown("### 🧩 Customer Traffic Segmentation")
            # K-Means
            X_cluster = df_filtered[['Footfall', 'Revenue_AED']].dropna()
            if len(X_cluster) > 5:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_cluster)
                kmeans = KMeans(n_clusters=3, random_state=42)
                df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)
                
                fig_clus = px.scatter(df_filtered, x='Footfall', y='Revenue_AED', color=df_filtered['Cluster'].astype(str),
                                      title="3 Types of Sales Days", labels={'Cluster': 'Segment'})
                st.plotly_chart(fig_clus, use_container_width=True)
                
                # Cluster Profiling Table
                cluster_profile = df_filtered.groupby('Cluster')[['Footfall', 'Revenue_AED', 'Avg_Ticket_AED']].mean().reset_index()
                cluster_profile['Description'] = ["Low Traffic / Low Rev", "Med Traffic / Med Rev", "High Traffic / High Rev"] # Simplified labeling logic
                st.dataframe(cluster_profile.style.format("{:.0f}", subset=['Footfall', 'Revenue_AED']))
            else:
                st.warning("Not enough data for clustering")

    # ==========================================
    # TAB 3: FINANCIAL IMPACT
    # ==========================================
    with tab3:
        st.subheader("Financial Stress Testing & Planning")
        
        # Financial Inputs
        with st.expander("⚙️ Edit Financial Assumptions", expanded=True):
            col_f1, col_f2, col_f3 = st.columns(3)
            fixed_cost = col_f1.number_input("Monthly Fixed Cost (AED)", value=60000)
            cogs_pct = col_f2.slider("COGS % (Variable Cost)", 10, 80, 30) / 100
            avg_rev = df_filtered['Revenue_AED'].mean() * 30 # Approx monthly
            
        col_fin1, col_fin2 = st.columns(2)
        
        # 1. Break-Even Visual
        with col_fin1:
            st.markdown("### 📉 Break-Even Analysis")
            # Create a range of revenue scenarios
            rev_range = np.linspace(0, avg_rev * 2, 50)
            total_costs = fixed_cost + (rev_range * cogs_pct)
            
            fig_be = go.Figure()
            fig_be.add_trace(go.Scatter(x=rev_range, y=rev_range, name='Revenue', line=dict(color='green')))
            fig_be.add_trace(go.Scatter(x=rev_range, y=total_costs, name='Total Cost', line=dict(color='red', dash='dash')))
            
            # Find crossing point
            be_point = fixed_cost / (1 - cogs_pct)
            
            fig_be.add_vline(x=be_point, line_dash="dot", annotation_text=f"Break-even: {be_point:,.0f}")
            fig_be.update_layout(title="Revenue vs. Cost Curves", xaxis_title="Revenue", yaxis_title="Amount")
            st.plotly_chart(fig_be, use_container_width=True)

        # 2. Scenario Table
        with col_fin2:
            st.markdown("### 🔮 Scenario Planning")
            
            scenarios = {
                "Scenario": ["Worst Case (-20%)", "Base Case (Current)", "Best Case (+20%)"],
                "Monthly Revenue": [avg_rev * 0.8, avg_rev, avg_rev * 1.2],
            }
            df_scen = pd.DataFrame(scenarios)
            df_scen['Variable Cost'] = df_scen['Monthly Revenue'] * cogs_pct
            df_scen['Fixed Cost'] = fixed_cost
            df_scen['Net Profit'] = df_scen['Monthly Revenue'] - df_scen['Variable Cost'] - df_scen['Fixed Cost']
            df_scen['Margin %'] = (df_scen['Net Profit'] / df_scen['Monthly Revenue'] * 100).fillna(0)
            
            # Formatting
            st.table(df_scen.style.format({
                "Monthly Revenue": "AED {:,.0f}", 
                "Variable Cost": "AED {:,.0f}",
                "Fixed Cost": "AED {:,.0f}",
                "Net Profit": "AED {:,.0f}",
                "Margin %": "{:.1f}%"
            }))
            
        # 3. Sensitivity Heatmap (Re-added from previous)
        st.markdown("### 🌡️ Profit Sensitivity Matrix")
        st.caption("Net Profit at different Footfall & Ticket Size combinations")
        
        base_footfall = df_filtered['Footfall'].mean() * 30
        base_ticket = df_filtered['Avg_Ticket_AED'].mean()
        
        f_range = np.linspace(base_footfall * 0.7, base_footfall * 1.3, 10)
        t_range = np.linspace(base_ticket * 0.8, base_ticket * 1.2, 10)
        
        z_vals = []
        for t in t_range:
            row_vals = []
            for f in f_range:
                rev = f * t
                profit = rev - (rev * cogs_pct) - fixed_cost
                row_vals.append(profit)
            z_vals.append(row_vals)
            
        fig_sens = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=[f"{x:,.0f}" for x in f_range],
            y=[f"{y:.1f}" for y in t_range],
            colorscale='RdBu', zmid=0,
            colorbar=dict(title='Net Profit')
        ))
        fig_sens.update_layout(xaxis_title="Monthly Footfall", yaxis_title="Avg Ticket Size")
        st.plotly_chart(fig_sens, use_container_width=True)
