import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# --- Page Config ---
st.set_page_config(page_title="Bunk Station Analytics Pro", layout="wide")

# --- Title & Context ---
st.title("📊 Bunk Station: Strategic Analytics Dashboard (Pro)")
st.markdown("""
> **Strategic Context:** Leveraging front-loaded fixed investments (Q1 2021). 
> This advanced dashboard includes anomaly detection, day-segmentation clustering, and financial sensitivity matrices.
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
        
        # Feature Engineering
        df['Day_of_Week'] = df['Date'].dt.day_name()
        df['Day_Index'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month_name()
        df['Week_Year'] = df['Date'].dt.strftime('%Y-%U')
        df['Is_Weekend'] = df['Day_Index'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Calculate Conversion Rate
        if 'Orders' in df.columns and 'Footfall' in df.columns:
            df['Conversion_Rate'] = (df['Orders'] / df['Footfall'] * 100).fillna(0)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# --- Main Dashboard Logic ---
if df is not None:
    
    # Sidebar Filters
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
    tab1, tab2, tab3 = st.tabs(["📈 Descriptive Analytics", "🤖 AI/ML Deep Dive", "💰 Financial Simulation"])

    # ==========================================
    # TAB 1: DESCRIPTIVE ANALYTICS
    # ==========================================
    with tab1:
        st.subheader("Operational Overview")
        
        # Metric Row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Revenue", f"AED {df_filtered['Revenue_AED'].sum():,.0f}")
        c2.metric("Total Footfall", f"{df_filtered['Footfall'].sum():,.0f}")
        c3.metric("Avg Conversion", f"{df_filtered['Conversion_Rate'].mean():.2f}%")
        c4.metric("Avg Ticket", f"AED {df_filtered['Avg_Ticket_AED'].mean():.2f}")
        val_per_visitor = df_filtered['Revenue_AED'].sum() / df_filtered['Footfall'].sum() if df_filtered['Footfall'].sum() > 0 else 0
        c5.metric("Rev Per Visitor", f"AED {val_per_visitor:.2f}")

        st.markdown("---")

        # Row 1: Trends & Distribution
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("### 📅 Weekly Performance Heatmap")
            # Aggregating by Day of Week
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_stats = df_filtered.groupby('Day_of_Week')['Revenue_AED'].mean().reindex(day_order).reset_index()
            
            fig_heat = px.bar(daily_stats, x='Day_of_Week', y='Revenue_AED', 
                              color='Revenue_AED', title="Average Revenue by Day of Week",
                              color_continuous_scale='Viridis')
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_d2:
            st.markdown("### 🎟️ Ticket Size Distribution")
            fig_hist = px.histogram(df_filtered, x="Avg_Ticket_AED", nbins=20, 
                                    title="Are customers spending Little or Lot?",
                                    color_discrete_sequence=['#636EFA'])
            fig_hist.add_vline(x=df_filtered['Avg_Ticket_AED'].mean(), line_dash="dash", annotation_text="Avg")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Row 2: Conversion Funnel
        st.markdown("### 📉 Sales Velocity")
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Footfall'], name='Footfall', line=dict(color='gray', width=1)))
        fig_dual.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Revenue_AED'], name='Revenue', yaxis='y2', line=dict(color='green', width=2)))
        
        fig_dual.update_layout(
            title="Footfall vs Revenue (Correlation Check)",
            yaxis=dict(title="Footfall"),
            yaxis2=dict(title="Revenue AED", overlaying='y', side='right')
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    # ==========================================
    # TAB 2: AI/ML DEEP DIVE
    # ==========================================
    with tab2:
        st.subheader("Advanced Machine Learning Insights")
        
        col_ai1, col_ai2 = st.columns(2)

        # --- ML 1: CLUSTERING (Segmentation) ---
        with col_ai1:
            st.markdown("### 🧩 Day Segmentation (Clustering)")
            st.caption("We use K-Means to group days into 'Performance Profiles'.")
            
            # Prepare data
            X_cluster = df_filtered[['Footfall', 'Conversion_Rate', 'Avg_Ticket_AED']].dropna()
            if len(X_cluster) > 10:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_cluster)
                
                kmeans = KMeans(n_clusters=3, random_state=42)
                df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)
                
                # Rename clusters based on logic (simplified)
                cluster_summary = df_filtered.groupby('Cluster')[['Footfall', 'Conversion_Rate', 'Revenue_AED']].mean()
                st.dataframe(cluster_summary.style.highlight_max(axis=0, color='lightgreen'))
                
                fig_cluster = px.scatter(df_filtered, x='Footfall', y='Conversion_Rate', 
                                         color=df_filtered['Cluster'].astype(str), size='Revenue_AED',
                                         title="Clusters: High Traffic vs. High Efficiency",
                                         hover_data=['Date'])
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.warning("Not enough data for clustering.")

        # --- ML 2: ANOMALY DETECTION ---
        with col_ai2:
            st.markdown("### 🚨 Anomaly Detection")
            st.caption("AI detects days where sales were unexpectedly High or Low.")
            
            if len(df_filtered) > 10:
                iso = IsolationForest(contamination=0.05, random_state=42)
                df_filtered['Anomaly'] = iso.fit_predict(df_filtered[['Revenue_AED', 'Footfall']])
                
                anomalies = df_filtered[df_filtered['Anomaly'] == -1]
                
                fig_anom = px.scatter(df_filtered, x='Date', y='Revenue_AED', 
                                      color=df_filtered['Anomaly'].astype(str),
                                      color_discrete_map={'-1': 'red', '1': 'blue'},
                                      title="Red Dots = Detected Anomalies")
                st.plotly_chart(fig_anom, use_container_width=True)
                
                with st.expander("View Anomalous Dates Details"):
                    st.dataframe(anomalies[['Date', 'Revenue_AED', 'Footfall', 'Conversion_Rate']])
            else:
                st.warning("Not enough data for anomaly detection.")

        st.markdown("---")
        
        # --- ML 3: REVENUE PREDICTOR ---
        st.markdown("### 🔮 Revenue Driver & Simulator")
        
        features = ['Footfall', 'Avg_Ticket_AED', 'Day_Index', 'Is_Weekend']
        target = 'Revenue_AED'
        
        if len(df_filtered) > 10:
            X = df_filtered[features]
            y = df_filtered[target]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Simulator UI
            c_sim1, c_sim2, c_sim3 = st.columns(3)
            with c_sim1:
                s_foot = st.slider("Forecast Footfall", 50, 2000, int(df_filtered['Footfall'].mean()))
            with c_sim2:
                s_tix = st.slider("Forecast Ticket (AED)", 10.0, 500.0, float(df_filtered['Avg_Ticket_AED'].mean()))
            with c_sim3:
                s_day = st.selectbox("Scenario Day", ["Weekday", "Weekend"])
                is_wknd = 1 if s_day == "Weekend" else 0
                
            pred_rev = model.predict([[s_foot, s_tix, 4, is_wknd]])[0]
            st.metric(label="Predicted Revenue", value=f"AED {pred_rev:,.2f}", 
                      delta=f"{pred_rev - df_filtered['Revenue_AED'].mean():,.0f} vs Avg")

    # ==========================================
    # TAB 3: FINANCIAL SIMULATION
    # ==========================================
    with tab3:
        st.subheader("Financial Stress Testing")
        
        col_f1, col_f2 = st.columns([1,3])
        
        with col_f1:
            st.markdown("#### ⚙️ Cost Structure")
            fixed_cost = st.number_input("Monthly Fixed Cost (AED)", value=60000)
            cogs_pct = st.slider("Variable Cost (COGS) %", 10, 80, 30) / 100
            initial_invest = st.number_input("2021 Initial Investment (AED)", value=500000)

        with col_f2:
            st.markdown("#### 🌡️ Profit Sensitivity Matrix")
            st.caption("How does Net Profit change if Footfall or Conversion Rate shifts?")
            
            # Create Sensitivity Grid
            base_footfall = df_filtered['Footfall'].mean() * 30 # Monthly approx
            base_ticket = df_filtered['Avg_Ticket_AED'].mean()
            
            footfall_range = np.linspace(base_footfall * 0.5, base_footfall * 1.5, 10)
            ticket_range = np.linspace(base_ticket * 0.8, base_ticket * 1.2, 10)
            
            z_values = []
            for t in ticket_range:
                row = []
                for f in footfall_range:
                    rev = f * t # Footfall * Ticket = Revenue
                    var_cost = rev * cogs_pct
                    profit = rev - var_cost - fixed_cost
                    row.append(profit)
                z_values.append(row)
            
            # FIXED HERE: Removed 'midpoint', used 'zmid'
            fig_matrix = go.Figure(data=go.Heatmap(
                z=z_values,
                x=[f"{x:,.0f}" for x in footfall_range],
                y=[f"{y:.1f}" for y in ticket_range],
                colorscale='RdBu', 
                zmid=0,
                colorbar=dict(title='Net Profit')
            ))
            fig_matrix.update_layout(
                title="Monthly Profit Scenarios (X=Mthly Footfall, Y=Avg Ticket)",
                xaxis_title="Monthly Footfall",
                yaxis_title="Avg Ticket Size (AED)"
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

        st.markdown("---")
        
        # Cumulative Cash Flow / ROI
        st.markdown("### 💰 ROI Tracker")
        
        df_monthly = df_filtered.copy()
        df_monthly['YearMonth'] = df_monthly['Date'].dt.to_period('M')
        roi_data = df_monthly.groupby('YearMonth')['Revenue_AED'].sum().reset_index()
        
        # Calculate Monthly Profit
        roi_data['Monthly_Profit'] = (roi_data['Revenue_AED'] * (1-cogs_pct)) - fixed_cost
        roi_data['Cumulative_Cash'] = roi_data['Monthly_Profit'].cumsum() - initial_invest
        
        roi_data['YearMonth'] = roi_data['YearMonth'].astype(str)
        
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(x=roi_data['YearMonth'], y=roi_data['Monthly_Profit'], name='Monthly Net Profit'))
        fig_roi.add_trace(go.Scatter(x=roi_data['YearMonth'], y=roi_data['Cumulative_Cash'], name='Cumulative Cash Position', line=dict(color='orange', width=3)))
        
        fig_roi.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="Break-Even Point")
        
        fig_roi.update_layout(title="Payback Period & ROI Timeline", yaxis_title="AED Amount")
        st.plotly_chart(fig_roi, use_container_width=True)
