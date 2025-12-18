# Bunk Station Analytics Dashboard

This Streamlit application provides a comprehensive analysis of retail performance data ("Bunk_Station_Synthetic_Data"). It combines descriptive statistics, AI/ML-driven insights, and financial impact modeling to help visualize the ROI of fixed investments.

## Features

1.  **Descriptive Analytics:**
    * Interactive KPIs (Revenue, Footfall, Conversion Rate, Avg Ticket).
    * Trend analysis over time.
    * Correlation heatmaps to see relationships between traffic and sales.

2.  **Advanced AI/ML Analysis:**
    * **Revenue Driver Analysis:** Uses a Random Forest Regressor to determine which factors (Footfall vs. Ticket Size vs. Day of Week) most heavily influence revenue.
    * **What-If Scenarios:** An interactive ML tool to predict revenue changes based on simulated adjustments to Footfall or Marketing spend.

3.  **Financial Impact & Operating Leverage:**
    * **Profitability Calculator:** Input your monthly fixed costs to visualize your break-even point.
    * **Operating Leverage:** Demonstrates how the "front-loaded investment" strategy amplifies profit as revenue scales.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## Data Format
The app expects a CSV file named `Bunk_Station_Synthetic_Data.csv` (or uploaded via the UI) with the following columns:
* `Date` (YYYY-MM-DD)
* `Footfall` (Integer)
* `Avg_Ticket_AED` (Numeric)
* `Revenue_AED` (Numeric)
* `Orders` (Integer)
