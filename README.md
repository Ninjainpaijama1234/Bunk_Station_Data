# Bunk Station Analytics Dashboard

This Streamlit application provides a comprehensive analysis of retail performance data. It is hardcoded to load `Bunk_Station_Synthetic_Data.xlsx` directly from the repository.

## Features

1.  **Descriptive Analytics:**
    * Interactive KPIs (Revenue, Footfall, Conversion Rate, Avg Ticket).
    * Trend analysis over time.
    * Daily conversion efficiency tracking.

2.  **Advanced AI/ML Analysis:**
    * **Revenue Driver Analysis:** Uses a Random Forest Regressor to determine which factors (Footfall vs. Ticket Size vs. Day of Week) most heavily influence revenue.
    * **What-If Scenarios:** An interactive ML tool to predict revenue changes based on simulated adjustments to Footfall or Ticket size.

3.  **Financial Impact & Operating Leverage:**
    * **Profitability Calculator:** Input monthly fixed costs to visualize the break-even point.
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

## Data Requirements
The app expects a file named `Bunk_Station_Synthetic_Data.xlsx` in the same directory, containing:
* `Date`
* `Footfall`
* `Avg_Ticket_AED`
* `Revenue_AED`
* `Orders`
