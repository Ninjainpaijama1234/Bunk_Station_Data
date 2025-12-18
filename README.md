# Bunk Station Retail Analytics Dashboard

This Streamlit application provides a comprehensive suite of retail analytics, ranging from descriptive KPIs to advanced Machine Learning forecasting and financial impact modeling.

## Features

* **Descriptive Analysis:** View key metrics (Total Revenue, Footfall, Conversion Rate) and visualize trends over time.
* **Deep Dive Analysis:** Explore the correlation between Footfall and Conversion, and analyze day-of-week performance.
* **Advanced ML Forecasting:** Uses the `Prophet` library to predict future Revenue and Footfall for the next 30 days.
* **Financial Impact Simulator:** An interactive tool to see how improvements in Conversion Rate or Average Ticket size impact total Revenue.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Preparation:**
    Ensure your data file is named `Bunk_Station_Synthetic_Data.csv` (or `.xlsx` if it's an Excel file) and is located in the root directory.
    * *Note: The code currently defaults to looking for a CSV file. If you have an Excel file, please modify the `load_data` function in `app.py` to use `pd.read_excel`.*

4.  **Run the Dashboard:**
    ```bash
    streamlit run app.py
    ```

## Data Columns Required
The input file must contain the following headers:
* `Date`
* `Footfall`
* `Avg_Ticket_AED`
* `Revenue_AED`
* `Orders`
