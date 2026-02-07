import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import io

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Stockify - Smart Inventory",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme and Styling
st.markdown("""
    <style>
        /* Force Dark Background and White Text */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #262730;
        }
        
        /* Metric Cards Styling */
        div[data-testid="metric-container"] {
            background-color: #1F2229;
            border: 1px solid #4B4B4B;
            padding: 15px;
            border-radius: 10px;
            color: white;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #4CAF50 !important; /* Green accent for headings */
        }
        
        /* Custom Button Styling */
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 5px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }

        /* Logo Text Styling */
        .logo-text {
            font-size: 40px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 0px;
        }
        .logo-sub {
            font-size: 16px;
            color: #A0A0A0;
            text-align: center;
            margin-bottom: 30px;
        }
    </style>"""
, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def generate_sample_data():
    """Generates a synthetic dataset for demonstration."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
    products = {
        'T-Shirt': {'cat': 'Apparel', 'price': 15},
        'Jeans': {'cat': 'Apparel', 'price': 40},
        'Sneakers': {'cat': 'Footwear', 'price': 80},
        'Socks': {'cat': 'Apparel', 'price': 5},
        'Backpack': {'cat': 'Accessories', 'price': 50},
        'Watch': {'cat': 'Accessories', 'price': 120},
        'Hat': {'cat': 'Accessories', 'price': 20},
        'Jacket': {'cat': 'Apparel', 'price': 90},
    }
    
    data = []
    for _ in range(500): # 500 records
        date = np.random.choice(dates)
        prod_name = np.random.choice(list(products.keys()))
        info = products[prod_name]
        units = np.random.randint(1, 20)
        revenue = units * info['price']
        
        data.append({
            'Date': date,
            'Product Name': prod_name,
            'Category': info['cat'],
            'Units Sold': units,
            'Unit Price': info['price'],
            'Revenue': revenue
        })
    
    return pd.DataFrame(data).sort_values('Date')

def process_data(df):
    """Basic cleaning and calculations."""
    # Ensure Date format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate Revenue if missing but Price and Units exist
    if 'Revenue' not in df.columns and 'Units Sold' in df.columns and 'Unit Price' in df.columns:
        df['Revenue'] = df['Units Sold'] * df['Unit Price']
        
    return df

def categorize_demand_level(df_grouped):
    """Categorize products into High, Medium, Low based on quantiles of Units Sold."""
    q1 = df_grouped['Units Sold'].quantile(0.33)
    q2 = df_grouped['Units Sold'].quantile(0.66)
    
    def get_level(units):
        if units >= q2: return 'High'
        elif units <= q1: return 'Low'
        else: return 'Medium'
        
    df_grouped['Demand Level'] = df_grouped['Units Sold'].apply(get_level)
    return df_grouped

# -----------------------------------------------------------------------------
# 3. SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'data' not in st.session_state:
    st.session_state['data'] = None

# -----------------------------------------------------------------------------
# 4. LOGIN PAGE
# -----------------------------------------------------------------------------
def login_page():
    # Placeholder for a Logo (SVG)
    st.markdown("""
        <div style="text-align: center;">
            <svg width="100" height="100" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M20 7L12 3L4 7M20 7V17L12 21M20 7L12 11M4 7V17L12 21M4 7L12 11M12 11V21" stroke="#4CAF50" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div class="logo-text">Stockify</div>
        <div class="logo-sub">Smart Inventory Solutions for Retailers</div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                # Simple hardcoded check for demo purposes
                if username == "admin" and password == "admin":
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("Invalid username or password (try admin/admin)")

# -----------------------------------------------------------------------------
# 5. DATA UPLOAD PAGE
# -----------------------------------------------------------------------------
def upload_page():
    st.title("📂 Import Sales Data")
    st.markdown("Upload your historical sales data to generate insights.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Check for required columns
                required_cols = ['Date', 'Product Name', 'Category', 'Units Sold', 'Revenue']
                missing = [c for c in required_cols if c not in df.columns]
                
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    st.session_state['data'] = process_data(df)
                    st.success("Data loaded successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")

    with col2:
        st.info("No data? Use our sample dataset to see how it works.")
        if st.button("Load Sample Data"):
            st.session_state['data'] = generate_sample_data()
            st.rerun()

# -----------------------------------------------------------------------------
# 6. DASHBOARD PAGES (Overview, Products, Trends, Insights)
# -----------------------------------------------------------------------------

def overview_page(df):
    st.title("📊 Dashboard Overview")
    
    # 1. KPIs
    total_rev = df['Revenue'].sum()
    total_units = df['Units Sold'].sum()
    total_products = df['Product Name'].nunique()
    avg_order_val = df['Revenue'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total_rev:,.2f}")
    c2.metric("Total Units Sold", f"{total_units:,}")
    c3.metric("Unique Products", total_products)
    c4.metric("Avg Sale Value", f"${avg_order_val:.2f}")

    st.markdown("---")

    # 2. Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Sales by Category")
        # Aggregation
        cat_df = df.groupby('Category')['Revenue'].sum().reset_index()
        fig_cat = px.bar(cat_df, x='Category', y='Revenue', color='Category', 
                         template='plotly_dark', title="Total Revenue per Category")
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_chart2:
        st.subheader("Revenue Distribution")
        fig_pie = px.pie(cat_df, values='Revenue', names='Category', 
                         template='plotly_dark', title="Sales Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Top 5 Products
    st.subheader("Top 5 Products by Revenue")
    prod_rev = df.groupby('Product Name')['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False).head(5)
    fig_top = px.bar(prod_rev, x='Revenue', y='Product Name', orientation='h', 
                     color='Revenue', template='plotly_dark', 
                     color_continuous_scale='Greens')
    st.plotly_chart(fig_top, use_container_width=True)

def products_page(df):
    st.title("📦 Product Analysis")
    
    # Aggregation for Product Level Analysis
    prod_group = df.groupby(['Product Name', 'Category']).agg({
        'Units Sold': 'sum',
        'Revenue': 'sum',
        'Date': 'count' # Using count to simulate avg sales occurrences
    }).reset_index()
    prod_group.rename(columns={'Date': 'Transaction Count'}, inplace=True)
    prod_group['Avg Sale'] = prod_group['Revenue'] / prod_group['Transaction Count']
    
    # Calculate Demand Levels
    final_df = categorize_demand_level(prod_group)
    
    # Demand Summary
    high_count = final_df[final_df['Demand Level'] == 'High'].shape[0]
    med_count = final_df[final_df['Demand Level'] == 'Medium'].shape[0]
    low_count = final_df[final_df['Demand Level'] == 'Low'].shape[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("🔥 High Demand Products", high_count)
    c2.metric("⚖️ Medium Demand Products", med_count)
    c3.metric("❄️ Low Demand Products", low_count)
    
    st.markdown("---")
    
    # Filter Option
    filter_demand = st.multiselect("Filter by Demand Level", ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])
    filtered_view = final_df[final_df['Demand Level'].isin(filter_demand)]
    
    # Display Table
    st.subheader("Detailed Product Inventory List")
    
    # Coloring the Demand Level column
    def color_demand(val):
        color = '#4CAF50' if val == 'High' else '#FFC107' if val == 'Medium' else '#FF5722'
        return f'color: {color}; font-weight: bold;'

    st.dataframe(
        filtered_view.style.map(color_demand, subset=['Demand Level'])
        .format({"Revenue": "${:.2f}", "Avg Sale": "${:.2f}"}),
        use_container_width=True
    )

def trends_page(df):
    st.title("📈 Trend Analysis")
    
    # Ensure date sorting
    df = df.sort_values('Date')
    
    # Time Granularity
    time_option = st.selectbox("Select Time Period", ["Daily", "Weekly", "Monthly"])
    
    if time_option == "Daily":
        resample_code = 'D'
    elif time_option == "Weekly":
        resample_code = 'W'
    else:
        resample_code = 'ME' # Month End

    # Resample Data
    trend_df = df.set_index('Date').resample(resample_code).sum(numeric_only=True).reset_index()
    
    # Revenue Trend
    st.subheader("Revenue Over Time")
    fig_rev = px.line(trend_df, x='Date', y='Revenue', markers=True, 
                      template='plotly_dark', line_shape='spline')
    fig_rev.update_traces(line_color='#4CAF50')
    st.plotly_chart(fig_rev, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Volume Trend")
        fig_vol = px.bar(trend_df, x='Date', y='Units Sold', template='plotly_dark')
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with col2:
        st.subheader("Category Performance Over Time")
        # Need to re-group including category for this chart
        cat_trend = df.groupby([pd.Grouper(key='Date', freq=resample_code), 'Category'])['Revenue'].sum().reset_index()
        fig_cat_trend = px.area(cat_trend, x='Date', y='Revenue', color='Category', template='plotly_dark')
        st.plotly_chart(fig_cat_trend, use_container_width=True)

def insights_page(df):
    st.title("💡 Smart Insights")
    
    # Preparation
    prod_group = df.groupby(['Product Name']).agg({'Units Sold': 'sum', 'Revenue': 'sum'}).reset_index()
    prod_group = categorize_demand_level(prod_group)
    
    high_demand = prod_group[prod_group['Demand Level'] == 'High']
    low_demand = prod_group[prod_group['Demand Level'] == 'Low']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 Stock Recommendations (Increase)")
        st.info("These items are moving fast. Consider increasing inventory to avoid stockouts.")
        if not high_demand.empty:
            for index, row in high_demand.iterrows():
                st.markdown(f"- **{row['Product Name']}** (Sold: {row['Units Sold']})")
        else:
            st.write("No high demand outliers detected.")
            
    with col2:
        st.markdown("### 🔴 Stock Recommendations (Reduce)")
        st.warning("These items are moving slowly. Consider running promotions or reducing future orders.")
        if not low_demand.empty:
            for index, row in low_demand.iterrows():
                st.markdown(f"- **{row['Product Name']}** (Sold: {row['Units Sold']})")
        else:
            st.write("No low demand items detected.")

    st.markdown("---")
    st.subheader("📚 General Best Practices")
    st.markdown("""
    > **1. The 80/20 Rule (Pareto Principle):** Usually, 80% of your revenue comes from 20% of your products. Focus your energy on keeping your High Demand items in stock.
    
    > **2. Seasonal Review:** Check the 'Trends' page to see if certain categories dip or spike during specific months. Prepare 2 months in advance.
    
    > **3. Dead Stock Clearance:** For items in the 'Reduce' list above, consider bundling them with popular items or offering a 'Buy 1 Get 1' discount to clear shelf space.
    """)

# -----------------------------------------------------------------------------
# 7. MAIN APP LOGIC
# -----------------------------------------------------------------------------

def main():
    if not st.session_state['logged_in']:
        login_page()
    else:
        # Sidebar Navigation
        with st.sidebar:
            st.markdown("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h2 style="color: #4CAF50;">Stockify</h2>
                </div>
            """, unsafe_allow_html=True)
            
            menu = st.radio("Navigation", ["Overview", "Products", "Trends", "Insights"])
            
            st.markdown("---")
            if st.button("Reset Data"):
                st.session_state['data'] = None
                st.rerun()
            
            if st.button("Logout"):
                st.session_state['logged_in'] = False
                st.session_state['data'] = None
                st.rerun()

        # Check if data is loaded
        if st.session_state['data'] is None:
            upload_page()
        else:
            df = st.session_state['data']
            
            if menu == "Overview":
                overview_page(df)
            elif menu == "Products":
                products_page(df)
            elif menu == "Trends":
                trends_page(df)
            elif menu == "Insights":
                insights_page(df)

if __name__ == "__main__":
    main()