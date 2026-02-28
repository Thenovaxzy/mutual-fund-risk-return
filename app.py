import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_loader import fetch_data
from metrics import calculate_metrics, perform_cross_sectional_regression

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Mutual Fund Risk-Return Forecaster", page_icon="📈", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("📈 Mutual Fund Risk-Return Tradeoff")
st.markdown("Forecast and visualize the relationship between risk (Beta/Volatility) and expected returns using Linear Regression (CAPM Framework).")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

default_tickers = "VTSAX, FXAIX, PRASX, FCNTX, VUG, VTV, VWELX, DODGX, QQQ, SCHD"
tickers_input = st.sidebar.text_area("Mutual Fund/ETF Tickers (comma-separated)", value=default_tickers)
benchmark = st.sidebar.text_input("Benchmark Index", value="^GSPC", help="S&P 500 is ^GSPC")
years = st.sidebar.slider("Historical Data Years", min_value=1, max_value=20, value=5)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.1) / 100

st.sidebar.markdown("---")
if st.sidebar.button("Run Analysis", type="primary"):
    
    with st.spinner(f"Fetching {years} years of data from Yahoo Finance..."):
        try:
            # Fetch data
            returns_df = fetch_data(tickers_input, benchmark_ticker=benchmark, years=years)
            
            if returns_df.empty:
                 st.error("No data fetched. Please check the tickers.")
            else:
                
                # Calculate metrics
                st.spinner("Calculating Risk and Return Metrics...")
                metrics_df = calculate_metrics(returns_df, benchmark_ticker=benchmark, risk_free_rate=risk_free_rate)
                
                if metrics_df.empty:
                      st.error("Error calculating metrics.")
                else:
                    # Perform Regression for Security Market Line
                    sml_model = perform_cross_sectional_regression(metrics_df)
                    
                    # --- Main Layout ---
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Security Market Line (SML) & Fund Positioning")
                        
                        # Plotly Scatter Plot
                        fig = px.scatter(
                            metrics_df, x="Beta", y="Return (Annualized)", 
                            text="Fund", hover_data=["Volatility (Annualized)", "Sharpe Ratio"],
                            title="Risk (Beta) vs. Return (Annualized)",
                            labels={"Beta": "Systematic Risk (Beta)", "Return (Annualized)": "Expected Annualized Return"},
                            color="Sharpe Ratio", color_continuous_scale=px.colors.sequential.Viridis
                        )
                        
                        # Add Regression Line (SML)
                        x_range = np.array([metrics_df['Beta'].min() * 0.9, metrics_df['Beta'].max() * 1.1])
                        y_pred = sml_model['slope'] * x_range + sml_model['intercept']
                        
                        fig.add_trace(go.Scatter(
                            x=x_range, y=y_pred, mode='lines', 
                            name='Security Market Line (Regression)',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_traces(textposition='top center')
                        fig.update_layout(height=600, showlegend=True, template="plotly_white")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        
                    with col2:
                        st.subheader("Regression Statistics (SML)")
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="margin-top:0;">Regression Equation</h4>
                            <p><i>Expected Return = ({sml_model['slope']:.4f} * Beta) + {sml_model['intercept']:.4f}</i></p>
                            <hr>
                            <p><b>Market Risk Premium (Slope):</b> {sml_model['slope']*100:.2f}%</p>
                            <p><b>Implied Risk-Free Rate (Intercept):</b> {sml_model['intercept']*100:.2f}%</p>
                            <p><b>R-Squared:</b> {sml_model['r_squared']:.4f}</p>
                            <p style="font-size:0.8rem; color:gray;">
                                Note: This represents the cross-sectional relationship across the selected funds. A high R-squared indicates that Beta explains a large portion of the variance in returns across these specific funds.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.subheader("Correlation Matrix (Returns)")
                        corr_matrix = returns_df.corr()
                        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
                        fig_corr.update_layout(height=400)
                        st.plotly_chart(fig_corr, use_container_width=True)
                        

                    # Data Table
                    st.subheader("Fund Metrics Data")
                    
                    # Format DataFrame for display
                    display_df = metrics_df.copy()
                    display_df["Return (Annualized)"] = display_df["Return (Annualized)"].apply(lambda x: f"{x*100:.2f}%")
                    display_df["Volatility (Annualized)"] = display_df["Volatility (Annualized)"].apply(lambda x: f"{x*100:.2f}%")
                    display_df["Alpha (Annualized)"] = display_df["Alpha (Annualized)"].apply(lambda x: f"{x*100:.2f}%")
                    display_df["Beta"] = display_df["Beta"].apply(lambda x: f"{x:.4f}")
                    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].apply(lambda x: f"{x:.4f}")
                    
                    st.dataframe(display_df, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.exception(e) # Good for debugging if something breaks
else:
    st.info("👈 Please configure the parameters in the sidebar and click 'Run Analysis' to begin.")
    
    st.markdown("""
    ### About the Model
    
    This application utilizes the **Capital Asset Pricing Model (CAPM)** framework:
    
    $$ E(R_i) = R_f + \\beta_i (E(R_m) - R_f) $$
    
    Where:
    *   $E(R_i)$: Expected return of the investment
    *   $R_f$: Risk-free rate
    *   $\\beta_i$: Beta of the investment (measure of systematic risk, calculated via linear regression of fund excess returns vs market excess returns)
    *   $E(R_m)$: Expected return of the market
    *   $(E(R_m) - R_f)$: Market risk premium
    
    The application performs two types of Linear Regression:
    1.  **Time-Series Regression:** To calculate the Beta ($\\beta$) for each individual fund against the benchmark over time.
    2.  **Cross-Sectional Regression:** To plot the Security Market Line (SML) showing the aggregate relationship between Beta and Expected Return across all selected funds.
    """)
