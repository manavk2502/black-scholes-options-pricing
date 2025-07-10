import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Options Pricing Model", layout="wide", initial_sidebar_state="expanded")

# Global font size adjustment
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-size: 20px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📊 Heatmap Parameters")
S = st.sidebar.number_input("Current Asset Price", value=100.0)
K = st.sidebar.number_input("Strike Price", value=105.0)
T = st.sidebar.number_input("Time to Maturity (Years)", value=1.0)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
r = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05)

# Black-Scholes pricing formulas
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

# Page title and description
st.markdown("## 📌 Options Price - Interactive Heatmap")
st.markdown(
    "Explore how option prices fluctuate with varying 'Spot Prices' and Volatility levels "
    "using interactive heatmap parameters, all while maintaining a constant 'Strike Price'."
)

# Spot and Volatility Ranges
spot_prices = np.round(np.linspace(S * 0.8, S * 1.2, 8), 2)
min_vol = round(sigma - 0.20, 2)  # 5 levels below sigma
max_vol = round(sigma + 0.20, 2)  # 5 levels above sigma
volatilities = np.round(np.arange(min_vol, max_vol + 0.001, 0.04), 2)

# Create Heatmap DataFrames
call_matrix = np.zeros((len(volatilities), len(spot_prices)))
put_matrix = np.zeros((len(volatilities), len(spot_prices)))

for i, vol in enumerate(volatilities):
    for j, spot in enumerate(spot_prices):
        call_matrix[i, j] = black_scholes_call(spot, K, T, r, vol)
        put_matrix[i, j] = black_scholes_put(spot, K, T, r, vol)

call_df = pd.DataFrame(call_matrix, index=volatilities, columns=spot_prices)
put_df = pd.DataFrame(put_matrix, index=volatilities, columns=spot_prices)

# Draw Heatmaps in separate square white boxes
col1, col2 = st.columns([1, 1])

with col1:
    fig1, ax1 = plt.subplots(figsize=(10, 10), dpi=100)
    ax1.set_facecolor('white')
    fig1.patch.set_facecolor('white')
    sns.heatmap(call_df, annot=True, fmt=".2f", cmap="viridis", ax=ax1, 
                cbar_kws={"label": "Price ($)"}, annot_kws={"size": 9})
    ax1.set_title("Call Price Heatmap", fontsize=18)
    ax1.set_xlabel("Spot Price", fontsize=14)
    ax1.set_ylabel("Volatility", fontsize=14)
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(10, 10), dpi=100)
    ax2.set_facecolor('white')
    fig2.patch.set_facecolor('white')
    sns.heatmap(put_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax2, 
                cbar_kws={"label": "Price ($)"}, annot_kws={"size": 9})
    ax2.set_title("Put Price Heatmap", fontsize=18)
    ax2.set_xlabel("Spot Price", fontsize=14)
    ax2.set_ylabel("Volatility", fontsize=14)
    st.pyplot(fig2)

# Footer disclaimer
st.markdown(
    "<br><hr style='border: 1px solid gray'><small><i>Disclaimer: This is a demonstration app for educational purposes and does not constitute financial advice.</i></small>",
    unsafe_allow_html=True,
)