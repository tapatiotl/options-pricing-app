import streamlit as st
import math
import scipy.stats as si
import numpy as np
import plotly.express as px

# Black-Scholes function with Greeks
def black_scholes_with_greeks(option_type, S, K, days, r, sigma, q):
    t = days / 365  # Convert days to years
    d1 = (math.log(S / K) + (r - q + (sigma ** 2) / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    # Calculate option price
    if option_type == "Call":
        price = S * math.exp(-q * t) * si.norm.cdf(d1) - K * math.exp(-r * t) * si.norm.cdf(d2)
    elif option_type == "Put":
        price = K * math.exp(-r * t) * si.norm.cdf(-d2) - S * math.exp(-q * t) * si.norm.cdf(-d1)
    else:
        return None, None, None, None, None, None  # Invalid option type

    # Calculate Greeks
    delta = math.exp(-q * t) * (si.norm.cdf(d1) if option_type == "Call" else si.norm.cdf(d1) - 1)
    gamma = (math.exp(-q * t) * si.norm.pdf(d1)) / (S * sigma * math.sqrt(t))
    vega = S * math.exp(-q * t) * math.sqrt(t) * si.norm.pdf(d1)
    theta = (- (S * math.exp(-q * t) * si.norm.pdf(d1) * sigma) / (2 * math.sqrt(t))
             - r * K * math.exp(-r * t) * si.norm.cdf(d2)
             + q * S * math.exp(-q * t) * si.norm.cdf(d1)) if option_type == "Call" else \
            (- (S * math.exp(-q * t) * si.norm.pdf(d1) * sigma) / (2 * math.sqrt(t))
             + r * K * math.exp(-r * t) * si.norm.cdf(-d2)
             - q * S * math.exp(-q * t) * si.norm.cdf(-d1))
    rho = K * t * math.exp(-r * t) * (si.norm.cdf(d2) if option_type == "Call" else -si.norm.cdf(-d2))

    return price, delta, gamma, theta, vega, rho

# Streamlit UI
st.title("📈 Black-Scholes Options Pricing with Greeks & Live Charts")

# User inputs
option_type = st.selectbox("Select Option Type", ["Call", "Put"])
S = st.number_input("Stock Price (S0)", min_value=0.01, value=100.0)
K = st.number_input("Strike Price (K)", min_value=0.01, value=110.0)
days = st.number_input("Days to Expiration", min_value=1, value=30)
r = st.number_input("Risk-Free Interest Rate (as decimal, e.g., 0.050 for 5%)", min_value=0.00, value=0.050)
sigma = st.number_input("Volatility (as decimal, e.g., 0.2 for 20%)", min_value=0.01, value=0.2)
q = st.number_input("Dividend Yield (as decimal, e.g., 0.03 for 3%)", min_value=0.0, value=0.0)

# Calculate option price and Greeks when user clicks button
if st.button("Calculate Option Price and Greeks"):
    price, delta, gamma, theta, vega, rho = black_scholes_with_greeks(option_type, S, K, days, r, sigma, q)
    if price is not None:
        st.success(f"💰 The fair price of the {option_type} option is: **${price:.2f}**")

        # Display Greeks
        st.subheader("📊 Option Greeks")
        col1, col2, col3 = st.columns(3)
        col1.metric("Delta (Δ)", f"{delta:.4f}")
        col2.metric("Gamma (Γ)", f"{gamma:.4f}")
        col3.metric("Theta (Θ)", f"{theta:.4f}")

        col4, col5 = st.columns(2)
        col4.metric("Vega (ν)", f"{vega:.4f}")
        col5.metric("Rho (ρ)", f"{rho:.4f}")

        # Sensitivity Analysis Charts
        st.subheader("📊 Option Price Sensitivity Analysis")

        # 1️⃣ Option Price vs. Stock Price (S)
        S_range = np.linspace(S * 0.5, S * 1.5, 50)  # Generate stock price range
        prices_vs_S = [black_scholes_with_greeks(option_type, s, K, days, r, sigma, q)[0] for s in S_range]

        fig1 = px.line(x=S_range, y=prices_vs_S, title="Option Price vs. Stock Price (S)")
        fig1.update_xaxes(title="Stock Price (S)")
        fig1.update_yaxes(title="Option Price")
        st.plotly_chart(fig1)

        # 2️⃣ Option Price vs. Volatility (σ)
        sigma_range = np.linspace(0.05, 1.0, 50)  # Vary volatility from 5% to 100%
        prices_vs_sigma = [black_scholes_with_greeks(option_type, S, K, days, r, s, q)[0] for s in sigma_range]

        fig2 = px.line(x=sigma_range, y=prices_vs_sigma, title="Option Price vs. Volatility (σ)")
        fig2.update_xaxes(title="Volatility (σ)")
        fig2.update_yaxes(title="Option Price")
        st.plotly_chart(fig2)

        # 3️⃣ Option Price vs. Time to Expiration (T)
        days_range = np.linspace(1, 365, 50)  # From 1 day to 1 year
        prices_vs_days = [black_scholes_with_greeks(option_type, S, K, d, r, sigma, q)[0] for d in days_range]

        fig3 = px.line(x=days_range, y=prices_vs_days, title="Option Price vs. Time to Expiration (T)")
        fig3.update_xaxes(title="Days to Expiration")
        fig3.update_yaxes(title="Option Price")
        st.plotly_chart(fig3)

    else:
        st.error("Invalid option type entered. Please select 'Call' or 'Put'.")
