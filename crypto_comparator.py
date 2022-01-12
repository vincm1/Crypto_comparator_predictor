import streamlit as st
import yfinance as yf
from datetime import datetime, date
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from prophet import Prophet
from prophet.plot import plot_plotly
#from plotly import graph_obs as go


import pandas as pd
import numpy as np

START = "2015-01-01"
TODAY = "2021-12-31"

st.title('Cryptocurrency comparator')


currencies = ("BTC-USD","ETH-USD","USDT-USD","BNB-USD","USDC-USD","SOL-USD","ADA-USD","HEX-USD","XRP-USD","LUNA1-USD","DOT-USD","AVAX-USD","DOGE-USD","SHIB-USD","MATIC-USD","BUSD-USD","CRO-USD","LINK-USD","WBTC-USD","UST-USD","UNI1-USD","ATOM-USD","NEAR-USD","DAI-USD","ALGO-USD","LTC-USD","BCH-USD","FTM-USD","TRX-USD","XLM-USD","ICP-USD","STETH-USD","MANA-USD","VET-USD","HBAR-USD","FTT-USD","BTCB-USD","FIL-USD","SAND-USD","AXS-USD","THETA-USD","ETC-USD","EGLD-USD","HNT-USD","XTZ-USD","ONE1-USD","LEO-USD","XMR-USD","KLAY-USD","MIOTA-USD","GRT1-USD","AAVE-USD","EOS-USD","STX-USD","GALA-USD","CAKE-USD","WBNB-USD","BTT-USD","FLOW-USD","LRC-USD","CRV-USD","RUNE-USD","QNT-USD","KSM-USD","MKR-USD","BSV-USD","ENJ-USD","XEC-USD","FRAX-USD","CVX-USD","ZEC-USD","AMP-USD","CELO-USD","HBTC-USD","KCS-USD","NEO-USD","AR-USD","BAT-USD","OKB-USD","CHZ-USD","KDA-USD","WAVES-USD","SAFEMOON-USD","HT-USD","DASH-USD","NEXO-USD","TUSD-USD","ROSE-USD","MINA-USD","YOUC-USD","YFI-USD","COMP1-USD","RVN-USD","CCXX-USD","HOT1-USD","XDC-USD","XYM-USD","XEM-USD","IOTX-USD","LN-USD")
selected_currencies = st.selectbox("Select currencies for comparison", currencies)

n_years = st.slider("Years of prediction:",1,3)
period = n_years * 365 

@st.cache
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_currencies)

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"],y=data["Open"],name="Currency open"))
    fig.add_trace(go.Scatter(x=data["Date"],y=data["Close"],name="Currency close"))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data