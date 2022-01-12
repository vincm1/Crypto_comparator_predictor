import streamlit as st
import yfinance as yf
from datetime import datetime, date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from prophet import Prophet
from prophet.plot import plot_plotly
#from plotly import graph_obs as go

st.title('Cryptocurrency comparator')

currencies = ("BTC-USD","ETH-USD","USDT-USD","BNB-USD","USDC-USD","SOL-USD","ADA-USD","HEX-USD","XRP-USD","LUNA1-USD","DOT-USD","AVAX-USD","DOGE-USD","SHIB-USD","MATIC-USD","BUSD-USD","CRO-USD","LINK-USD","WBTC-USD","UST-USD","UNI1-USD","ATOM-USD","NEAR-USD","DAI-USD","ALGO-USD","LTC-USD","BCH-USD","FTM-USD","TRX-USD","XLM-USD","ICP-USD","STETH-USD","MANA-USD","VET-USD","HBAR-USD","FTT-USD","BTCB-USD","FIL-USD","SAND-USD","AXS-USD","THETA-USD","ETC-USD","EGLD-USD","HNT-USD","XTZ-USD","ONE1-USD","LEO-USD","XMR-USD","KLAY-USD","MIOTA-USD","GRT1-USD","AAVE-USD","EOS-USD","STX-USD","GALA-USD","CAKE-USD","WBNB-USD","BTT-USD","FLOW-USD","LRC-USD","CRV-USD","RUNE-USD","QNT-USD","KSM-USD","MKR-USD","BSV-USD","ENJ-USD","XEC-USD","FRAX-USD","CVX-USD","ZEC-USD","AMP-USD","CELO-USD","HBTC-USD","KCS-USD","NEO-USD","AR-USD","BAT-USD","OKB-USD","CHZ-USD","KDA-USD","WAVES-USD","SAFEMOON-USD","HT-USD","DASH-USD","NEXO-USD","TUSD-USD","ROSE-USD","MINA-USD","YOUC-USD","YFI-USD","COMP1-USD","RVN-USD","CCXX-USD","HOT1-USD","XDC-USD","XYM-USD","XEM-USD","IOTX-USD","LN-USD")

start_date = st.date_input("Start date", value = datetime.date(datetime(2020, 1, 1)))
end_date = st.date_input("End date", value = datetime.today())

selected_currencies = st.multiselect("Select currencies for comparison", currencies, default = ["BTC-USD"])

@st.cache

def load_data(ticker):
    data = yf.download(ticker,start_date,end_date)
    data = data["Adj Close"]
    return data

data = load_data(selected_currencies)

st.header("Adj Close Prices")
st.line_chart(data)

st.header("Correlation")
fig3 = plt.figure()
sns.heatmap(data.corr(),annot=True)
st.pyplot(fig3)

### Log returns
log_ret = np.log(data/data.shift(1))

def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean()*weights*365)
    vola = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*365, weights)))
    sharp_ratio = ret / vola 
    return np.array([ret,vola,sharp_ratio])

def neg_sharp(weights):
    return get_ret_vol_sr(weights)[2]*-1

def check_sum(weights):
    return np.sum(weights)-1

cons = ({"type":"eq","fun":check_sum})

init_guess = [1/len(selected_currencies) for i in selected_currencies]

zero_one = ((0,1),)
bounds = ()

for i in range(0,len(selected_currencies)):
    bounds = bounds + zero_one

opt_result = minimize(neg_sharp, init_guess, method="SLSQP", bounds = bounds, constraints=cons)

st.header("Optimum Portfolio weight")
for i in range(len(selected_currencies)):
    st.write(selected_currencies[i], opt_result.x[i])


### Prophet 
# n_years = st.slider("Years of prediction:",1,3)
# period = n_years * 365 

### XG Boost / Random Forrest 