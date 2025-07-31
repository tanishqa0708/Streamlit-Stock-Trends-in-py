import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
from tensorflow import keras


# Function to load custom CSS
def load_custom_css(css_file):
    """Loads the CSS file for custom styling"""
    try:
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: {css_file} not found. Make sure the file exists.")

# Load the CSS file
load_custom_css("styles.css")


# Create columns: Left Sidebar | Main Content | Right Sidebar
col1, col2, col3 = st.columns([1, 3, 1])  # Adjust the ratio to resize sections

# Define the left sidebar content
with st.sidebar:
    st.markdown("""
        <div class='sidebar-left'>
            <h3>📌 Stock Insights</h3>
            <div class="sidebar-content">
    """, unsafe_allow_html=True)

    # Expander for Check Stock Trends
    with st.expander("🔹 Check Stock Trends"):
        st.markdown('<div class="expander-content">✔️ Daily & Weekly Trends</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Price Action Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Historical Volatility</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Support & Resistance Levels</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Stock Correlation Analysis</div>', unsafe_allow_html=True)

    # Expander for AI-based Predictions
    with st.expander("🔹 AI-based Predictions"):
        st.markdown('<div class="expander-content">✔️ Machine Learning Forecasts</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Sentiment Analysis on Stocks</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Neural Network-Based Trend Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Predictive Market Signals</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Earnings Forecasting</div>', unsafe_allow_html=True)

    # Expander for Moving Averages
    with st.expander("🔹 Moving Averages"):
        st.markdown('<div class="expander-content">✔️ Simple & Exponential Moving Averages</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Golden Cross & Death Cross Patterns</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Bollinger Bands</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Trend Strength Indicators</div>', unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)  # Close the sidebar-left div

# Define the right sidebar content
with st.sidebar:
    st.markdown("""
        <div class='sidebar-right'>
            <h3>🏦 Market News</h3>
    """, unsafe_allow_html=True)

    # Expander for Market Trends
    with st.expander("📢 Latest Finance Updates"):
        st.markdown('<div class="expander-content">✔️ Fed hikes interest rates.</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Inflation reports show mixed trends.</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Tech stocks are rebounding this quarter.</div>', unsafe_allow_html=True)

    # Expander for Global Market Trends
    with st.expander("📉 Global Market Trends"):
        st.markdown('<div class="expander-content">✔️ US & Europe markets show volatility.</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Asia-Pacific stocks are gaining strength.</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Emerging markets outperform predictions.</div>', unsafe_allow_html=True)

    # Expander for Trading Tips
    with st.expander("💡 Trading Tips & Guides"):
        st.markdown('<div class="expander-content">✔️ Always set stop-loss levels.</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Diversify your investments.</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Monitor economic indicators before trading.</div>', unsafe_allow_html=True)

    # Expander for Cryptocurrency Insights
    with st.expander("🌍 Cryptocurrency Insights"):
        st.markdown('<div class="expander-content">✔️ Bitcoin hovers around $50k.</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ Ethereum gas fees drop significantly.</div>', unsafe_allow_html=True)
        st.markdown('<div class="expander-content">✔️ New altcoins gain market attention.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # Close the sidebar-right div

#main content python code
start = '2010-01-01'
end = '2019-12-31'


st.title('Stock Prediction by Python')
user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input, start=start, end=end)


#describing data 
st.subheader('Data from 2010-2019')
st.write(df.describe()) 

#Visulisation 
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100  = df.Close.rolling(100).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100  = df.Close.rolling(100).mean()
ma200  = df.Close.rolling(200).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


#splitting data into traing and testing
import pandas as pd
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

custom_objects = {"mse": keras.losses.MeanSquaredError()}
model = keras.models.load_model("keras_model.h5", custom_objects=custom_objects)

#Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i -100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test_scaled = y_test * scale_factor


#Final Graph
st.subheader('Predictions vs Originals')
fig2, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_test_scaled , 'b', label='Original Price')
ax.plot(y_predicted, 'r', label='Predicted Price')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.legend()

st.pyplot(fig2)
