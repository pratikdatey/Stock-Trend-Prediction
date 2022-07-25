
from copyreg import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from keras.models import load_model
import streamlit as st
import base64




start = '2010-01-01'
end   =  date.today()

st.title('Stock Trend Prediction')

st.markdown('Project by - **Pratik Datey**')
st.markdown("***")

user_input = st.text_input("Enter the Stock ",'ITC.NS')
st.write("You can Find The Name Of the Stocks From This [Link](https://finance.yahoo.com/quote/INFY?p=INFY&.tsrc=fin-srch)")


df= data.DataReader(user_input,'yahoo',start,end)

data = df.copy()

# Describing Data
st.markdown("***")
st.subheader('Stock Data from Year 2010 to Present Day')
st.write(data.describe())


# Visualizations
st.markdown("***")
st.subheader("High Price Vs Time")
high_plot = px.line(data['High'])
st.plotly_chart(high_plot)



st.markdown("***")
st.subheader("High Price Vs 100 Simple Moving Avg")
data['100_sma']=(data.High).rolling(100).mean()
fig= plt.figure(figsize=(14,7))
plt.plot(data['High'],label = 'High')
plt.plot(data['100_sma'],label='100 Simple Moving Avg')
plt.legend()
st.pyplot(fig)


st.markdown("***")
st.subheader("High Vs 200 Simple Moving Avg")
data['200_sma']=(data.High).rolling(200).mean()
fig= plt.figure(figsize=(14,7))
plt.plot(data['High'],label = 'High')
plt.plot(data['200_sma'],label = '200 Simple Moving Avg')
plt.legend()
st.pyplot(fig)



st.markdown("***")
st.subheader("High Vs 100 Simple Moving Avg Vs 200 Simple Moving Avg")
data['200_sma']=(data.High).rolling(200).mean()
fig= plt.figure(figsize=(14,7))
plt.plot(data['High'],label = 'High')
plt.plot(data['100_sma'],label='100 Simple Moving Avg')
plt.plot(data['200_sma'],label = '200 Simple Moving Avg')
plt.legend()
st.pyplot(fig)


st.markdown("***")
st.subheader("High Vs Exponential Weighted Moving Average")

data['ewma']=(data['High']).ewm(span=30, adjust=False).mean()


fig= plt.figure(figsize=(14,7))
plt.plot(data[['High','ewma']])
plt.legend()
st.pyplot(fig)




# Spliting data into Train and Test

data_training = pd.DataFrame(data['High'][0:int(len(data)*0.70)])
data_testing  = pd.DataFrame(data['High'][int(len(data)*0.70):])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Spliting data into x and y



# Load Model


model= load_model('keras_model100.h5')

# Testing Part

past_80 = data_training.tail(80)
final_df = past_80.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(80,input_data.shape[0]):
    x_test.append(input_data[i-80: i])
    y_test.append(input_data[i, 0])


x_test,y_test = np.array(x_test),np.array(y_test)

# Predictions 

y_predicted = model.predict(x_test)

scaler= scaler.scale_

scale_factor = 1/scaler[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Visuals

st.markdown("***")
st.subheader('Predicted Trend VS Original Trend')
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test,'g',label = 'Original Price')
plt.plot(y_predicted,'r',label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# prediction
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))

#tail=data_testing.tail(80)
#ms_tail=scaler.fit_transform(tail)
#test_pred=(ms_tail).reshape([1,80])

#pred_value=model.predict(test_pred)
#pred_value=int(pred_value * (tail.max()[0]))

#st.markdown("***")
#st.subheader("Prediction Price")

#st.write('Predicted HIGH price of {} stock on {}  is  {}.'.format(user_input,date.today(),pred_value))

