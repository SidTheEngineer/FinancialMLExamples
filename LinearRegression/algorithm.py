'''
Simple example of using Linear Regression to model a relationship between
time and price of a company's EOD, closing stock price
'''
import sys
sys.path.append('..')

import quandl
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import config

quandl.ApiConfig.api_key = config.quandl_api_key

start_date = datetime.date(2017, 1, 1)
end_date = datetime.date.today()

def get_daily_close_df(company_ticker):
    kwargs = {
        'start_date': start_date,
        'end_date': end_date,
        'collapse': "daily"
    }
    return quandl.get('EOD/{}.4'.format(company_ticker), **kwargs).reset_index()

def convert_to_1d_vector(df_data):
    return np.reshape(df_data, (len(df_data), 1))

def create_regressor(x_vector, y_vector):
    return LinearRegression().fit(x_vector, y_vector)

def generate_visuals(regressor, x_vector, y_vector):
    plt.scatter(x_vector, y_vector, color='green', label= 'Actual Price') #plotting the initial datapoints
    plt.plot(x_vector, regressor.predict(x_vector), color='red', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
    plt.title('Linear Regression | Time vs. Price')
    plt.legend()
    plt.xlabel('Date Integer')
    plt.show()

if __name__ == "__main__" :
    if len(sys.argv) < 2:
        print("Please provide a stock ticker as an arg for the example")
        sys.exit()


    daily_close_df = get_daily_close_df(sys.argv[1])
    prices = convert_to_1d_vector(daily_close_df['Close'].tolist())
    date_indeces = convert_to_1d_vector(daily_close_df.index.tolist())

    generate_visuals(create_regressor(date_indeces, prices), date_indeces, prices)
