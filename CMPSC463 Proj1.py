#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import statistics
from scipy.spatial import cKDTree
from typing import List, Tuple


# In[2]:


# Collect Data (and shuffle it for MergeSort since its already sorted)
data = pd.read_csv('coin_Ethereum.csv')

data['Date'] = pd.to_datetime(data['Date'])
data = data.reindex(np.random.permutation(data.index))

print(data.head()) #printing to prove data is shuffled


# In[3]:


# Sort the dataset
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i, j = 0, 0
    
    while i < len(left) and j < len(right):
        if left[i].name < right[j].name:  # Compare the name attribute (which is the date)
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

# Load and prepare data
data = pd.read_csv('coin_Ethereum.csv')

# Date converted to index for easier understanding and organization
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Convert DataFrame to list of Series for merge sort
data_list = [row for _, row in data.iterrows()]

# Perform merge sort
sorted_data_list = merge_sort(data_list)

# Convert sorted list back to DataFrame
sorted_data = pd.DataFrame(sorted_data_list)

# Display the first few rows of the sorted data
print(sorted_data.head()) #printing to prove data is sorted now


# In[4]:


def kadane_like_max_gain_period(data):
    max_gain = 0
    current_gain = 0
    start_index = 0
    max_start_index = 0
    max_end_index = 0

    for i in range(1, len(data)):
        price_change = data.iloc[i]['Close'] - data.iloc[i-1]['Close']
        current_gain = max(0, current_gain + price_change)
        
        if current_gain > max_gain:
            max_gain = current_gain
            max_end_index = i
            max_start_index = start_index
        
        if current_gain == 0:
            start_index = i

    return max_start_index, max_end_index, max_gain

# Usage
max_start, max_end, max_gain = kadane_like_max_gain_period(sorted_data)
start_date = sorted_data.index[max_start]
end_date = sorted_data.index[max_end]
start_price = sorted_data.iloc[max_start]['Close']
end_price = sorted_data.iloc[max_end]['Close']

def kadane_like_max_loss_period(data):
    max_loss = 0
    current_loss = 0
    start_index = 0
    max_start_index = 0
    max_end_index = 0

    for i in range(1, len(data)):
        price_change = data.iloc[i]['Close'] - data.iloc[i-1]['Close']
        current_loss = min(0, current_loss + price_change)
        
        if current_loss < max_loss:
            max_loss = current_loss
            max_end_index = i
            max_start_index = start_index
        
        if current_loss == 0:
            start_index = i

    return max_start_index, max_end_index, abs(max_loss)

# Usage
max_start, max_end, max_loss = kadane_like_max_loss_period(sorted_data)
start_date = sorted_data.index[max_start]
end_date = sorted_data.index[max_end]
start_price = sorted_data.iloc[max_start]['Close']
end_price = sorted_data.iloc[max_end]['Close']

print(f"Maximum Gain Period:")
print(f"Start date: {start_date}, Price: ${start_price:.2f}")
print(f"End date: {end_date}, Price: ${end_price:.2f}")
print(f"Gain: ${max_gain:.2f}")
print("\n")
print(f"Maximum Loss Period:")
print(f"Start date: {start_date}, Price: ${start_price:.2f}")
print(f"End date: {end_date}, Price: ${end_price:.2f}")
print(f"Loss: ${max_loss:.2f}")

max_gain_start, max_gain_end, max_gain = kadane_like_max_gain_period(sorted_data)
max_loss_start, max_loss_end, max_loss = kadane_like_max_loss_period(sorted_data)


# In[5]:


# Detect Anomalies using closest pair points and add anomalies to see if it would be detected
import pandas as pd

def inject_anomaly(data, date, multiplier = 2.0) -> pd.DataFrame:
    modified_data = data.copy()
    try:
        # Append '23:59:59' to the input date
        full_datetime = f"{date} 23:59:59"
        date_to_modify = pd.to_datetime(full_datetime)
        
        if date_to_modify in modified_data.index:
            original_price = modified_data.loc[date_to_modify, 'Close']
            modified_data.loc[date_to_modify, 'Close'] *= multiplier
            print(f"Injected anomaly on {date_to_modify}: Original price ${original_price:.2f}, "
                  f"Modified price ${modified_data.loc[date_to_modify, 'Close']:.2f}")
        else:
            print(f"Date {full_datetime} not found in the dataset. No anomaly injected.")
            closest_date = modified_data.index[modified_data.index.get_loc(date_to_modify, method='nearest')]
            print(f"Closest available date in the dataset: {closest_date}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    return modified_data

def relative_distance(p1, p2):
    return abs((p2 - p1) / p1)

def find_anomalies_in_window(window, threshold):
    prices = window['Close'].values
    dates = window.index
    anomalies = []
    
    for i in range(len(prices) - 1):
        distance = relative_distance(prices[i], prices[i+1])
        if distance > threshold:
            anomalies.append((dates[i], prices[i]))
            anomalies.append((dates[i+1], prices[i+1]))
    
    return anomalies

def detect_anomalies(data, window_size = 30, threshold = 0.1):
    anomalies = set()
    
    for i in range(0, len(data) - window_size + 1):
        window = data.iloc[i:i+window_size]
        window_anomalies = find_anomalies_in_window(window, threshold)
        anomalies.update(window_anomalies)
    
    return sorted(list(anomalies))

# Usage
window_size = 30
threshold = .25 # Adjust this value to control sensitivity (.25 = detect 25%+ changes)

anomalies = detect_anomalies(sorted_data, window_size, threshold)


# In[6]:


# Inject an anomaly
anomaly_date = '2018-02-07 23:59:59'  # Choose a date in the dataset from 2015-08-08 to 2021-07-06
modified_data = inject_anomaly(sorted_data, anomaly_date, multiplier=3.0)

# Detect anomalies with injection(s)
anomalies = detect_anomalies(modified_data, window_size, threshold)


# In[7]:


# Generate a report
def generate_report(data: pd.DataFrame, anomalies: List[Tuple[pd.Timestamp, float]]) -> str:
    # Calculate cumulative return
    data['Daily_Return'] = data['Close'].pct_change()
    data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1

    report = "Ethereum Price Analysis Report\n"
    report += "==============================\n\n"

    # Overall statistics
    report += "1. Overall Statistics:\n"
    report += f"   Date range: {data.index[0].date()} to {data.index[-1].date()}\n"
    report += f"   Starting price: ${data['Close'].iloc[0]:.2f}\n"
    report += f"   Ending price: ${data['Close'].iloc[-1]:.2f}\n"
    total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    report += f"   Total return: {total_return:.2f}%\n\n"

    # Maximum profit period
    max_gain_start, max_gain_end, max_gain = kadane_like_max_gain_period(data)
    start_date = data.index[max_gain_start]
    end_date = data.index[max_gain_end]
    start_price = data.iloc[max_gain_start]['Close']
    end_price = data.iloc[max_gain_end]['Close']
    report += "2. Maximum Profit Period:\n"
    report += f"   Start date: {start_date.date()}, Price: ${start_price:.2f}\n"
    report += f"   End date: {end_date.date()}, Price: ${end_price:.2f}\n"
    report += f"   Maximum gain: ${max_gain:.2f}\n"
    report += f"   Return: {((end_price / start_price) - 1) * 100:.2f}%\n\n"

    # Maximum loss period
    max_loss_start, max_loss_end, max_loss = kadane_like_max_loss_period(data)
    start_date = data.index[max_loss_start]
    end_date = data.index[max_loss_end]
    start_price = data.iloc[max_loss_start]['Close']
    end_price = data.iloc[max_loss_end]['Close']
    report += "3. Maximum Loss Period:\n"
    report += f"   Start date: {start_date.date()}, Price: ${start_price:.2f}\n"
    report += f"   End date: {end_date.date()}, Price: ${end_price:.2f}\n"
    report += f"   Maximum loss: ${max_loss:.2f}\n"
    report += f"   Return: {((end_price / start_price) - 1) * 100:.2f}%\n\n"

    # Trend analysis
    report += "4. Price Trend Analysis:\n"
    yearly_returns = data.resample('YE')['Close'].last().pct_change()
    for year, return_value in yearly_returns.items():
        if pd.isna(return_value):
            report += f"   {year.year}: Initial year, no previous year for comparison\n"
        else:
            report += f"   {year.year}: {return_value*100:.2f}%\n"
    report += "\n"

    # Anomaly summary
    report += "5. Detected Anomalies:\n"
    report += f"   Total anomalies detected: {len(anomalies)}\n"
    report += "   Top 5 largest anomalies:\n"
    
    # Calculate percentage changes for all anomalies
    anomaly_changes = []
    for date, price in anomalies:
        prev_price = data.loc[:date].iloc[-2]['Close']
        change = (price - prev_price) / prev_price * 100
        anomaly_changes.append((date, price, change))
    
    # Sort anomalies by absolute percentage change
    sorted_anomalies = sorted(anomaly_changes, key=lambda x: abs(x[2]), reverse=True)
    
    for date, price, change in sorted_anomalies[:5]:
        report += f"   - Date: {date.date()}, Price: ${price:.2f}, Change: {change:.2f}%\n"

    return report, data

# Assuming you have 'modified_data' and 'anomalies' from previous steps
report_text, modified_data_with_returns = generate_report(modified_data, anomalies)

# Print the report
print(report_text)

# Save the report to a file
with open('ethereum_price_analysis_report.txt', 'w') as f:
    f.write(report_text)

print("Report saved as 'ethereum_price_analysis_report.txt'")

# Generate visualizations
plt.figure(figsize=(15, 10))
plt.plot(modified_data_with_returns.index, modified_data_with_returns['Close'], label='Close Price')
plt.title('Ethereum Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ethereum_price_trend.png')
plt.show()

plt.figure(figsize=(15, 10))
modified_data_with_returns['Cumulative_Return'].plot()
plt.title('Ethereum Cumulative Return')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ethereum_cumulative_return.png')
plt.show()

print("Visualizations saved as 'ethereum_price_trend.png' and 'ethereum_cumulative_return.png'")


# In[ ]:




