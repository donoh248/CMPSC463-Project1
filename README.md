## Description of the Project

This project is an analysis tool designed to handle financial datasets, specifically focusing on Ethereum cryptocurrency data. The primary objective of this project is to process and analyze time-series data to uncover trends and detect any anomalies. The tool is built with a focus on providing accurate sorting of the dataset and analyzing patterns that emerge from the financial data. The key components of the project include data collection, shuffling for randomness(to sort later as is required for the project), implementation of merge sort, determining the subarray in the dataset to find maximum gain and loss, detect anomalies that go over a specific threshold, and visualization of the findings.

### Type-Specific Considerations

The project works with financial datasets where the temporal aspect (date) is crucial. The choice of Ethereum as a dataset provides a real-world application, especially given the volatility and trend-sensitive nature of cryptocurrency markets. The primary algorithm used is Merge Sort, which is selected due to its efficiency in handling large, shuffled datasets. Kadane's Algorithm is used for finding the Maximum Gain/Loss. This algorithm is employed to find the subarray (a sequence of days) with the maximum sum of price changes, which translates to the maximum potential gain (or loss) in a given time period. Kadane’s algorithm is a well-suited dynamic programming technique due to its linear time complexity, making it ideal for analyzing long sequences of financial data without significant performance penalties. This method allows us to quickly identify key market trends where price gains or losses were the highest. Closest Point Algorithm is used for Anomaly Detection. Detecting outliers is critical when working with financial data. The project uses a Closest Point algorithm to detect anomalies by analyzing the spatial relationships between data points. These anomalies can indicate sudden, unexpected price spikes or drops, providing useful insights into unusual market behavior. This approach ensures that the detection of outliers is efficient, even for large datasets. Additionally, the threshold can be configured to detect specific sizes of change in the dataset. Below is the website where the dataset was taken, and where multiple other cryptocurrencies exist that can be used for this program.
https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory?select=coin_Ethereum.csv

## Structure of the Code

### Block Diagram

Below is a conceptual block diagram of how the project is structured:

```
[Data Collection] --> [Data Shuffling] --> [Merge Sort Algorithm] --> [Finding Max Gain/Loss] --> [Anomaly Detection] --> [Trend Analysis & Visualization] --> [Report Generation]
```

1. **Data Collection**: Reads the Ethereum dataset from a CSV file (`coin_Ethereum.csv`).
2. **Data Shuffling**: Randomizes the dataset to ensure that the sorting algorithm can process unsorted data.
3. **Merge Sort Algorithm**: Implements a custom merge sort that sorts the dataset based on the date.
4. **Finding Max Gain/Loss**: Uses **Kadane's Algorithm** to detect the time period with the maximum price gain or loss, helping to identify significant market movements.
5. **Anomaly Detection**: Applies a **Closest Point Algorithm** to detect anomalies or outliers by identifying data points that deviate significantly from normal price movements.
6. **Trend Analysis & Visualization**: Uses libraries like `matplotlib` and `seaborn` to visualize trends in the data.
7. **Report Generation**: Produces visual and textual output summarizing the analysis.

### Summary of Classes and Methods

- **`merge_sort`**: Recursively divides the dataset into smaller arrays and sorts them.
- - **`kadane_algorithm`**: Implements Kadane’s Algorithm to find the subarray with the maximum sum of price changes, representing the period of highest gain or loss.
- **`closest_point_anomaly_detection`**: Detects anomalies in the dataset by applying the Closest Point algorithm, which identifies outliers based on the distance between data points in terms of price or time.
- **Visualization Methods**: Various plotting functions using `matplotlib` and `seaborn` to visualize trends in the Ethereum dataset.

## Instructions on How to Use the System

1. **Loading Data**: The system loads the Ethereum dataset from `coin_Ethereum.csv`. Ensure that this file is available in the same directory as the script.
2. **Performing Analysis**:
   - Run the script to shuffle and then sort the data using the custom merge sort algorithm.
   - The script automatically visualizes the trends in the sorted data.
3. **Anomaly Injection**: Should you wish to test the anomaly detection more, you can inject an anomaly which modifies one of the prices on a specific date. Simply put in a date as YYYY-MM-DD after the anomaly detection function. You can see the dates of the dataset in the report
4. **Generating Reports**: After the analysis, the script generates visual plots showing data trends over time.

To run the system:
- Ensure you have Python installed, along with the required packages (`pandas`, `matplotlib`, `seaborn`, `numpy`, etc.).
- Run the script from the command line or within an IDE that supports Python execution.

python CMPSC463_Proj1.py


## Verification of Code Functionality

Below are screenshots demonstrating the functionality of the system:

1. **Injecting an anomaly into the dataset**: One price in the dataset if multiplied by a large amount to create an anomaly
   ![Anomaly Injection](CMPSC463-Project1/Anomaly Injection.png)

2. **Report Output**: Generate the general information of the dataset.
   ![Report Output](CMPSC463-Project1/Report Output.png)

## Discussion of Findings

The analysis of Ethereum data revealed several key trends, including sharp price fluctuations that aligned with broader cryptocurrency market events. The project successfully implemented a merge sort to handle large, unsorted datasets, and the visualization tools provided insights into historical price trends.

### Limitations and Areas for Improvement

- **Scalability**: The current system handles medium-sized datasets efficiently. However, scaling it to handle real-time data streams or much larger datasets may require further optimization.
- **Anomaly Detection**: This isn't the most advanced anomaly detection. The program just analyzes the dataset and looks for any points where there is a percentage increase that goes over the threshold. While it can still find anomalies, it can mistake normal, large increases in price as anomalies

### Challenges Faced

- **Proper Sorting**: Originally sorting the dataset had some challenges. Rather than swapping the whole item after shuffling, it would only sort the date and leave the rest of the information untouched.
- **Determining Max Gain/Loss**: Implementing an efficient Algorithm to find the greatest gain and loss was tricky. While the program ran with no issue, the loss would go backwards and lead to inaccuracies
- **Closest Point Algorithm**: Detecting anomalies in large financial datasets using the Closest Point algorithm posed a challenge due to the complexity of calculating distances between data points.
