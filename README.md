# Stock Price Prediction using LSTM and RNN

This project focuses on predicting stock prices using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. By leveraging historical stock price data for Apple Inc. (AAPL) from Yahoo Finance, the project aims to build a robust prediction system that captures temporal dependencies in financial time series data, offering valuable insights for investors and financial analysts.

## Table of Contents
- [Project Overview](#project-overview)
- [Aim](#aim)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Approach](#approach)
  - [Neural Networks Basics](#neural-networks-basics)
  - [Loading Time Series Data](#loading-time-series-data)
  - [Data Transformations](#data-transformations)
  - [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
  - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
  - [Multivariate Input and LSTMs](#multivariate-input-and-lstms)
- [Modular Code Overview](#modular-code-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Challenges and Limitations](#challenges-and-limitations)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Stock price prediction is a vital task in financial markets, enabling investors and fund managers to make informed decisions. Accurate forecasts can help identify opportunities to maximize profits or minimize losses, optimize investment strategies, and improve risk management. Traditional methods often struggle with the complexity and volatility of financial data, but machine learning and deep learning techniques, such as RNN and LSTM, excel at capturing intricate temporal patterns. This project implements these advanced neural network models to predict stock prices, serving as a foundation for exploring deep learning in financial forecasting.

Stock price prediction plays a crucial role in financial markets, and accurate forecasts can have significant implications for businesses, investors, and financial institutions. 

Stock price prediction helps investors and fund managers make informed investment decisions. By accurately forecasting future stock prices, investors can identify potential opportunities for maximizing profits or minimizing losses. It enables them to allocate their capital wisely and adjust their investment portfolios based on predicted price movements.

Machine and Deep Learning have demonstrated their potential to enhance stock price prediction accuracy and assist investors, traders, and financial analysts in making informed decisions. By leveraging ML techniques, businesses can gain valuable insights, optimize investment strategies, and improve risk management in stock market's complex and dynamic realm.

The prediction of stock prices is a challenging task due to the inherent complexity and volatility of financial markets. Traditional methods often fail to capture the intricate patterns and dependencies present in stock price data. However, RNN and LSTM models have shown great potential in capturing temporal dependencies and making accurate predictions in various time series forecasting tasks.

## Aim
The primary objectives of this project are:
- To develop a robust stock price prediction system using RNN and LSTM models.
- To understand the fundamentals of deep learning and its application in financial forecasting, including how these models handle time series data.

## Dataset
The dataset comprises historical stock prices for Apple Inc. (AAPL), sourced from the Yahoo Finance API. It includes:
- **Daily Data**: Opening, closing, highest, and lowest prices.
- **Additional Features**: Trading volume for each trading day.

This comprehensive dataset provides the temporal sequence needed to train and test the RNN and LSTM models.

## Tech Stack
The project is built using the following tools and libraries:
- **Language**: Python
- **Libraries**:
  - `Keras`: For building and training neural networks.
  - `TensorFlow`: Backend for deep learning computations.
  - `Statsmodels`: For statistical modeling and analysis.
  - `NumPy`: For numerical operations.
  - `Pandas`: For data manipulation and analysis.
  - `yfinance`: To fetch stock price data from Yahoo Finance.
  - `pandas_datareader`: For additional data retrieval capabilities.

## Approach
The project follows a step-by-step methodology to design, train, and evaluate the stock price prediction models:

### Neural Networks Basics
- Review the structure and functionality of neural networks to establish a foundational understanding of deep learning concepts.

### Loading Time Series Data
- Retrieve historical stock price data for AAPL from Yahoo Finance using the `yfinance` library.

### Data Transformations
- **Feature Scaling/Normalization**: Standardize the data to a consistent range for better model performance.
- **Overlapping Window Creation**: Segment the time series into overlapping windows to prepare it for sequence-based training.

### Recurrent Neural Networks (RNN)
- **Model Building and Training**: Construct an RNN model to capture sequential patterns in the stock price data.
- **Sequence Generation and Evaluation**: Generate predictions and assess the model's accuracy using appropriate metrics.

### Long Short-Term Memory (LSTM)
- **Model Building and Training**: Implement an LSTM model to address long-term dependencies in the data, overcoming limitations of standard RNNs.
- **Sequence Generation and Evaluation**: Evaluate the LSTM model's predictive performance on the test dataset.

### Multivariate Input and LSTMs
- **Creating Technical Indicators**: Generate features like Exponential Moving Average (EMA) to enrich the dataset.
- **Creating Labels**: Define target variables (e.g., future stock prices) for supervised learning.
- **Model Building and Training**: Train an LSTM model with multivariate inputs.
- **Evaluation**: Analyze the model's performance using metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).

## Modular Code Overview
The project is organized into a modular structure for maintainability and scalability:
- **`lib/`**: Contains the original IPython notebook from the lectures as a reference.
- **`ml_pipeline/`**: Houses the machine learning pipeline code.
- **`engine.py`**: Main script to execute the prediction workflow.
- **`output/`**: Directory for storing model outputs, predictions, and visualizations.
- **`requirements.txt`**: Lists all Python dependencies.
- **`readme.md`**: This documentation file.

## Installation
To set up the project locally, follow these steps:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git


## **Execution Instructions**
<br>
 
### Option 1: Running on your computer locally
 
To run the notebook on your local system set up a [python](https://www.python.org/) environment. Set up the [jupyter notebook](https://jupyter.org/install) with python or by using [anaconda distribution](https://anaconda.org/anaconda/jupyter). Download the notebook and open a jupyter notebook to run the code on local system.
 
The notebook can also be executed by using [Visual Studio Code](https://code.visualstudio.com/), and [PyCharm](https://www.jetbrains.com/pycharm/).

**Python Version: 3.8.10**

* Create a python environment using the command 'python3 -m venv myenv'.

* Activate the environment by running the command 'myenv\Scripts\activate.bat'.

* Install the requirements using the command 'pip install -r requirements.txt'

* Run engine.py with the command 'python3 engine.py'.
 

 
### Option 2: Executing with Colab
Colab, or "Collaboratory", allows you to write and execute Python in your browser, with access to GPUs free of charge and easy sharing.
 
You can run the code using [Google Colab](https://colab.research.google.com/) by uploading the ipython notebook. 
