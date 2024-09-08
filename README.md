# Detection of Lead-Lag Relationships in Stock Returns with Intraday Data

This project explores how using finer time resolutions, from daily to minute-level data, can improve the detection of lead-lag relationships and enhance trading strategies in financial markets. It introduces a new framework that reduces computational demands, making it feasible to use high-frequency data for better trading performance.

## Project Structure

### 0_data_preprocessing
This folder contains scripts for data preprocessing:
- `0_get_mid_price_1min_from_data_by_stocks.ipynb`
- `1_get_log_return_from_mid_price_1min.ipynb`
- `2_data_cleaning.ipynb`

### 1_exp_time_resolution
The script `test_resolution.py` in this folder generates the results for **Chapter 3: Time Resolution Analysis: From Monthly to Minutely**, evaluating the effect of different time resolutions and different lookback windows on lead-lag detection.

### 2_exp_new_framework
This folder contains `test_new_framework.py`, which produces results for **Chapter 5, Section 5.1: Validation of the Proposed Framework**. The new framework reduces computational complexity, making finer time resolutions feasible for real-time trading.

### 3_exp_new_metrics
This folder includes scripts for comparing different lead-lag metrics in daily and minute-level trading:
- `test_DCPLag1.py`, `test_DCPLag1_1min.py`
- `test_EWMA.py`, `test_EWMA_1min.py`
- `test_pcmci.py`, `test_pcmci_1min.py`
- `test_granger.py`, `test_granger_1min.py`

These scripts produce results for **Chapter 5, Section 5.2.2: Comparison of Pairwise Lead-Lag Metrics**.

### 4_exp_new_ranking_methods
This folder contains scripts for evaluating methods to identify leaders and followers:
- `test_MF_kmeans.py`, `test_MF_kmeans_1min.py`
- `test_greedy.py`, `test_greedy_1min.py`
- `test_biobj.py`, `test_biobj_1min.py`
- `test_sort_kmeans.py`, `test_sort_kmeans_1min.py`

These files generate results for **Chapter 5, Section 5.2.3: Comparison of Leaders and Followers Identification Methods**.

### 5_exp_new_trading_strategy
This folder contains scripts for preparing and testing trading strategies:
- `3_test_theta.py`, `3_test_theta_1min.py`: Scripts for testing the theta-based trading strategies.
  - `1_prepare_for_theta.py`, `1_prepare_for_theta_1min.py`, `2_prepare_for_theta.ipynb`, `2_prepare_for_theta_1min.ipynb`: Scripts that generate the leader average return distributions for the theta-based trading strategies.
- `test_biobj_weighted.py`, `test_biobj_weighted_1min.py`: Scripts for testing weighted bi-objective trading strategies.

These files generate results for **Chapter 5, Section 5.2.4: Comparison of Trading Strategies.**

