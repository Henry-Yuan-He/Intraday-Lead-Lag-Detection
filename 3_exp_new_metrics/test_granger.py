import pandas as pd
import numpy as np
import iisignature
from tqdm import tqdm
import matplotlib.pyplot as plt
from p_tqdm import p_map
import multiprocessing as mp
from multiprocessing import set_start_method, Pool
import os
from statsmodels.tsa.stattools import grangercausalitytests


set_start_method('spawn', force=True)
import matplotlib
matplotlib.use('Agg')
plt.ioff()

def aggregate_log_return(df, bucket):
    # Define the mapping of buckets to their respective minutes
    bucket_dict = {
        '1min': 1,
        '5min': 5,
        '10min': 10,
        '30min': 30,
        '1h': 60,
        '3h': 60*3,
        '5h': 60*5,
        '1d': 60*6.5, 
        '2d': 60*6.5*2,
        '7d': 60*6.5*7,
        '14d': 60*6.5*14,
        '21d': 60*6.5*21,
        '30d': 60*6.5*30
    }
    
    # Validate the provided bucket
    if bucket not in bucket_dict:
        raise ValueError("Invalid bucket. Please choose from '1min', '5min', '10min', '30min', '1h', '1d', '2d', '7d', '14d', '21d', '30d'.")
    
    # No processing needed for 1-minute intervals
    if bucket == '1min':
        return df
    
    # Number of rows to aggregate
    n = int(bucket_dict[bucket])
    
    # Save date and time columns
    dates = df['date'].values
    times = df['time'].values
    
    # Drop date and time columns for calculation
    data = df.drop(columns=['date', 'time']).values
    
    # Add 1 to all values
    data += 1
    
    # Determine the new number of rows
    new_row_count = len(data) // n
    
    # Reshape the data to (new_row_count, n, num_columns)
    reshaped_data = data[:new_row_count * n].reshape(new_row_count, n, -1)
    
    # Multiply along the second axis and subtract 1
    aggregated_data = reshaped_data.prod(axis=1) - 1
    
    # Extract the corresponding date and time for the new rows
    new_dates = dates[(np.arange(new_row_count) + 1) * n - 1]
    new_times = times[(np.arange(new_row_count) + 1) * n - 1]
    
    # Create the new DataFrame
    aggregated_df = pd.DataFrame(aggregated_data, columns=df.columns[2:])
    aggregated_df['date'] = new_dates
    aggregated_df['time'] = new_times
    
    # Reorder columns to put Date and Time first
    cols = ['date', 'time'] + [col for col in aggregated_df.columns if col not in ['date', 'time']]
    aggregated_df = aggregated_df[cols]
    
    return aggregated_df



from numba import njit
@njit
def cal_pearson_corr(Rp, Rq):
    mean_Rp = np.mean(Rp)
    mean_Rq = np.mean(Rq)
    std_Rp = np.std(Rp)
    std_Rq = np.std(Rq)

    cov = np.mean((Rp - mean_Rp) * (Rq - mean_Rq))
    correlation = cov / (std_Rp * std_Rq)
    return correlation

@njit
def compute_lead_lag_with_numba(data_subset, method):
    Rp = data_subset[:, 0]
    Rq = data_subset[:, 1]

    if method == 'ccf_at_lag': # paper 1, eq 1
        # lag = 1
        lag = 1
        Rp_lag = Rp[lag:]
        Rq_shifted = Rq[:-lag]
        # correlation = np.corrcoef(Rp_lag, Rq_shifted)[0, 1]
        correlation = cal_pearson_corr(Rp_lag, Rq_shifted)

        # lag = -1
        lag = -1
        Rp_lag = Rp[:lag]
        Rq_shifted = Rq[-lag:]
        # correlation2 = np.corrcoef(Rp_lag, Rq_shifted)[0, 1]
        correlation2 = cal_pearson_corr(Rp_lag, Rq_shifted)

        lead_lag_measure = correlation2 - correlation
        return lead_lag_measure

    elif method == 'argmax_cross_correlation': # paper 2, eq 1
        max_lag = 1
        correlations = np.zeros(2 * max_lag + 1)
        lag_range = np.arange(-max_lag, max_lag + 1)
        
        for i, lag in enumerate(lag_range):
            if lag > 0:
                Rp_lag = Rp[lag:]
                Rq_shifted = Rq[:-lag]
            elif lag < 0:
                Rp_lag = Rp[:lag]
                Rq_shifted = Rq[-lag:]
            else:
                Rp_lag = Rp
                Rq_shifted = Rq

            correlations[i] = cal_pearson_corr(Rp_lag, Rq_shifted)

        max_corr = correlations[0]
        best_lag = lag_range[0]
        for j in range(1, len(correlations)):
            if correlations[j] > max_corr:
                max_corr = correlations[j]
                best_lag = lag_range[j]
        lead_lag_measure = -best_lag
        return lead_lag_measure
    elif method == 'argmax_cross_correlation2': # paper 2, eq 1
            max_lag = 1
            correlations = np.zeros(2 * max_lag + 1)
            lag_range = np.arange(-max_lag, max_lag + 1)
            
            for i, lag in enumerate(lag_range):
                if lag > 0:
                    Rp_lag = Rp[lag:]
                    Rq_shifted = Rq[:-lag]
                elif lag < 0:
                    Rp_lag = Rp[:lag]
                    Rq_shifted = Rq[-lag:]
                else:
                    Rp_lag = Rp
                    Rq_shifted = Rq

                correlations[i] = cal_pearson_corr(Rp_lag, Rq_shifted)

            max_corr = abs(correlations[0])
            best_lag = lag_range[0]
            for j in range(1, len(correlations)):
                correlation = correlations[j]
                if abs(correlation) > max_corr:
                    max_corr = abs(correlation)
                    if correlation > 0:
                        best_lag = -lag_range[j]
                    else:
                        best_lag = lag_range[j]
            lead_lag_measure = best_lag
            return lead_lag_measure

    else:
        raise NotImplementedError
        
    

def cal_levy_area(paths):
    signature = iisignature.sig(paths, 2, 1)
    levy_area = signature[1][1] - signature[1][2]
    return levy_area

def compute_lead_lag(data_subset, method):
    if method in ['levy_area', 'signature']:
        lead_lag_measure = cal_levy_area(data_subset)
    else:
        lead_lag_measure = compute_lead_lag_with_numba(data_subset, method)
        
    return lead_lag_measure


def get_lead_lag_matrix_t(df, lookback_window, method, t):
    tickers = df.columns
    num_tickers = len(tickers)

    std_window = lookback_window
    df = (df - df.iloc[t-std_window:t].mean(axis=0)) / df.iloc[t-std_window:t].std(axis=0)
    current_data = df.iloc[t-lookback_window:t].values.astype(np.float32)
    # current_data = (current_data - current_data.mean(axis=0)) / current_data.std(axis=0)
    lead_lag_matrix = np.zeros((num_tickers, num_tickers), dtype=np.float32)
    
    for i in range(num_tickers):
        for j in range(i+1, num_tickers):
            # if i <= j:
            data_subset = current_data[:, [i, j]]
            lead_lag_matrix[i, j] = compute_lead_lag(data_subset, method=method)
    
    # lead_lag_matrix = lead_lag_matrix.fillna(0)
    lead_lag_matrix = np.nan_to_num(lead_lag_matrix, nan=0)
    lead_lag_matrix = lead_lag_matrix - lead_lag_matrix.T
    
    return lead_lag_matrix

def get_lead_lag_matrix_t_granger(df, lookback_window, method, t):
    tickers = df.columns
    num_tickers = len(tickers)

    std_window = lookback_window
    df2 = (df - df.iloc[t-std_window:t].mean(axis=0)) / df.iloc[t-std_window:t].std(axis=0)
    current_data = df2.iloc[t-lookback_window:t].values.astype(np.float32)
    # current_data = (current_data - current_data.mean(axis=0)) / current_data.std(axis=0)
    lead_lag_matrix = np.zeros((num_tickers, num_tickers), dtype=np.float32)
    granger_matrix = np.zeros((num_tickers, num_tickers), dtype=np.float32)
    
    for i in range(num_tickers):
        for j in range(num_tickers):
            if i != j:
                result = grangercausalitytests(current_data[:, [i, j]], maxlag=[1], verbose=False)

                p_value = result[1][0]['ssr_ftest'][1]

                if p_value < 0.05: 
                    f_statistic = result[1][0]['ssr_ftest'][0]
                    granger_matrix[i, j] = f_statistic
                else:
                    granger_matrix[i, j] = 0

    lead_lag_matrix = granger_matrix - granger_matrix.T
    lead_lag_matrix = np.nan_to_num(lead_lag_matrix, nan=0)

    np.fill_diagonal(lead_lag_matrix, 0)

    return lead_lag_matrix


def identify_leaders_followers_t(lead_lag_matrix, data, top_percent=20, bottom_percent=20):

    column_means = lead_lag_matrix.mean(axis=1)
    num_stocks = len(column_means)
    top_n = int(num_stocks * top_percent / 100)
    bottom_n = int(num_stocks * bottom_percent / 100)
    
    sorted_indices = np.argsort(column_means)
    leaders = data.columns[sorted_indices[-top_n:]].values 
    followers = data.columns[sorted_indices[:bottom_n]].values 
    
    return leaders, followers


def execute_trades_t(df, leaders, followers, market_returns, t):

    leaders_return = df.iloc[t][leaders].mean()

    followers_return = df.iloc[t + 1][followers].mean()
    market_return = market_returns.iloc[t + 1]

    if leaders_return >= 0:
        portfolio_return =  followers_return - market_return
    else:
        portfolio_return = market_return - followers_return

    return portfolio_return



def backtest_strategy(strategy_returns, bucket):
    unit_to_year = {
        '1min': 252 * 6.5 * 60, 
        '5min': 252 * 6.5 * 12,
        '10min': 252 * 6.5 * 6,
        '30min': 252 * 6.5 * 2,
        '1h': 252 * 6.5,
        '3h': 252 * 2.17,
        '5h': 252 * 1.3,
        '1d': 252,
        '2d': 252 / 2,
        '7d': 252 / 7,
        '14d': 252 / 14,
        '21d': 252 / 21,
        '30d': 252 / 30
    }
    
    trading_days_per_year = unit_to_year[bucket]
    strategy_returns = np.array(strategy_returns)
    
    cumulative_return = np.prod(1 + strategy_returns) - 1
    
    n_periods = len(strategy_returns) / trading_days_per_year
    annualized_return = (1 + cumulative_return) ** (1 / n_periods) - 1
    
    annualized_volatility = np.std(strategy_returns) * np.sqrt(trading_days_per_year)
    
    sharpe_ratio = annualized_return / annualized_volatility
    
    return annualized_return, annualized_volatility, sharpe_ratio


def plot_cumulative_returns(returns, data_index, bucket, lookback_window, method):
    annualized_return, annualized_volatility, sharpe_ratio = backtest_strategy(returns, bucket)

    cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
    times = [data_index[lookback_window + i] for i in range(len(returns))]
    years = [t.year + (t.day_of_year / 365.25) for t in times]
    
    plt.figure(figsize=(10, 6))
    plt.plot(years, cumulative_returns)
    plt.xlabel('Years')
    plt.ylabel('Cumulative Return (%)')
    plt.title(f'annualized return: {annualized_return*100:.3f}%, annualized volatility: {annualized_volatility*100:.3f}%, sharpe ratio: {sharpe_ratio:.3f}')
    plt.grid(True)
    plt.savefig(f'./results_granger/cum_return_bucket{bucket}_lookback_window{lookback_window}_{method}.pdf')
    plt.close()


def process_combination(args):
    df, bucket, lookback_window, method = args
    file_path = f'./results_granger/returns_bucket{bucket}_lookback_window{lookback_window}_{method}.pkl'
    if os.path.exists(file_path):
        return

    print(f"Start bucket: {bucket}, lookback_window: {lookback_window}, method: {method}.")
    try:
        data = aggregate_log_return(df, bucket)
        data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
        data.set_index('datetime', inplace=True)
        data.drop(columns=['date', 'time'], inplace=True)
        market_returns = data['SPY']

        strategy_returns = []
        for t in tqdm(range(lookback_window, len(data)-1)):
            lead_lag_matrix = get_lead_lag_matrix_t_granger(data, lookback_window, method, t)
            leaders, followers = identify_leaders_followers_t(lead_lag_matrix, data)
        
            strategy_returns.append(execute_trades_t(data, leaders, followers, market_returns,t))

        if not strategy_returns:
            print(f"Skip bucket: {bucket}, lookback_window: {lookback_window}, method: {method}.")
            return
        
        pd.Series(strategy_returns).to_pickle(file_path)
        plot_cumulative_returns(strategy_returns, data_index=data.index, bucket=bucket, lookback_window=lookback_window, method=method)
        print(f"Finish bucket: {bucket}, lookback_window: {lookback_window}, method: {method}.")
    except Exception as e:
        print(f"Error bucket: {bucket}, lookback_window: {lookback_window}, method: {method}. Error: {e}")


if __name__ == '__main__':
    df = pd.read_csv('../0_data_preprocessing/log_returns_1min_252.csv')
    df.iloc[:,2:] = np.exp(df.iloc[:,2:])-1

    buckets = ['1d']

    lookback_window_dict = {
        '1d': [10]
    }
    methods = ['granger']

    combinations = [(df, bucket, lookback_window, method) 
                for bucket in buckets 
                for lookback_window in lookback_window_dict[bucket] 
                for method in methods]

    for comb in combinations:
        process_combination(comb)
