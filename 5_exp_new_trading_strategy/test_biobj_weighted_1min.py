import pandas as pd
import numpy as np
import iisignature
from tqdm import tqdm
import matplotlib.pyplot as plt
from p_tqdm import p_map
import multiprocessing as mp
from multiprocessing import set_start_method, Pool
import os
from numba import njit
from numba.typed import List
import traceback

set_start_method('fork', force=True)
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
        return df.copy()
    
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


def identify_leaders_followers_t(lead_lag_matrix, data, top_percent=20, bottom_percent=20):

    column_means = lead_lag_matrix.mean(axis=1)
    num_stocks = len(column_means)
    top_n = int(num_stocks * top_percent / 100)
    bottom_n = int(num_stocks * bottom_percent / 100)
    
    sorted_indices = np.argsort(column_means)
    leaders = data.columns[sorted_indices[-top_n:]].values 
    followers = data.columns[sorted_indices[:bottom_n]].values 
    
    return leaders, followers

@njit
def calculate_lead_strength(matrix):
    row_sums = np.sum(matrix, axis=1)
    positive_counts = np.sum(matrix > 0, axis=1)
    return row_sums, positive_counts

@njit
def non_dominated_sort(row_sums, positive_counts):
    num_stocks = len(row_sums)
    domination_counts = np.zeros(num_stocks, dtype=np.int32)
    dominated_set = np.full((num_stocks, num_stocks), -1, dtype=np.int32)
    dominated_set_size = np.zeros(num_stocks, dtype=np.int32)
    
    fronts = np.full((num_stocks, num_stocks), -1, dtype=np.int32)
    front_sizes = np.zeros(num_stocks, dtype=np.int32)
    
    current_front = 0
    
    for i in range(num_stocks):
        for j in range(i + 1, num_stocks):
            dominates_i = dominates_j = False

            if (row_sums[i] > row_sums[j] and positive_counts[i] > positive_counts[j]) or \
               (row_sums[i] >= row_sums[j] and positive_counts[i] > positive_counts[j]) or \
               (row_sums[i] > row_sums[j] and positive_counts[i] >= positive_counts[j]):
                dominates_i = True
            
            if (row_sums[j] > row_sums[i] and positive_counts[j] > positive_counts[i]) or \
               (row_sums[j] >= row_sums[i] and positive_counts[j] > positive_counts[i]) or \
               (row_sums[j] > row_sums[i] and positive_counts[j] >= positive_counts[i]):
                dominates_j = True
            
            if dominates_i and not dominates_j:
                dominated_set[i, dominated_set_size[i]] = j
                dominated_set_size[i] += 1
                domination_counts[j] += 1
            elif dominates_j and not dominates_i:
                dominated_set[j, dominated_set_size[j]] = i
                dominated_set_size[j] += 1
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            fronts[0, front_sizes[0]] = i
            front_sizes[0] += 1

    while front_sizes[current_front] > 0:
        next_front_size = 0
        for i_idx in range(front_sizes[current_front]):
            i = fronts[current_front, i_idx]
            for j_idx in range(dominated_set_size[i]):
                j = dominated_set[i, j_idx]
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    fronts[current_front + 1, next_front_size] = j
                    next_front_size += 1
        current_front += 1
        front_sizes[current_front] = next_front_size

    result_fronts = []
    for i in range(current_front):
        front = fronts[i, :front_sizes[i]]
        result_fronts.append(front)

    return result_fronts

@njit
def calculate_front_weights(fronts, row_sums, is_leader=True):
    num_fronts = len(fronts)
    front_avg_strengths = np.zeros(num_fronts)
    front_weights = np.zeros(num_fronts)
    
    for i in range(num_fronts):
        front = fronts[i]
        if len(front) > 0:
            front_avg_strengths[i] = np.mean(row_sums[front])
    
    if is_leader:
        total_strength = 0.0
        for i in range(num_fronts):
            if front_avg_strengths[i] > 0:
                total_strength += front_avg_strengths[i]
        
        if total_strength > 0:
            for i in range(num_fronts):
                if front_avg_strengths[i] > 0:
                    front_weights[i] = front_avg_strengths[i] / total_strength
    
    else:
        total_strength = 0.0
        for i in range(num_fronts):
            if front_avg_strengths[i] < 0:
                total_strength -= front_avg_strengths[i]
        
        if total_strength > 0:
            for i in range(num_fronts):
                if front_avg_strengths[i] < 0:
                    front_weights[i] = -front_avg_strengths[i] / total_strength

    return front_weights

def identify_leaders_followers_t2(lead_lag_matrix, rank_method='biobj2', top_percent=20, bottom_percent=20):
    if rank_method == 'biobj2':
        row_sums, positive_counts = calculate_lead_strength(lead_lag_matrix)
        fronts = non_dominated_sort(row_sums, positive_counts)

        total_stocks = len(lead_lag_matrix)
        top_limit = int(np.ceil(total_stocks * top_percent / 100))
        bottom_limit = int(np.ceil(total_stocks * bottom_percent / 100))

        leaders = []
        followers = []

        leader_front_indices = []
        current_count = 0
        for i, front in enumerate(fronts):
            leaders.extend(front)
            leader_front_indices.append(i)
            current_count += len(front)
            if current_count > top_limit:
                break

        follower_front_indices = []
        current_count = 0
        for i in range(len(fronts) - 1, -1, -1): 
            front = fronts[i]
            followers.extend(front)
            follower_front_indices.append(i)
            current_count += len(front)
            if current_count > bottom_limit:
                break

        leader_weights_per_front = calculate_front_weights(List([fronts[i] for i in leader_front_indices]), row_sums, is_leader=True)
        follower_weights_per_front = calculate_front_weights(List([fronts[i] for i in follower_front_indices]), row_sums, is_leader=False)

        leader_weights = np.zeros(len(leaders))
        follower_weights = np.zeros(len(followers))

        leader_index = 0
        for i, front_index in enumerate(leader_front_indices):
            front = fronts[front_index]
            leader_weights[leader_index:leader_index+len(front)] = leader_weights_per_front[i]
            leader_index += len(front)

        follower_index = 0
        for i, front_index in enumerate(follower_front_indices):
            front = fronts[front_index]
            follower_weights[follower_index:follower_index+len(front)] = follower_weights_per_front[i]
            follower_index += len(front)

        # plot_leaders_followers(row_sums, positive_counts, leaders, followers, leader_weights, follower_weights)
    return np.array(leaders), np.array(followers), leader_weights, follower_weights


def execute_trades_t(df, leaders, followers, market_returns, t):

    leaders_return = df.iloc[t][leaders].mean()

    followers_return = df.iloc[t + 1][followers].mean()
    market_return = market_returns.iloc[t + 1]

    if leaders_return >= 0:
        portfolio_return =  followers_return - market_return
    else:
        portfolio_return = market_return - followers_return

    return portfolio_return

def execute_trades_weight_t(df, leaders, followers, leader_weights, follower_weights, market_returns, t):

    leaders_return = np.average(df.iloc[t][leaders], weights=leader_weights)

    followers_return = np.average(df.iloc[t + 1][followers], weights=follower_weights)
    market_return = market_returns.iloc[t + 1]

    if leaders_return >= 0:
        portfolio_return = followers_return - market_return
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


def plot_cumulative_returns(returns, data_index, bucket, lookback_window, method, update_interval, rank_method):
    update_interval_t, update_interval_label = update_interval
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
    plt.savefig(f'./results_biobj_weighted_1min/cum_return_bucket{bucket}_lookback_window{lookback_window}_{method}_update_interval_t{int(update_interval_t)}_label{update_interval_label}_{rank_method}.pdf')
    plt.close()

    
def process_combination(args):
    df, bucket, lookback_window, method, update_interval, rank_method = args
    update_interval_t, update_interval_label = update_interval
    file_path = f'./results_biobj_weighted_1min/returns_bucket{bucket}_lookback_window{lookback_window}_{method}_update_interval_t{int(update_interval_t)}_label{update_interval_label}_{rank_method}.pkl'
    if os.path.exists(file_path):  # skip already processed parameters
        return

    print(f"Start bucket: {bucket}, lookback_window: {lookback_window}, method: {method}, update_interval_t{int(update_interval_t)}_label{update_interval_label}.")
    try:
        data = aggregate_log_return(df.copy(), bucket)
        data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
        data.set_index('datetime', inplace=True)
        data.drop(columns=['date', 'time'], inplace=True)
        market_returns = data['SPY']  # Assuming SPY is the ticker for the market index

        strategy_returns = []
        lead_lag_matrix = None  # Initialize the lead-lag matrix
        last_update_t = lookback_window  # Track the last time the matrix was updated

        for t in tqdm(range(lookback_window, len(data)-1)):
            # Update lead-lag matrix at the first time step (t = lookback_window) and every `update_interval` steps
            if t == lookback_window or (t - last_update_t) >= update_interval_t:
                lead_lag_matrix = get_lead_lag_matrix_t(data, lookback_window, method, t)
                last_update_t = t  # Update the last update time step

                # leaders, followers = identify_leaders_followers_t(lead_lag_matrix, data)
                leaders_ind, followers_ind, leader_weights, follower_weights = identify_leaders_followers_t2(lead_lag_matrix, rank_method=rank_method)
                leaders = data.columns.values[leaders_ind]
                followers = data.columns.values[followers_ind]
        
            # Execute trades and calculate strategy returns
            strategy_returns.append(execute_trades_weight_t(data, leaders, followers, leader_weights, follower_weights, market_returns,t))

        if not strategy_returns:
            print(f"Skip bucket: {bucket}, lookback_window: {lookback_window}, method: {method}, update_interval_t{int(update_interval_t)}_label{update_interval_label}.")
            return
        
        pd.Series(strategy_returns).to_pickle(file_path)
        plot_cumulative_returns(strategy_returns, data_index=data.index, bucket=bucket, lookback_window=lookback_window, method=method, update_interval=update_interval, rank_method=rank_method)
        print(f"Finish bucket: {bucket}, lookback_window: {lookback_window}, method: {method}, update_interval_t{int(update_interval_t)}_label{update_interval_label}.")
    except Exception as e:
        print(f"Error bucket: {bucket}, lookback_window: {lookback_window}, method: {method}, update_interval_t{update_interval_t}_label{update_interval_label}. Error: {e}")


if __name__ == '__main__':
    df = pd.read_csv('../0_data_preprocessing/log_returns_1min_252.csv')
    df.iloc[:,2:] = np.exp(df.iloc[:,2:])-1

    buckets = ['1min']
    lookback_window_dict = {
        '1min': [100],
    }
    methods = ['levy_area']

    update_intervals_dict = {
        '1min': [
            (60 * 6.5 * 3, '3d'),
        ],
    }

    rank_methods = ['biobj2']

    combinations = [(df.copy(), bucket, lookback_window, method, update_interval, rank_method)
                for bucket in buckets 
                for lookback_window in lookback_window_dict[bucket]
                for method in methods
                for update_interval in update_intervals_dict[bucket]
                for rank_method in rank_methods]

    for comb in combinations:
        process_combination(comb)