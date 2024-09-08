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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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
    lead_lag_matrix = np.zeros((num_tickers, num_tickers), dtype=np.float32)
    
    for i in range(num_tickers):
        for j in range(i+1, num_tickers):
            # if i <= j:
            data_subset = current_data[:, [i, j]]
            lead_lag_matrix[i, j] = compute_lead_lag(data_subset, method=method)
    
    lead_lag_matrix = np.nan_to_num(lead_lag_matrix, nan=0)
    lead_lag_matrix = lead_lag_matrix - lead_lag_matrix.T
    
    return lead_lag_matrix



def identify_leaders_followers_t2(lead_lag_matrix, rank_method='RS_svd_kmeans', top_percent=20, bottom_percent=20):

 
    if rank_method == 'RS_svd_kmeans':
        num_stocks = lead_lag_matrix.shape[0]
        top_n = int(num_stocks * top_percent / 100)
        bottom_n = int(num_stocks * bottom_percent / 100)
        U, S, Vt = np.linalg.svd(lead_lag_matrix)
        
        lead_matrix = U[:, :top_n] * S[:top_n]
        follow_matrix = Vt.T[:, :bottom_n] * S[:bottom_n]
        
        # Step 2: Calculate cosine similarity between vectors
        similarity_matrix = cosine_similarity(lead_matrix)
        similarity_matrix2 = cosine_similarity(follow_matrix)

        # Step 3: Perform K-means clustering with 5 clusters
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(similarity_matrix)
        kmeans2 = KMeans(n_clusters=5, random_state=42)
        clusters2 = kmeans2.fit_predict(similarity_matrix2)

        # Step 4: Calculate the sum of each row (stock) as its value
        row_sums = np.sum(lead_lag_matrix, axis=1)

        # Step 5: Calculate the average sum for each cluster
        cluster_means = []
        for cluster in range(5):
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_mean = np.mean(row_sums[cluster_indices])
            cluster_means.append(cluster_mean)

        cluster_means2 = []
        for cluster in range(5):
            cluster_indices = np.where(clusters2 == cluster)[0]
            cluster_mean = np.mean(row_sums[cluster_indices])
            cluster_means2.append(cluster_mean)
        
        # Step 6: Identify the leaders and followers clusters
        leaders_cluster = np.argmax(cluster_means)
        followers_cluster = np.argmin(cluster_means2)

        # Step 7: Extract the indices of leaders and followers
        leaders = np.where(clusters == leaders_cluster)[0]
        followers = np.where(clusters2 == followers_cluster)[0]
        followers = np.setdiff1d(followers, leaders)
    elif rank_method == 'RS_mf_kmeans':
        steps = 20
        lambda_ = 0.05
        K = 20
        N = lead_lag_matrix.shape[0]
        M = lead_lag_matrix.shape[1]

        top_n = int(N * top_percent / 100)
        bottom_n = int(N * bottom_percent / 100)
        
        # Initialize matrices P and Q with random values, ensuring they are float64
        P = np.random.rand(N, K).astype(np.float64)
        Q = np.random.rand(M, K).astype(np.float64)
        lead_lag_matrix = lead_lag_matrix.astype(np.float64)
        
        for step in range(steps):
            # Fix Q, update P
            for i in range(N):
                P[i] = np.linalg.solve(np.dot(Q.T, Q) + lambda_ * np.eye(K), 
                                    np.dot(Q.T, lead_lag_matrix[i].T))
            
            # Fix P, update Q
            for j in range(M):
                Q[j] = np.linalg.solve(np.dot(P.T, P) + lambda_ * np.eye(K), 
                                    np.dot(P.T, lead_lag_matrix[:, j]))
        
        # Step 2: Calculate cosine similarity between vectors
        similarity_matrix = cosine_similarity(P)

        # Step 3: Perform K-means clustering with 5 clusters
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(similarity_matrix)

        # Step 4: Calculate the sum of each row (stock) as its value
        row_sums = np.sum(lead_lag_matrix, axis=1)

        # Step 5: Calculate the average sum for each cluster
        cluster_means = []
        for cluster in range(5):
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_mean = np.mean(row_sums[cluster_indices])
            cluster_means.append(cluster_mean)
        
        # Step 6: Identify the leaders and followers clusters
        leaders_cluster = np.argmax(cluster_means)
        followers_cluster = np.argmin(cluster_means)

        # Step 7: Extract the indices of leaders and followers
        leaders = np.where(clusters == leaders_cluster)[0]
        followers = np.where(clusters == followers_cluster)[0]

    elif rank_method == 'RS_mf_kmeans2':
        steps = 20
        lambda_ = 0.05
        K = 20
        N = lead_lag_matrix.shape[0]
        M = lead_lag_matrix.shape[1]

        top_n = int(N * top_percent / 100)
        bottom_n = int(N * bottom_percent / 100)
        
        # Initialize matrices P and Q with random values, ensuring they are float64
        P = np.random.rand(N, K).astype(np.float64)
        Q = np.random.rand(M, K).astype(np.float64)
        lead_lag_matrix = lead_lag_matrix.astype(np.float64)
        
        for step in range(steps):
            # Fix Q, update P
            for i in range(N):
                P[i] = np.linalg.solve(np.dot(Q.T, Q) + lambda_ * np.eye(K), 
                                    np.dot(Q.T, lead_lag_matrix[i].T))
            
            # Fix P, update Q
            for j in range(M):
                Q[j] = np.linalg.solve(np.dot(P.T, P) + lambda_ * np.eye(K), 
                                    np.dot(P.T, lead_lag_matrix[:, j]))
                
        # Step 2: Calculate cosine similarity between vectors
        similarity_matrix = cosine_similarity(P)
        similarity_matrix2 = cosine_similarity(Q)

        # Step 3: Perform K-means clustering with 5 clusters
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(similarity_matrix)
        clusters2 = kmeans.fit_predict(similarity_matrix2)

        # Step 4: Calculate the sum of each row (stock) as its value
        row_sums = np.sum(lead_lag_matrix, axis=1)

        # Step 5: Calculate the average sum for each cluster
        cluster_means = []
        for cluster in range(5):
            cluster_indices = np.where(clusters == cluster)[0]
            cluster_mean = np.mean(row_sums[cluster_indices])
            cluster_means.append(cluster_mean)

        cluster_means2 = []
        for cluster in range(5):
            cluster_indices = np.where(clusters2 == cluster)[0]
            cluster_mean = np.mean(row_sums[cluster_indices])
            cluster_means2.append(cluster_mean)
        
        # Step 6: Identify the leaders and followers clusters
        leaders_cluster = np.argmax(cluster_means)
        followers_cluster = np.argmin(cluster_means2)

        # Step 7: Extract the indices of leaders and followers
        leaders = np.where(clusters == leaders_cluster)[0]
        followers = np.where(clusters == followers_cluster)[0]
                
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


def plot_cumulative_returns(returns, data_index, bucket, lookback_window, method, rank_method):
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
    plt.savefig(f'./results_MF_kmeans/cum_return_bucket{bucket}_lookback_window{lookback_window}_{method}_{rank_method}.pdf')
    plt.close()

    
def process_combination(args):
    df, bucket, lookback_window, method, rank_method = args
    file_path = f'./results_MF_kmeans/returns_bucket{bucket}_lookback_window{lookback_window}_{method}_{rank_method}.pkl'
    if os.path.exists(file_path): 
        return

    print(f"Start bucket: {bucket}, lookback_window: {lookback_window}, method: {method}, rank_method: {rank_method}.")
    try:
        data = aggregate_log_return(df, bucket)
        data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
        data.set_index('datetime', inplace=True)
        data.drop(columns=['date', 'time'], inplace=True)
        market_returns = data['SPY']

        strategy_returns = []
        for t in tqdm(range(lookback_window, len(data)-1)):
            lead_lag_matrix = get_lead_lag_matrix_t(data, lookback_window, method, t)
            leaders_ind, followers_ind = identify_leaders_followers_t2(lead_lag_matrix, rank_method=rank_method)
            leaders = data.columns.values[leaders_ind]
            followers = data.columns.values[followers_ind]
        
            strategy_returns.append(execute_trades_t(data, leaders, followers, market_returns,t))

        if not strategy_returns:
            print(f"Skip bucket: {bucket}, lookback_window: {lookback_window}, method: {method}, rank_method: {rank_method}.")
            return
        
        pd.Series(strategy_returns).to_pickle(file_path)
        plot_cumulative_returns(strategy_returns, data_index=data.index, bucket=bucket, lookback_window=lookback_window, method=method, rank_method=rank_method)
        print(f"Finish bucket: {bucket}, lookback_window: {lookback_window}, method: {method}, rank_method: {rank_method}.")
    except Exception as e:
        print(f"Error bucket: {bucket}, lookback_window: {lookback_window}, method: {method}. Error: {e}")




if __name__ == '__main__':
    df = pd.read_csv('../0_data_preprocessing/log_returns_1min_252.csv')
    df.iloc[:,2:] = np.exp(df.iloc[:,2:])-1

    buckets = ['1d']
    lookback_window_dict = {
        '1d': [10],
    }
    methods = ['levy_area']  
    rank_methods = ['RS_svd_kmeans']

    combinations = [(df, bucket, lookback_window, method, rank_method) 
                for bucket in buckets 
                for lookback_window in lookback_window_dict[bucket] 
                for method in methods
                for rank_method in rank_methods]


    for comb in combinations:
        process_combination(comb)
