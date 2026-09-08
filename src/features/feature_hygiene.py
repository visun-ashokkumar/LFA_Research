import os
import sys
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler

# Suppress warnings to keep logs clean
warnings.filterwarnings("ignore")

def chronological_split(df, train_end="2024-12-31", holdout_end="2025-12-31", train_pct=None, holdout_pct=None, embargo_days=5):
    """
    Splits the dataframe chronologically into Train, Holdout, and Test partitions.
    Supports either hard dates or percentage-based splitting.
    Applies an embargo period (in days) to prevent overlap leakage.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    unique_dates = sorted(df['Date'].unique())
    
    if train_pct is not None and holdout_pct is not None:
        # Percentage-based splitting
        total_dates = len(unique_dates)
        train_idx = int(total_dates * train_pct)
        holdout_idx = int(total_dates * (train_pct + holdout_pct))
        
        # Keep indices within valid bounds
        train_idx = max(0, min(train_idx, total_dates - 1))
        holdout_idx = max(train_idx, min(holdout_idx, total_dates - 1))
        
        train_end_dt = unique_dates[train_idx]
        holdout_end_dt = unique_dates[holdout_idx]
        print(f"Percentage split calculated: Train end = {train_end_dt.strftime('%Y-%m-%d')}, Holdout end = {holdout_end_dt.strftime('%Y-%m-%d')}")
    else:
        # Date-based splitting
        train_end_dt = pd.to_datetime(train_end)
        holdout_end_dt = pd.to_datetime(holdout_end)
        
    embargo_offset = pd.Timedelta(days=embargo_days)
    
    train_df = df[df['Date'] <= train_end_dt].copy()
    
    holdout_start_dt = train_end_dt + embargo_offset
    holdout_df = df[(df['Date'] >= holdout_start_dt) & (df['Date'] <= holdout_end_dt)].copy()
    
    test_start_dt = holdout_end_dt + embargo_offset
    test_df = df[df['Date'] >= test_start_dt].copy()
    
    return train_df, holdout_df, test_df

def filter_collinearity(train_df, feature_cols, threshold=10.0, sample_size=20000, random_state=42, plot=False):
    """
    Iteratively computes VIF on a sample of train_df and drops columns exceeding the threshold.
    Uses the correlation matrix inverse diagonal shortcut for 100x faster execution.
    If plot=True, renders the VIF decay curve and before/after correlation heatmaps.
    """
    # 1. Filter out features that are mostly NaN (e.g. >50% NaNs in the training set)
    # This automatically drops mathematically impossible indicators like kurt_3d
    nan_ratios = train_df[feature_cols].isna().mean()
    valid_features = nan_ratios[nan_ratios <= 0.5].index.tolist()
    dropped_nan_features = set(feature_cols) - set(valid_features)
    if dropped_nan_features:
        print(f"Dropped {len(dropped_nan_features)} features due to high NaN ratio (>50%): {sorted(list(dropped_nan_features))}")
        
    retained_features = list(valid_features)
    
    # 2. Sample training set to speed up calculations
    if len(train_df) > sample_size:
        sample_df = train_df.sample(n=sample_size, random_state=random_state)
    else:
        sample_df = train_df.copy()
        
    # Instead of dropping rows with NaNs (which can empty the dataframe when many columns have few NaNs),
    # we fill NaNs with the column mean for the correlation calculation.
    sample_df = sample_df[retained_features].copy()
    for col in retained_features:
        col_mean = sample_df[col].mean()
        if pd.isna(col_mean):
            col_mean = 0.0
        sample_df[col] = sample_df[col].fillna(col_mean)
        
    if sample_df.empty:
        print("Warning: Sample dataframe is empty. No features dropped.")
        return valid_features
        
    # Store initial correlation for before heatmap
    if plot:
        corr_before = sample_df[retained_features].corr()
        
    iteration = 1
    vif_history = []
    while True:
        if len(retained_features) <= 1:
            break
            
        corr = sample_df[retained_features].corr().values
        # Add a tiny diagonal perturbation to guarantee invertibility (handling perfect duplicates)
        corr_perturbed = corr + np.eye(corr.shape[0]) * 1e-9
        
        try:
            inv_corr = np.linalg.inv(corr_perturbed)
            vifs = np.diag(inv_corr)
        except Exception:
            # Fallback to pseudo-inverse if inversion fails
            inv_corr = np.linalg.pinv(corr_perturbed)
            vifs = np.diag(inv_corr)
            
        max_idx = np.argmax(vifs)
        max_vif = vifs[max_idx]
        vif_history.append(float(max_vif))
        
        if max_vif > threshold:
            dropped_feature = retained_features.pop(max_idx)
            # Suppress excessive logging, but print milestone status every 20 features
            if iteration % 20 == 0 or iteration == 1:
                print(f"  Iteration {iteration}: Dropping '{dropped_feature}' with VIF = {max_vif:.2f}")
            iteration += 1
        else:
            break
            
    print(f"Collinearity filtering complete. Retained {len(retained_features)} / {len(feature_cols)} features.")
    
    # Render visual plots if plot=True
    if plot and vif_history:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 1. Plot VIF Decay Curve
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(vif_history) + 1), vif_history, marker='o', color='purple', linestyle='-', markersize=4)
            plt.axhline(y=threshold, color='red', linestyle='--', label=f'VIF Threshold = {threshold}')
            plt.title('VIF Decay Curve during Iterative Collinearity Filtering')
            plt.xlabel('Iteration Number')
            plt.ylabel('Max VIF Score')
            plt.yscale('log')  # Log scale for extremely large initial VIF values
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # 2. Plot Correlation Heatmap (Before vs After)
            corr_after = sample_df[retained_features].corr()
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            sns.heatmap(corr_before, cmap='coolwarm', ax=axes[0], xticklabels=False, yticklabels=False, cbar=True, vmin=-1.0, vmax=1.0)
            axes[0].set_title(f'Correlation Matrix (Before VIF: {len(valid_features)} features)')
            
            sns.heatmap(corr_after, cmap='coolwarm', ax=axes[1], xticklabels=False, yticklabels=False, cbar=True, vmin=-1.0, vmax=1.0)
            axes[1].set_title(f'Correlation Matrix (After VIF: {len(retained_features)} features)')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Warning: matplotlib or seaborn is not installed. Skipping visualization.")
        except Exception as e:
            print(f"Warning: Failed to render VIF visualization: {str(e)}")
            
    return retained_features

def get_weights(d, size):
    """
    Generates fractional differentiation weights up to a given lookback size.
    """
    w = [1.0]
    for k in range(1, size):
        w.append(w[-1] / k * (k - 1 - d))
    return np.array(w)

def frac_diff_series(series, d, threshold=1e-4, max_lookback=250):
    """
    Applies fractional differentiation to a single pandas Series.
    """
    if d == 0.0:
        return series
        
    w = [1.0]
    for k in range(1, len(series)):
        val = w[-1] / k * (k - 1 - d)
        if abs(val) < threshold or k > max_lookback:
            break
        w.append(val)
    w = np.array(w)
    
    summed = None
    for k in range(len(w)):
        shifted = series.shift(k) * w[k]
        if summed is None:
            summed = shifted
        else:
            summed = summed + shifted
    return summed

def get_optimal_d_for_column(train_df, col, symbols_to_test, start_d=0.1, step_d=0.1, threshold=1e-4):
    """
    Finds the minimum d in (0.1, 1.0) that makes the column stationary across test symbols.
    If the original column is already stationary (d=0.0), returns 0.0.
    """
    # 1. Test d=0.0 (original series)
    p_values = []
    for sym in symbols_to_test:
        series = train_df[train_df['Symbol'] == sym][col].dropna()
        if len(series) < 30: # Need enough history for ADF
            continue
        try:
            res = adfuller(series, maxlag=10)
            p_values.append(res[1]) # p-value
        except Exception:
            pass
            
    if p_values and np.median(p_values) <= 0.05:
        return 0.0
        
    # 2. Grid search d from 0.1 to 1.0
    for d in np.arange(start_d, 1.0 + 1e-5, step_d):
        p_values_d = []
        for sym in symbols_to_test:
            series = train_df[train_df['Symbol'] == sym][col].dropna()
            if len(series) < 30:
                continue
            
            diff_series = frac_diff_series(series, d, threshold=threshold)
            diff_series_clean = diff_series.dropna()
            if len(diff_series_clean) < 30:
                continue
                
            try:
                res = adfuller(diff_series_clean, maxlag=10)
                p_values_d.append(res[1])
            except Exception:
                pass
                
        if p_values_d and np.median(p_values_d) <= 0.05:
            return round(d, 2)
            
    return 1.0

def process_stationarity_and_diff(df, train_df, feature_cols, max_workers=8, threshold=1e-4):
    """
    Identifies non-stationary features, finds their optimal d from train_df,
    and applies fractional differentiation to the full dataframe.
    """
    print(f"Reviewing stationarity for {len(feature_cols)} features...")
    
    # Select 5 symbols with the most training observations to use as representatives
    symbol_counts = train_df['Symbol'].value_counts()
    test_symbols = symbol_counts.head(5).index.tolist()
    
    if not test_symbols:
        test_symbols = train_df['Symbol'].unique()[:5].tolist()
        
    # 1. Determine optimal d for all features concurrently
    optimal_d_dict = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_optimal_d_for_column, train_df, col, test_symbols, threshold=threshold): col
            for col in feature_cols
        }
        
        for future in as_completed(futures):
            col = futures[future]
            try:
                optimal_d = future.result()
                optimal_d_dict[col] = optimal_d
            except Exception as e:
                print(f"Error calculating stationarity for {col}: {str(e)}")
                optimal_d_dict[col] = 0.0 # Fallback to no differentiation
                
    non_stationary_cols = {col: d for col, d in optimal_d_dict.items() if d > 0.0}
    print(f"Stationarity review complete: {len(non_stationary_cols)} / {len(feature_cols)} features are non-stationary.")
    for col, d in non_stationary_cols.items():
        print(f"  Feature '{col}' will be fractionally differentiated with d = {d}")
        
    # 2. Apply differentiation to the full dataframe (grouping by Symbol to prevent boundary bleeding)
    df_diff = df.copy()
    
    # We apply shifts grouping by symbol
    for col, d in non_stationary_cols.items():
        # Get decayed weights lookback count
        w = [1.0]
        for k in range(1, 1000):
            val = w[-1] / k * (k - 1 - d)
            if abs(val) < threshold or k > 250:
                break
            w.append(val)
        w = np.array(w)
        
        summed = None
        grouped = df_diff.groupby('Symbol')[col]
        for k in range(len(w)):
            shifted = grouped.shift(k) * w[k]
            if summed is None:
                summed = shifted
            else:
                summed = summed + shifted
                
        df_diff[col] = summed
        
    return df_diff, optimal_d_dict

def scale_features(train_df, holdout_df, test_df, feature_cols):
    """
    Fits a StandardScaler on the Train partition and scales Train, Holdout, and Test.
    Returns scaled copies of the dataframes and the scaler object.
    """
    print("Fitting StandardScaler on Train partition...")
    scaler = StandardScaler()
    
    # Fit strictly on Train
    scaler.fit(train_df[feature_cols])
    
    # Transform copies of all three
    train_scaled = train_df.copy()
    holdout_scaled = holdout_df.copy()
    test_scaled = test_df.copy()
    
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    
    if not holdout_df.empty:
        holdout_scaled[feature_cols] = scaler.transform(holdout_df[feature_cols])
    if not test_df.empty:
        test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
        
    return train_scaled, holdout_scaled, test_scaled, scaler
