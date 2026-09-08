import numpy as np
import pandas as pd

def calculate_symbol_features(df_symbol, windows, threshold_1d=0.0, threshold_5d=0.0):
    """
    Computes over 200 technical, momentum, volatility, and volume indicators for a single symbol.
    
    Parameters:
    -----------
    df_symbol : pd.DataFrame
        Dataframe containing a single stock's historical data (must be sorted by Date).
    windows : list of int
        Rolling windows to use for indicator calculations.
    threshold_1d : float
        Threshold for T+1 classification target.
    threshold_5d : float
        Threshold for T+5 classification target.
        
    Returns:
    --------
    pd.DataFrame
        Dataframe copy with added features and isolated targets.
    """
    df = df_symbol.copy()
    
    # 1. Base Prices & Volumes
    Open = df['Open']
    High = df['High']
    Low = df['Low']
    Close = df['Close']
    Volume = df['Volume']
    
    new_features = {}
    
    # Pre-calculate standard returns for volatilities and stats
    ret1d = Close.pct_change(1)
    ret5d = Close.pct_change(5)
    
    # 2. Multi-window Technical Indicators
    for k in windows:
        # Momentum & Returns
        new_features[f'return_{k}d'] = Close.pct_change(k)
        new_features[f'log_return_{k}d'] = np.log(Close / Close.shift(k))
        new_features[f'roc_{k}d'] = (Close - Close.shift(k)) / Close.shift(k)
        
        # Volatility & Risk
        new_features[f'volatility_{k}d'] = ret1d.rolling(k).std()
        
        # True Range and ATR
        high_low = High - Low
        high_close = (High - Close.shift(1)).abs()
        low_close = (Low - Close.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        new_features[f'atr_{k}d'] = tr.rolling(k).mean()
        
        # Trend
        new_features[f'ema_ratio_{k}d'] = Close / Close.ewm(span=k, adjust=False).mean()
        new_features[f'sma_ratio_{k}d'] = Close / Close.rolling(k).mean()
        
        # Bollinger Bands %B
        sma = Close.rolling(k).mean()
        rstd = Close.rolling(k).std()
        upper_bb = sma + 2 * rstd
        lower_bb = sma - 2 * rstd
        new_features[f'bb_pct_{k}d'] = (Close - lower_bb) / (upper_bb - lower_bb).replace(0, np.nan)
        
        # Volume Dynamics
        new_features[f'vol_z_{k}d'] = (Volume - Volume.rolling(k).mean()) / Volume.rolling(k).std().replace(0, np.nan)
        new_features[f'norm_vol_{k}d'] = Volume / Volume.rolling(k).mean().replace(0, np.nan)
        
        # VWAP Ratio
        pv = Close * Volume
        rolling_pv = pv.rolling(k).sum()
        rolling_vol = Volume.rolling(k).sum()
        vwap = rolling_pv / rolling_vol.replace(0, np.nan)
        new_features[f'vwap_ratio_{k}d'] = Close / vwap
        
        # Extremes & Ranges
        new_features[f'max_ratio_{k}d'] = Close / Close.rolling(k).max().replace(0, np.nan)
        new_features[f'min_ratio_{k}d'] = Close / Close.rolling(k).min().replace(0, np.nan)
        new_features[f'high_low_ratio_{k}d'] = High.rolling(k).max() / Low.rolling(k).min().replace(0, np.nan)
        
        # Statistical Moments
        new_features[f'skew_{k}d'] = ret1d.rolling(k).skew()
        new_features[f'kurt_{k}d'] = ret1d.rolling(k).kurt()

    # 3. Non-Window Specific (Global) Indicators
    # MACD standard (12, 26, 9)
    ema_fast = Close.ewm(span=12, adjust=False).mean()
    ema_slow = Close.ewm(span=26, adjust=False).mean()
    new_features['macd'] = ema_fast - ema_slow
    new_features['macd_signal'] = new_features['macd'].ewm(span=9, adjust=False).mean()
    new_features['macd_hist'] = new_features['macd'] - new_features['macd_signal']
    
    # MACD fast (5, 15, 5)
    ema_fast_f = Close.ewm(span=5, adjust=False).mean()
    ema_slow_f = Close.ewm(span=15, adjust=False).mean()
    new_features['macd_fast'] = ema_fast_f - ema_slow_f
    new_features['macd_fast_signal'] = new_features['macd_fast'].ewm(span=5, adjust=False).mean()
    new_features['macd_fast_hist'] = new_features['macd_fast'] - new_features['macd_fast_signal']
    
    # MACD slow (24, 52, 18)
    ema_fast_s = Close.ewm(span=24, adjust=False).mean()
    ema_slow_s = Close.ewm(span=52, adjust=False).mean()
    new_features['macd_slow'] = ema_fast_s - ema_slow_s
    new_features['macd_slow_signal'] = new_features['macd_slow'].ewm(span=18, adjust=False).mean()
    new_features['macd_slow_hist'] = new_features['macd_slow'] - new_features['macd_slow_signal']

    # 4. RSI Wilder's calculation for windows
    for k in windows:
        delta = Close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=k - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=k - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        new_features[f'rsi_{k}d'] = 100 - (100 / (1 + rs))

    # 5. Lags of basic returns
    for lag in range(1, 6):
        new_features[f'return_1d_lag{lag}'] = ret1d.shift(lag)
        
    for lag in range(1, 6):
        new_features[f'return_5d_lag{lag}'] = ret5d.shift(lag)

    # 6. Date-based Calendar Features
    if 'Date' in df.columns:
        date_col = pd.to_datetime(df['Date'])
        new_features['day_of_week'] = date_col.dt.dayofweek
        new_features['month'] = date_col.dt.month
        new_features['year'] = date_col.dt.year

    # 7. Targets Generation
    # Regression targets
    next_open = Open.shift(-1)
    future_close_1d = Close.shift(-1)
    future_close_5d = Close.shift(-5)
    
    new_features['target_reg_1d'] = (future_close_1d - next_open) / next_open
    new_features['target_reg_5d'] = (future_close_5d - next_open) / next_open
    
    # Classification targets (cast to float first, keeping NaNs as NaN)
    target_clf_1d = pd.Series(np.nan, index=df.index)
    valid_1d = new_features['target_reg_1d'].notna()
    target_clf_1d.loc[valid_1d] = (new_features['target_reg_1d'].loc[valid_1d] >= threshold_1d).astype(int)
    new_features['target_clf_1d'] = target_clf_1d.astype('Int64')
    
    target_clf_5d = pd.Series(np.nan, index=df.index)
    valid_5d = new_features['target_reg_5d'].notna()
    target_clf_5d.loc[valid_5d] = (new_features['target_reg_5d'].loc[valid_5d] >= threshold_5d).astype(int)
    new_features['target_clf_5d'] = target_clf_5d.astype('Int64')
    
    # Concatenate features to df in one go to prevent fragmentation warnings
    features_df = pd.DataFrame(new_features, index=df.index)
    
    # Drop target columns from df if they already exist, to avoid duplication
    cols_to_drop = [c for c in ['target_reg_1d', 'target_reg_5d', 'target_clf_1d', 'target_clf_5d'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        
    df = pd.concat([df, features_df], axis=1)
    return df

def generate_all_features(df, windows=None, threshold_1d=0.0, threshold_5d=0.0):
    """
    Computes high-dimensional features for all symbols in a consolidated dataframe.
    """
    if windows is None:
        windows = [3, 5, 10, 15, 20, 30, 50, 75, 100, 200]
        
    # Sort chronologically to preserve time-series ordering
    df_sorted = df.sort_values(['Symbol', 'Date']).copy()
    
    symbols = df_sorted['Symbol'].unique()
    all_dfs = []
    
    for symbol in symbols:
        df_symbol = df_sorted[df_sorted['Symbol'] == symbol]
        df_symbol_feats = calculate_symbol_features(
            df_symbol, 
            windows=windows, 
            threshold_1d=threshold_1d, 
            threshold_5d=threshold_5d
        )
        all_dfs.append(df_symbol_feats)
        
    consolidated_df = pd.concat(all_dfs, ignore_index=True)
    return consolidated_df
