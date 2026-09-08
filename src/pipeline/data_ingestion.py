import io
import os
import sys
import argparse
import time
import random
import threading
import warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress all deprecation, future, and third-party library warnings
warnings.filterwarnings("ignore")

# Thread lock to avoid yfinance internal cache concurrency issues (e.g. dictionary changed size)
download_lock = threading.Lock()

# Default configuration
CONSTITUENTS_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = datetime.today().strftime('%Y-%m-%d')
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

def fetch_nifty500_constituents(verbose=False):
    """
    Downloads the official list of Nifty 500 constituents from the NSE website.
    Falls back to local CSV file if the request fails or times out.
    """
    if verbose:
        print(f"Fetching NIFTY 500 constituents from {CONSTITUENTS_URL}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(CONSTITUENTS_URL, headers=headers, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = [col.strip() for col in df.columns]
            if verbose:
                print(f"Successfully retrieved {len(df)} constituents from online source.")
            return df
        else:
            if verbose:
                print(f"Failed to fetch online constituents. HTTP status: {response.status_code}")
    except Exception as e:
        if verbose:
            print(f"Error fetching online constituents: {str(e)}")
        
    # Local fallback
    local_path = os.path.join(DATA_DIR, "nifty500_constituents.csv")
    if os.path.exists(local_path):
        if verbose:
            print(f"Using cached constituents list from: {local_path}")
        df = pd.read_csv(local_path)
        df.columns = [col.strip() for col in df.columns]
        return df
    else:
        raise Exception("Failed to fetch constituents online, and no local fallback exists.")

def download_ticker_data(ticker, start_date, end_date):
    """
    Downloads historical daily data for a single ticker from Yahoo Finance.
    Includes a robust retry mechanism to handle rate limits and concurrency issues.
    """
    yf_symbol = f"{ticker}.NS"
    max_retries = 5
    backoff_factor = 1.5
    
    for attempt in range(max_retries):
        try:
            # Serialize the yfinance call to prevent internal scraper cache race conditions
            with download_lock:
                data = yf.download(
                    yf_symbol, 
                    start=start_date, 
                    end=end_date, 
                    progress=False, 
                    group_by='ticker',
                    auto_adjust=True
                )
                
            # If rate limited or empty, check if we should retry
            if data.empty:
                if attempt < max_retries - 1:
                    sleep_time = (backoff_factor ** attempt) + random.uniform(0.5, 2.0)
                    time.sleep(sleep_time)
                    continue
                return ticker, None
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Standardise columns (flatten MultiIndex if any)
            if isinstance(data.columns, pd.MultiIndex):
                ohlcv_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'}
                level_to_keep = 0
                for i in range(data.columns.nlevels):
                    if any(col in ohlcv_cols for col in data.columns.get_level_values(i)):
                        level_to_keep = i
                        break
                levels_to_drop = [i for i in range(data.columns.nlevels) if i != level_to_keep]
                for lvl in sorted(levels_to_drop, reverse=True):
                    data.columns = data.columns.droplevel(lvl)
                
            # Ensure the date column is named 'Date'
            data.rename(columns={'index': 'Date', 'Date': 'Date', '': 'Date'}, inplace=True)
            if 'Date' not in data.columns and data.columns[0] != 'Date':
                cols = list(data.columns)
                cols[0] = 'Date'
                data.columns = cols
                
            data['Symbol'] = ticker
            return ticker, data
            
        except RuntimeError as e:
            # Catch "dictionary changed size during iteration" or other concurrency errors
            if attempt < max_retries - 1:
                sleep_time = 0.5 + random.uniform(0.1, 0.5)
                time.sleep(sleep_time)
                continue
            print(f"RuntimeError downloading {ticker} (attempt {attempt+1}): {str(e)}")
            return ticker, None
            
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = (backoff_factor ** attempt) + random.uniform(0.5, 2.0)
                time.sleep(sleep_time)
                continue
            print(f"Error downloading {ticker} (attempt {attempt+1}): {str(e)}")
            return ticker, None
            
    return ticker, None

def add_target_columns(df, horizon=1, threshold=0.0):
    """
    Adds regression and classification target columns to the dataframe.
    Calculates forward returns grouping by Symbol to prevent stock boundary bleed.
    
    The target regression column is the forward return from T+1 Open to T+horizon Close:
        (Close_(t+horizon) - Open_(t+1)) / Open_(t+1)
        
    The target classification column is 1 if target regression return >= threshold, else 0.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Consolidated dataframe containing Symbol, Open, Close.
    horizon : int
        Chronological horizon in days (e.g., 1 or 5).
    threshold : float
        Customisable return threshold above which class is 1, else 0.
        
    Returns:
    --------
    pd.DataFrame
        Dataframe copy with added target columns.
    """
    # Ensure it's sorted chronologically per Symbol
    df_sorted = df.sort_values(['Symbol', 'Date']).copy()
    
    # Calculate group-by shifts
    next_open = df_sorted.groupby('Symbol')['Open'].shift(-1)
    future_close = df_sorted.groupby('Symbol')['Close'].shift(-horizon)
    
    reg_col = f'target_reg_{horizon}d'
    clf_col = f'target_clf_{horizon}d'
    
    df_sorted[reg_col] = (future_close - next_open) / next_open
    
    # Binary classification target (0 or 1), keeping NaNs as NaN
    df_sorted[clf_col] = np.nan
    valid_mask = df_sorted[reg_col].notna()
    df_sorted.loc[valid_mask, clf_col] = (df_sorted.loc[valid_mask, reg_col] >= threshold).astype(int)
    
    return df_sorted

def ingest_nifty500_data(start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE, max_workers=10, constituents_df=None, verbose=False):
    """
    Orchestrates the download of historical data for all constituents.
    """
    # 1. Ensure data directories exist
    raw_data_dir = os.path.join(DATA_DIR, "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    # 2. Get constituents if not provided
    if constituents_df is None:
        constituents_df = fetch_nifty500_constituents(verbose=verbose)
        # Save the constituents list for reference
        constituents_path = os.path.join(DATA_DIR, "nifty500_constituents.csv")
        constituents_df.to_csv(constituents_path, index=False)
        if verbose:
            print(f"Saved constituents list to {constituents_path}")
    
    tickers = constituents_df['Symbol'].tolist()
    
    # 3. Download daily historical data concurrently
    all_data = []
    failed_tickers = []
    
    if verbose:
        print(f"Downloading daily data from {start_date} to {end_date} using {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_ticker_data, ticker, start_date, end_date): ticker 
            for ticker in tickers
        }
        
        total_tickers = len(tickers)
        completed = 0
        
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                ticker, df = future.result()
                if df is not None and not df.empty:
                    all_data.append(df)
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                # Use a newline to print exceptions clearly so they don't break the progress bar format
                if verbose:
                    sys.stdout.write(f"\nException generated for {ticker}: {str(e)}\n")
                failed_tickers.append(ticker)
                
            completed += 1
            percent = 100.0 * completed / total_tickers
            filled_length = int(40 * completed // total_tickers)
            bar = '█' * filled_length + '-' * (40 - filled_length)
            sys.stdout.write(f"\r[{bar}] {percent:.1f}% | {completed}/{total_tickers} stocks processed")
            sys.stdout.flush()
            
        sys.stdout.write("\n")
        sys.stdout.flush()
                
    # 4. Consolidate and Save
    if all_data:
        consolidated_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure correct column naming convention
        consolidated_df.columns = [col.replace(' ', '_').strip() for col in consolidated_df.columns]
        
        # Calculate target columns (1d and 5d with threshold 0.0)
        try:
            if verbose:
                print("\nCalculating 1-day and 5-day forward target columns...")
            consolidated_df = add_target_columns(consolidated_df, horizon=1, threshold=0.0)
            consolidated_df = add_target_columns(consolidated_df, horizon=5, threshold=0.0)
            if verbose:
                print("Successfully calculated targets.")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to add target columns: {str(e)}")
            
        # Save as Parquet (efficient lookup, schema preservation)
        parquet_path = os.path.join(raw_data_dir, "nifty500_daily.parquet")
        consolidated_df.to_parquet(parquet_path, index=False)
        
        # Save as CSV as a fallback
        csv_path = os.path.join(raw_data_dir, "nifty500_daily.csv")
        consolidated_df.to_csv(csv_path, index=False)
        
        # Always print shape and basic stats when ingestion is complete
        print("\n=== Ingested Dataset Summary ===")
        print(f"Shape: {consolidated_df.shape}")
        print(f"Unique Symbols Ingested: {consolidated_df['Symbol'].nunique()}")
        if 'Date' in consolidated_df.columns:
            print(f"Date Range: {consolidated_df['Date'].min()} to {consolidated_df['Date'].max()}")
        
        print("\nBasic Statistics:")
        desc_cols = [c for c in consolidated_df.columns if c not in ['Symbol', 'Date']]
        print(consolidated_df[desc_cols].describe().round(4).to_string())
        
        if verbose:
            print("\n=== Ingestion Paths ===")
            print(f"Successfully ingested data for {len(tickers) - len(failed_tickers)} / {len(tickers)} stocks.")
            print(f"Parquet Saved to: {parquet_path}")
            print(f"CSV Saved to: {csv_path}")
            if failed_tickers:
                print(f"Failed Tickers: {failed_tickers}")
    else:
        print("No data was successfully downloaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NIFTY 500 Historical Data")
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=DEFAULT_END_DATE, help="End date (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=10, help="Number of download threads")
    
    args = parser.parse_args()
    
    print("--- Diagnostics Mode: Pulling NIFTY 500 Constituents ---")
    constituents_df = fetch_nifty500_constituents(verbose=True)
    constituents_path = os.path.join(DATA_DIR, "nifty500_constituents.csv")
    has_changes = True
    
    if os.path.exists(constituents_path):
        try:
            existing_df = pd.read_csv(constituents_path)
            # Standardise column names
            existing_df.columns = [col.strip() for col in existing_df.columns]
            
            existing_symbols = set(existing_df['Symbol'].dropna().tolist())
            new_symbols = set(constituents_df['Symbol'].dropna().tolist())
            
            added = new_symbols - existing_symbols
            removed = existing_symbols - new_symbols
            
            if not added and not removed:
                print(f"Diagnostics: Local constituents list is up-to-date. No changes detected compared to {constituents_path}.")
                has_changes = False
            else:
                print("Diagnostics: Changes detected in NIFTY 500 constituents!")
                if added:
                    print(f"  Added symbols ({len(added)}): {sorted(list(added))}")
                if removed:
                    print(f"  Removed symbols ({len(removed)}): {sorted(list(removed))}")
        except Exception as e:
            print(f"Diagnostics: Error during comparison ({str(e)}). Forcing rewrite.")
            
    if has_changes:
        os.makedirs(os.path.dirname(constituents_path), exist_ok=True)
        constituents_df.to_csv(constituents_path, index=False)
        print(f"Diagnostics: Saved updated constituents list to {constituents_path}")
        
    print("\n--- Starting daily data ingestion ---")
    # Run Ingestion using the fetched constituents in verbose mode
    ingest_nifty500_data(
        start_date=args.start_date, 
        end_date=args.end_date, 
        max_workers=args.workers, 
        constituents_df=constituents_df,
        verbose=True
    )
