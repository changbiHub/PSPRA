import numpy as np
import pandas as pd

import utls
from scipy.interpolate import interp1d

def preprocessor(data_df, mode='univariate', max_gap=21, interp_kind='linear'):
    """
    Unified preprocessor for both univariate and multivariate data.
    
    Parameters:
    -----------
    data_df : pd.DataFrame
        Input data with 'site', 'JD' columns and target values
    mode : str, default 'univariate'
        'univariate' for single variable processing, 'multivariate' for multiple toxin columns
    max_gap : int, default 21
        Maximum gap allowed in data
    interp_kind : str, default 'linear'
        Interpolation method for scipy.interpolate.interp1d
        
    Returns:
    --------
    For univariate mode:
        y_smoothed : np.array, target_df : pd.DataFrame, history_ls : list
    For multivariate mode:
        history_ls : list, target_df : pd.DataFrame, TI : np.array
    """
    sites = data_df['site'].unique()
    X_query = np.arange(-70, -7)    
    # Mode-specific configuration
    if mode == 'univariate':
        baseline_col = 'value_valeur'
        # target_col = 'value_log'
        target_col = 'value_valeur'
    elif mode == 'multivariate':
        baseline_col = 'PSP_total'
        target_col = None  # Multiple columns
    else:
        raise ValueError("Mode must be 'univariate' or 'multivariate'")
    
    baseline_ = data_df[baseline_col].min()
    results = []
    
    # Group by site once to avoid repeated filtering
    grouped = data_df.groupby('site')
    
    for site in sites:
        data_s = grouped.get_group(site).sort_values('JD').reset_index(drop=True)
        
        for idx, row in data_s.iterrows():
            target_time = row['JD']
            window_lower, window_upper = target_time - 70, target_time - 7
            
            # Get history data
            history = data_s[data_s['JD'].between(window_lower, window_upper)].copy()
            
            if len(history) == 0:
                continue
                
            # Calculate bounds
            lower_bound = history['JD'].min() - max_gap
            upper_bound = min(history['JD'].max() + max_gap, target_time - 1)
            
            # Check all conditions efficiently
            conditions = [
                history['JD'].diff().max() < max_gap,
                lower_bound <= window_lower,
                upper_bound >= window_upper,
                len(data_s[data_s['JD'].between(window_upper, upper_bound)]) > 0,
                len(data_s[data_s['JD'].between(lower_bound, window_lower)]) > 0,
                history[baseline_col].max() > baseline_
            ]
            
            if not all(conditions):
                continue
                
            # Get boundary points
            upper_pt = data_s[data_s['JD'].between(window_upper, upper_bound)].iloc[[0]]
            lower_pt = data_s[data_s['JD'].between(lower_bound, window_lower)].iloc[[-1]]
            
            # Extend history
            history_extended = pd.concat([lower_pt, history, upper_pt]).sort_values('JD')
            history_extended['relativeDay'] = history_extended['JD'] - target_time
            history_extended.drop_duplicates(subset=['relativeDay'], keep='first', inplace=True)
            
            # Mode-specific interpolation
            rday = history_extended['relativeDay'].values
            
            if mode == 'univariate':
                y = history_extended[target_col].values
                interper = interp1d(rday, y, kind=interp_kind)
                y_pred = interper(X_query)
                
                results.append({
                    'target': row,
                    'history': history_extended,
                    'smoothed': y_pred
                })
            else:  # multivariate
                reading = history_extended.iloc[:, 2:14].values  # Assuming columns 2-13 are toxin data
                
                # Interpolate all toxin columns simultaneously
                toxin_image = np.array([
                    interp1d(rday, reading[:, i], kind=interp_kind)(X_query) 
                    for i in range(reading.shape[1])
                ]).T
                
                results.append({
                    'target': row,
                    'history': history_extended,
                    'toxin_image': toxin_image
                })
    
    # Return mode-specific results
    if not results:
        if mode == 'univariate':
            return np.array([]), pd.DataFrame(), []
        else:
            return [], pd.DataFrame(), np.array([])
    
    target_df = pd.DataFrame([r['target'] for r in results])
    history_ls = [r['history'] for r in results]
    
    if mode == 'univariate':
        y_smoothed = np.array([r['smoothed'] for r in results])
        return y_smoothed, target_df, history_ls
    else:  # multivariate
        TI = np.array([r['toxin_image'] for r in results])
        return history_ls, target_df, TI


def preprocessor_univariate(data_df, max_gap=21, interp_kind='linear'):
    """Legacy wrapper for univariate preprocessing."""
    return preprocessor(data_df, mode='univariate', max_gap=max_gap, interp_kind=interp_kind)


def preprocessor_multivariate(data_df, max_gap=21, interp_kind='linear'):
    """Legacy wrapper for multivariate preprocessing."""
    return preprocessor(data_df, mode='multivariate', max_gap=max_gap, interp_kind=interp_kind)