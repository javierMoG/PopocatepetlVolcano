import numpy as np
import pandas as pd
from obspy import read
import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters

# Load seismic data with ObsPy
def load_and_preprocess_seismic_data(file_path, freqmin=1.0, freqmax=16.0):
    """Load and preprocess seismic data"""
    stream = read(file_path)
    processed = stream.copy()
    processed.detrend('linear')
    processed.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
    return processed

# Convert ObsPy trace to tsfresh compatible format
def prepare_data_for_tsfresh(trace, window_size=600, overlap=0.5):
    """
    Convert ObsPy trace to tsfresh compatible format with windowing
    
    Parameters:
    - trace: ObsPy Trace object
    - window_size: window size in seconds
    - overlap: overlap between consecutive windows (0-1)
    
    Returns:
    - DataFrame in tsfresh format
    """
    # Convert window_size from seconds to samples
    window_samples = int(window_size * trace.stats.sampling_rate)
    step = int(window_samples * (1 - overlap))
    
    # Create lists to store data
    all_data = []
    
    # Process each window
    for i in range(0, len(trace.data) - window_samples, step):
        window = trace.data[i:i+window_samples]
        window_start_time = trace.stats.starttime + i/trace.stats.sampling_rate
        window_id = window_start_time.datetime.strftime("%Y%m%d%H%M%S")
        
        # Create a DataFrame for this window
        df_window = pd.DataFrame({
            'id': window_id,
            'time': range(len(window)),
            'value': window
        })
        
        all_data.append(df_window)
    
    # Concatenate all windows
    return pd.concat(all_data, ignore_index=True)

# Extract relevant features using tsfresh
def extract_volcanic_features(df_tsfresh):
    """
    Extract features similar to the paper using tsfresh
    
    Parameters:
    - df_tsfresh: DataFrame in tsfresh format
    
    Returns:
    - DataFrame with extracted features
    """
    # Define specific features to extract
    # These correspond roughly to Energy, Shannon Entropy, Kurtosis, and Frequency Index
    fc_parameters = {
        # Energy-related features
        "abs_energy": None,  # Sum of squared values (energy)
        "root_mean_square": None,  # RMS of signal (related to energy)
        
        # Entropy-related features
        "sample_entropy": None,  # Sample entropy (similar to Shannon entropy)
        "approximate_entropy": [{'m': 2, 'r': 0.1}, {'m': 2, 'r': 0.3}],  # Approximate entropy
        
        # Kurtosis and distribution
        "kurtosis": None,  # Directly corresponds to kurtosis in the paper
        "skewness": None,  # Additional distribution characteristic
        
        # Frequency-related features
        "fft_aggregated": [{'aggtype': 'centroid'}],  # Spectral centroid (related to frequency index)
        "fft_aggregated": [{'aggtype': 'variance'}],  # Spectral variance
        "spkt_welch_density": [{'coeff': 0}, {'coeff': 1}, {'coeff': 2}],  # Power spectral density
        
        # Additional useful features
        "agg_linear_trend": [{'attr': 'slope', 'chunk_len': 50, 'f_agg': 'mean'}],  # For trend detection
        "change_quantiles": [{'ql': 0.1, 'qh': 0.9, 'isabs': True, 'f_agg': 'mean'}]  # For change detection
    }
    
    # Extract features
    features = extract_features(df_tsfresh, 
                               column_id='id', 
                               column_sort='time',
                               column_value='value', 
                               default_fc_parameters=fc_parameters,
                               n_jobs=4)  # Parallel processing
    
    return features

# Calculate custom frequency index (similar to paper)
def add_custom_frequency_index(df_features, df_tsfresh):
    """
    Add custom frequency index feature similar to the paper's definition
    
    Parameters:
    - df_features: DataFrame with tsfresh features
    - df_tsfresh: Original tsfresh format DataFrame
    
    Returns:
    - DataFrame with added frequency index
    """
    unique_ids = df_tsfresh['id'].unique()
    freq_indices = []
    
    for window_id in unique_ids:
        # Get window data
        window = df_tsfresh[df_tsfresh['id'] == window_id]['value'].values
        
        # Calculate FFT
        fft_values = np.abs(np.fft.rfft(window))
        freqs = np.fft.rfftfreq(len(window))
        
        # Define low and high frequency bands (as per paper)
        # Convert from normalized frequencies to indices
        low_freq_mask = (freqs >= 1/len(window)) & (freqs <= 5.5/len(window))
        high_freq_mask = (freqs >= 6/len(window)) & (freqs <= 16/len(window))
        
        # Calculate energy in each band
        low_energy = np.sum(fft_values[low_freq_mask]**2)
        high_energy = np.sum(fft_values[high_freq_mask]**2)
        
        # Calculate frequency index
        if (low_energy + high_energy) > 0:
            freq_index = low_energy / (low_energy + high_energy)
        else:
            freq_index = 0.5
        
        freq_indices.append(freq_index)
    
    # Add to features DataFrame
    df_features['custom_frequency_index'] = freq_indices
    
    return df_features

# Complete workflow
def analyze_seismic_with_tsfresh(file_path):
    """
    Complete workflow for extracting volcanic features using tsfresh
    
    Parameters:
    - file_path: Path to seismic data file
    
    Returns:
    - DataFrame with extracted features
    """
    # Load and preprocess data
    stream = load_and_preprocess_seismic_data(file_path)
    trace = stream[0]  # Use first trace for simplicity
    
    # Prepare data for tsfresh
    df_tsfresh = prepare_data_for_tsfresh(trace)
    
    # Extract features
    features = extract_volcanic_features(df_tsfresh)
    
    # Add custom frequency index
    features = add_custom_frequency_index(features, df_tsfresh)
    
    # Add first and second derivatives
    # This matches the paper's approach of using 12 parameters
    for column in features.columns:
        features[f"{column}_d1"] = features[column].diff()
        features[f"{column}_d2"] = features[f"{column}_d1"].diff()
    
    # Handle NaN values from differentiation
    features = features.fillna(0)
    
    return features