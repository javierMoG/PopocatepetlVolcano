import numpy as np
import pandas as pd
from obspy import read
from scipy.stats import norm
import matplotlib.pyplot as plt

def load_and_preprocess_seismic_data(file_path, freqmin=0.7, freqmax=30, low_freqmin=1, low_freqmax=5.5, high_freqmin=6, high_freqmax=16):
    """
    Load seismic data from a file and apply bandpass filtering to obtain full, low-frequency, and high-frequency signals.

    Parameters:
    - file_path: str
        Path to the seismic data file.
    - freqmin: float, optional
        Minimum frequency for the full bandpass filter (default: 0.7 Hz).
    - freqmax: float, optional
        Maximum frequency for the full bandpass filter (default: 30 Hz).
    - low_freqmin: float, optional
        Minimum frequency for the low-frequency bandpass filter (default: 1 Hz).
    - low_freqmax: float, optional
        Maximum frequency for the low-frequency bandpass filter (default: 5.5 Hz).
    - high_freqmin: float, optional
        Minimum frequency for the high-frequency bandpass filter (default: 6 Hz).
    - high_freqmax: float, optional
        Maximum frequency for the high-frequency bandpass filter (default: 16 Hz).

    Returns:
    - list of ObsPy Stream objects:
        [filtered_signal, s_low, s_high], where
        filtered_signal: full bandpass filtered signal,
        s_low: low-frequency bandpass filtered signal,
        s_high: high-frequency bandpass filtered signal.
    """ 
    # Load the seismic data
    stream = read(file_path)
    # Detrend the data to remove linear trends
    stream.detrend('linear')
    # Apply bandpass filters for full, low, and high frequencies
    signal = stream.copy()
    signal.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
    s_low = stream.copy()
    s_low.filter('bandpass', freqmin=low_freqmin, freqmax=low_freqmax)
    s_high = stream.copy()
    s_high.filter('bandpass', freqmin=high_freqmin, freqmax=high_freqmax)
    return [signal, s_low, s_high]

def plot_signal(tr):
    """
    Plot the waveform and spectrogram (with color bar) of a seismic trace.

    Parameters:
    - tr: ObsPy Trace object
        Seismic trace to plot.

    Displays:
    - A figure with three axes:
        - Top: Waveform (amplitude vs. time)
        - Middle: Spectrogram (logarithmic scale, dB)
        - Right: Color bar for spectrogram
    """
    sps = int(tr.stats.sampling_rate)
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.5])  # waveform
    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.5], sharex=ax1)  # spectrogram
    ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.5])  # colorbar

    # Time vector for waveform
    t = np.arange(tr.stats.npts) / tr.stats.sampling_rate
    ax1.plot(t, tr.copy().data, 'k')
    ax1.set_ylabel('Amplitude')

    # Spectrogram
    tr.spectrogram(wlen=.1*sps, per_lap=0.90, dbscale=True, log=True, axes=ax2)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_ylim((0.1, 50))

    # Set color limits and add colorbar
    ax2.collections[0].set_clim(vmin=-10, vmax=100)
    mappable = ax2.collections[0]
    cb = plt.colorbar(mappable=mappable, cax=ax3, ticks=np.arange(-10, 100, 10))
    cb.set_label('Power (dB/Hz)')

    ax1.set_title('Waveform and Spectrogram')
    ax2.set_xlabel('Time (s)')
    plt.show()

def entropy(s):
    """
    Calculate the Shannon entropy of a signal assuming a normal distribution.

    Parameters:
    - s: array-like
        Input signal data (e.g., seismic amplitudes).

    Returns:
    - shannon_entropy: float
        Shannon entropy value of the input signal.
    """
    # Fit a normal distribution to the signal data
    (mu, sigma) = norm.fit(s)
    # Sort the signal data to compute the PDF
    Sor = np.sort(s)
    # Calculate the probability density function (PDF) of the normal distribution
    y = norm.pdf(Sor, mu, sigma)
    # Avoid log(0) by adding a small value to the PDF
    y_adjusted = y + np.finfo(float).eps
    # Calculate the Shannon entropy
    shannon_entropy = -np.sum(y_adjusted * np.log2(y_adjusted))
    return shannon_entropy

def freq_index(s_low, s_high, samples):
    """
    Calculate the frequency index as the log-ratio of high-frequency to low-frequency energy.

    Parameters:
    - s_low: array-like
        Signal data for the low-frequency band.
    - s_high: array-like
        Signal data for the high-frequency band.
    - samples: int
        Number of samples in each window.

    Returns:
    - freq_index: float
        Logarithmic ratio of high-frequency to low-frequency energy.
    """
    ener_low = np.sum(s_low**2) / samples
    ener_high = np.sum(s_high**2) / samples

    # Avoid division by zero by adding a small value to the denominator
    freq_index = np.log10(ener_high / (ener_low + np.finfo(float).eps))
    return freq_index

def calculate_features(trace, trace_low, trace_high, window_size_seconds=600, overlap=0.5):
    """
    Calculate multiple statistical features from seismic signals using sliding windows with overlap.

    Parameters:
    - trace: ObsPy Trace object
        Full bandpass filtered seismic trace.
    - trace_low: ObsPy Trace object
        Low-frequency bandpass filtered seismic trace.
    - trace_high: ObsPy Trace object
        High-frequency bandpass filtered seismic trace.
    - window_size_seconds: int, optional
        Window size in seconds (default: 600 = 10 minutes).
    - overlap: float, optional
        Overlap fraction between windows (default: 0.5 = 50%).

    Returns:
    - results_df: pandas.DataFrame
        DataFrame containing timestamps and calculated features for each window:
        - timestamp: Center time of each window.
        - energy: Mean squared amplitude of the window.
        - kurtosis: Kurtosis (fourth standardized moment) of the window.
        - frequency_index: Logarithmic ratio of high-frequency to low-frequency energy.
        - entropy: Shannon entropy of the window.
    """
    # Get sampling rate and calculate window size in samples
    sampling_rate = trace.stats.sampling_rate
    window_samples = int(window_size_seconds * sampling_rate)
    
    # Calculate step size (distance between window starts)
    step_samples = int(window_samples * (1 - overlap))
    
    # Get the signal data
    signal = trace.data
    signal_low = trace_low.data
    signal_high = trace_high.data
    
    # Initialize lists to store results
    timestamps = []
    energy_values = []
    kurtosis_values = []
    freq_index_values = []
    entropy_values = []
    
    # Slide the window through the signal
    for i in range(0, len(signal) - window_samples + 1, step_samples):
        # Extract window
        window = signal[i:i + window_samples]
        window_low = signal_low[i:i + window_samples]
        window_high = signal_high[i:i + window_samples]
        
        # Calculate timestamp for the center of the window
        window_center_sample = i + window_samples // 2
        timestamp = trace.stats.starttime + window_center_sample / sampling_rate
        
        # Calculate energy: mean of squared amplitudes
        energy = np.sum(window**2)

        # Calculate kurtosis: fourth standardized moment
        if np.std(window) == 0:
            kurtosis = 0
        else:
            kurtosis = (np.sum(((window - np.mean(window))/(np.std(window)))**4))/window_samples

        # Calculate frequency index: logarithmic ratio of high-frequency to low-frequency energy
        freq_index_value = freq_index(window_low, window_high, window_samples)

        # Calculate entropy: Shannon entropy of the window
        entropy_value = entropy(window)
        
        # Store results
        timestamps.append(timestamp.datetime)
        energy_values.append(energy)
        kurtosis_values.append(kurtosis)
        freq_index_values.append(freq_index_value)
        entropy_values.append(entropy_value)
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'energy': energy_values,
        'kurtosis': kurtosis_values,
        'frequency_index': freq_index_values,
        'entropy': entropy_values
    })
    
    return results_df
