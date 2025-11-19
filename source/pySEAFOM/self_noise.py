import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import windows, detrend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_enbw(window):
    return np.sum(window**2) / np.sum(window)**2

def smooth_data(x, y, window=50, log_space=True):
    """
    Smooth data using a sliding window average, with optional log-spaced bins
    
    Args:
        x: Frequency array
        y: Amplitude array
        window: Number of points to average (default 50)
        log_space: If True, use logarithmically spaced bins
    """
    if log_space:
        # Create log-spaced bins with fixed number of points
        num_bins = 100  # Adjust this value to control smoothing
        log_f_min, log_f_max = np.log10(x[1]), np.log10(x[-1])
        log_bins = np.logspace(log_f_min, log_f_max, num_bins)
        
        # Average within each bin
        smooth_y = []
        smooth_x = []
        for i in range(len(log_bins)-1):
            mask = (x >= log_bins[i]) & (x < log_bins[i+1])
            if np.any(mask):
                smooth_y.append(np.mean(y[mask]))
                smooth_x.append(np.sqrt(log_bins[i] * log_bins[i+1]))  # Geometric mean of bin edges
                
        return np.array(smooth_x), np.array(smooth_y)
    else:
        # Simple sliding window average
        kernel = np.ones(window) / window
        smooth_y = np.convolve(y, kernel, mode='valid')
        smooth_x = x[window//2:-window//2+1]  # Center-aligned x values
        return smooth_x, smooth_y
    

def calculate_asd(time_series, interrogation_rate, window_function='blackman-harris', detrend_data=True):
    n = len(time_series)
    if detrend_data:
        time_series = detrend(time_series)

    if window_function.lower() == 'blackman-harris':
        window = windows.blackmanharris(n)
    elif window_function.lower() == 'flattop':
        window = windows.flattop(n)
    else:
        window = np.ones(n)

    windowed = time_series * window
    fft_result = fft(windowed)
    freqs = fftfreq(n, 1/interrogation_rate)

    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    fft_result = fft_result[pos_mask]

    amplitude = np.abs(fft_result) / np.sum(window)
    asd = amplitude * np.sqrt(2) / np.sqrt(calculate_enbw(window) * interrogation_rate)

    return freqs, asd

def calculate_self_noise(data_sections, interrogation_rate,window_function='blackman-harris', data_type='phase'):
    """
    Calculate self-noise amplitude spectral density from DAS data sections.
    
    Computes the RMS amplitude spectral density (ASD) across multiple channels
    for each test section using FFT analysis with configurable windowing.
    
    Parameters
    ----------
    data_sections : list of ndarray
        List of 2D arrays (channels × samples) containing DAS measurements for each test section.
        Each section is analyzed independently.
    interrogation_rate : float
        Sampling frequency in Hz (e.g., 10000 for 10 kHz).
    window_function : str, optional
        FFT window function. Options: 'blackman-harris', 'flattop', 'hann', 'hamming', 'none'.
        Default is 'blackman-harris'.
    data_type : str, optional
        Input data type: 'phase' (radians), 'pε' (picostrain), 'nε' (nanostrain), 
        'rad' (radians), or 'dphase' (phase rate - will be integrated). 
        Default is 'phase'.
    
    Returns
    -------
    list of tuple
        List of (frequencies, rms_asd) tuples, one per section.
        - frequencies : ndarray, frequency bins in Hz
        - rms_asd : ndarray, RMS amplitude spectral density in the same units as input data
    
    Examples
    --------
    >>> import numpy as np
    >>> from pySEAFOM import calculate_self_noise
    >>> 
    >>> # Generate synthetic data (100 channels, 120000 samples)
    >>> data = np.random.randn(100, 120000)
    >>> sections = [data[0:50, :], data[50:100, :]]
    >>> 
    >>> # Calculate self-noise for two sections
    >>> results = calculate_self_noise(sections, interrogation_rate=10000, 
    ...                                gauge_length=10.0, data_type='pε')
    >>> freqs, asd = results[0]  # First section results
    """
    results = []
    for section_idx, section in enumerate(data_sections):
        individual_asds = []
        print(f"Processing section {section_idx + 1}/{len(data_sections)}")
        
        for ssl_idx, ssl_ts in enumerate(section):
            # Convert dphase to phase if needed
            if data_type.lower() == 'dphase':
                ssl_ts = np.cumsum(ssl_ts) #/ interrogation_rate
            
            # Calculate ASD
            freqs, asd_radians = calculate_asd(ssl_ts, interrogation_rate, window_function)

            individual_asds.append(asd_radians)
            
            # Print progress every 100 SSLs
            if (ssl_idx + 1) % 100 == 0:
                print(f"  Processed {ssl_idx + 1}/{len(section)} SSLs")
        
        individual_asds = np.array(individual_asds)
        rms_asd = np.sqrt(np.mean(individual_asds**2, axis=0))
        results.append((freqs, rms_asd))
        
    return results



def plot_combined_self_noise_db(results, title, test_sections, gauge_length, data_unit='unit', 
                                 sampling_freq=None, n_channels=None, duration=None):
    """
    Create publication-quality self-noise plots in dB scale with metadata overlay.
    
    Plots multiple test sections on the same graph with log-space smoothing,
    frequency band highlighting (seismic 1-100 Hz, acoustic 100+ Hz), and
    measurement parameter metadata box.
    
    Parameters
    ----------
    results : list of tuple
        Output from `calculate_self_noise()`: list of (frequencies, asd) tuples.
    title : str
        Plot title.
    test_sections : list of str
        Names/labels for each test section (e.g., ['Section A', 'Section B']).
    gauge_length : float
        Gauge length in meters (displayed in metadata box).
    data_unit : str, optional
        Unit label for data ('pε', 'nε', 'rad', etc.). Default is 'unit'.
    sampling_freq : float, optional
        Sampling frequency in Hz (for metadata box). Default is None.
    n_channels : int, optional
        Number of channels per section (for metadata box). Default is None.
    duration : float, optional
        Total recording duration in seconds (for metadata box). Default is None.
    
    Returns
    -------
    None
        Displays matplotlib figure. Use `plt.show()` to render or `plt.savefig()` to save.
    
    Examples
    --------
    >>> from pySEAFOM import calculate_self_noise, plot_combined_self_noise_db
    >>> import matplotlib.pyplot as plt
    >>> 
    >>> # After calculating self-noise
    >>> plot_combined_self_noise_db(
    ...     results=results,
    ...     title='DAS Self-Noise Comparison',
    ...     test_sections=['Cable Section 1', 'Cable Section 2'],
    ...     gauge_length=10.0,
    ...     org_data_unit='pε',
    ...     sampling_freq=10000,
    ...     n_channels=100,
    ...     duration=120
    ... )
    >>> plt.show()
    """
    # Create figure and main axes
    fig = plt.figure(figsize=(12, 8))
    main_ax = fig.add_subplot(111)
    colors = ['blue', 'red', 'green']
    nyquist_freq = sampling_freq / 2 if sampling_freq else None
    for i, (frequencies, self_noise) in enumerate(results):
        # Convert to picostrain then to dB

        self_noise_db = 20 * np.log10(self_noise)
        
        # Original data (lighter color)
        main_ax.plot(frequencies, self_noise_db, 
                    alpha=0.2, color=colors[i])
        
        # Smoothed data
        smooth_f, smooth_noise = smooth_data(frequencies, self_noise, 
                                           window=50, log_space=True)
        smooth_noise_db = 20 * np.log10(smooth_noise)
        main_ax.plot(smooth_f, smooth_noise_db, 
                    linewidth=2, color=colors[i], label=f'{test_sections[i]}')
    
    # Set labels and title with larger font size
    label_fontsize = 16
    title_fontsize = 16
    legend_fontsize = 16

    main_ax.set_xlabel("Frequency (Hz)", fontsize=label_fontsize)
    main_ax.set_ylabel(f"Self-Noise (dB re. 1 {data_unit}/√Hz)", fontsize=label_fontsize)
    main_ax.set_title(title, fontsize=title_fontsize)
    
    # Configure grid
    main_ax.grid(True, which='both')
    main_ax.grid(True, which='major', linestyle='-')
    main_ax.grid(True, which='minor', linestyle='--', alpha=0.5)

    # Add legend with font size
    main_ax.legend(fontsize=legend_fontsize)

    xmax_orig=nyquist_freq
   
    # Set x and y limits
    plt.xlim(0.1, xmax_orig)
    
    # Calculate y_min in dB from the last section's data (for demonstration)
    # Find minimum dB value where frequency >= 0.1 Hz
    mask = frequencies >= 0.1
    if np.any(mask):
        y_min_db = np.min(self_noise_db[mask])
        y_max_db = np.max(self_noise_db[mask])
        plt.ylim(y_min_db , y_max_db)  
    
    # Get updated limits after setting them
    xmin, xmax = main_ax.get_xlim()
    ymin, ymax = main_ax.get_ylim()
    
    # Add metadata box on the right side if metadata is provided
    if sampling_freq is not None or n_channels is not None or duration is not None:
        metadata_text = "Measurement Parameters\n" + "="*25 + "\n"
        if sampling_freq is not None:
            metadata_text += f"Sampling Freq: {sampling_freq:.1f} Hz\n"
        if gauge_length is not None:
            metadata_text += f"Gauge Length: {gauge_length:.1f} m\n"
        if n_channels is not None:
            metadata_text += f"Channels/Section: {n_channels}\n"
        if duration is not None:
            metadata_text += f"Data Duration: {duration:.1f} s\n"
        
        # Create text box with white background on the right side
        metadata_props = dict(boxstyle="round,pad=0.5", fc='white', ec="black", alpha=0.9)
        main_ax.text(0.98, 0.65, metadata_text,
                    transform=main_ax.transAxes,
                    horizontalalignment='right',
                    verticalalignment='top',
                    fontsize=16,
                    family='monospace',
                    bbox=metadata_props)
    
    # Create text box with yellow background
    bbox_props = dict(boxstyle="round,pad=0.3", fc='yellow', ec="black", alpha=0.7)
    text_str = "pySEAFOM 0.1"
    main_ax.text(xmax * 0.98, ymax * 0.98, text_str,
                horizontalalignment='right',
                verticalalignment='top',
                fontsize=20,
                fontweight='bold',
                bbox=bbox_props,
                alpha=1.0)
    
    plt.tight_layout()





def report_self_noise(results, gauge_length, test_sections=('Section 1', 'Section 2', 'Section 3'), 
                     band_frequencies=None, report_in_db=False, data_unit='unit', 
                     render_tables=False, table_figsize=(11,2.5)):
    """
    Generate formatted text report of self-noise values at key frequencies and bands.
    
    Prints self-noise values at standard frequencies (10, 100, 1000, 10000 Hz) and
    optionally calculates RMS averages over custom frequency bands.
    
    Parameters
    ----------
    results : list of tuple
        Output from `calculate_self_noise()`: list of (frequencies, asd) tuples.
    gauge_length : float
        Gauge length in meters (displayed in report header).
    test_sections : tuple of str, optional
        Names/labels for each test section. Default is ('Section 1', 'Section 2', 'Section 3').
    band_frequencies : list of tuple, optional
        List of (low_freq, high_freq) tuples defining frequency bands for averaging.
        Example: [(1, 100), (100, 1000), (1000, 5000)]. Default is None.
    report_in_db : bool, optional
        If True, report values in dB scale. If False, use linear units. Default is False.
    data_unit : str, optional
        Unit label for data ('pε', 'nε', 'rad', etc.). Default is 'unit'.
    render_tables : bool, optional
        If True, renders matplotlib tables summarizing the reported values for each section.
    table_figsize : tuple, optional
        Figure size passed to matplotlib when `render_tables` is True. Default is (7, 3.5).
    
    Returns
    -------
    None
        Prints formatted report to stdout and optionally renders matplotlib tables.
    
    Examples
    --------
    >>> from pySEAFOM import calculate_self_noise, report_self_noise
    >>> 
    >>> # After calculating self-noise
    >>> report_self_noise(
    ...     results=results,
    ...     gauge_length=10.0,
    ...     test_sections=['Section A', 'Section B'],
    ...     band_frequencies=[(1, 100), (100, 1000)],
    ...     report_in_db=False,
    ...     org_data_unit='pε'
    ... )
    """
    # Store values for averaging
    freq_values = {freq: [] for freq in [10, 100, 1000, 10000]}
    band_values = {}
    if band_frequencies:
        band_values = {f"{low}-{high}": [] for low, high in band_frequencies}

    print(f"\npySEAFOM 0.1 - Self-Noise Test Report (Gauge Length: {gauge_length} m)")
    print("----------------------------------------------------------")
    
    # Process each section and collect values for averaging
    for i, (frequencies, self_noise) in enumerate(results):
        
        print(f"\n{test_sections[i]}:")
        print(f"  Number of frequency points: {len(frequencies)}")
        if report_in_db:
            print(f"  Self-noise values (dB re 1 {data_unit}/rtHz) at key frequencies:")
            print(f"    Frequency (Hz) | Self-Noise (dB)")
        else:
            print(f"  Self-noise values ({data_unit} per root Hz) at key frequencies:")
            print(f"    Frequency (Hz) | Self-Noise ({data_unit}/rtHz)")
        print(f"    --------------- | --------------------------")

        freq_table_rows = []

        for freq in [10, 100, 1000, 10000]:
            idx = np.argmin(np.abs(frequencies - freq))
            val = self_noise[idx]
            if report_in_db:
                val = 20 * np.log10(val)
            print(f"    {frequencies[idx]:<14.2f} | {val:<25.2e}")
            freq_values[freq].append(val)
            display_val = f"{val:.2f}" if report_in_db else f"{val:.2e}"
            freq_table_rows.append([f"{frequencies[idx]:.2f}", display_val])

        band_table_rows = []

        if band_frequencies:
            print("\n  Average self-noise in specified frequency bands:")
            print(f"    Frequency Band (Hz) | Avg Self-Noise (dB) | Avg Self-Noise ({data_unit}/√Hz)")
            print(f"    --------------- | ----------------- | -------------------")
            for band in band_frequencies:
                low, high = band
                band_mask = (frequencies >= low) & (frequencies <= high)
                if np.any(band_mask):
                    band_rms = np.sqrt(np.mean(self_noise[band_mask] ** 2))
                    band_rms_db = 20 * np.log10(band_rms)
                    print(f"    [{low:6.1f}, {high:6.1f}]   | {band_rms_db:12.2f} dB | {band_rms:12.2e}")
                    band_values[f"{low}-{high}"].append(band_rms)
                    band_table_rows.append([
                        f"[{low:.1f}, {high:.1f}]",
                        f"{band_rms_db:.2f}",
                        f"{band_rms:.2e}"
                    ])
                else:
                    print(f"    [{low:6.1f}, {high:6.1f}]   | {'N/A':>12}    | {'N/A':>12}")
                    band_table_rows.append([
                        f"[{low:.1f}, {high:.1f}]",
                        "N/A",
                        "N/A"
                    ])

        if render_tables:
            n_tables = 1 + bool(band_frequencies)
            fig, axs = plt.subplots(1, n_tables, figsize=table_figsize)
            if n_tables == 1:
                axs = [axs]
            fig.suptitle(f"{test_sections[i]} — Self-noise Report", fontsize=14, fontweight='bold')
            axs[0].axis('off')
            freq_col_labels = [
                "Frequency (Hz)",
                "Self-Noise (dB)" if report_in_db else f"Self-Noise ({data_unit}/√Hz)"
            ]
            freq_table = axs[0].table(cellText=freq_table_rows, colLabels=freq_col_labels, loc='center')
            freq_table.auto_set_font_size(False)
            freq_table.set_fontsize(10)
            freq_table.scale(1, 1.3)
            if band_frequencies:
                axs[1].axis('off')
                band_col_labels = [
                    "Band (Hz)",
                    "Avg Self-Noise (dB)",
                    f"Avg Self-Noise ({data_unit}/√Hz)"
                ]
                band_table = axs[1].table(cellText=band_table_rows, colLabels=band_col_labels, loc='center')
                band_table.auto_set_font_size(False)
                band_table.set_fontsize(10)
                band_table.scale(1, 1.3)
            plt.tight_layout()
