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

def calculate_self_noise(data_sections, interrogation_rate, gauge_length, wavelength=1550e-9, 
                        refractive_index=1.4682, photoelastic=0.78, window_function='blackman-harris',
                        data_type='phase'):
    """
    Calculate self-noise from DAS data sections.
    
    Args:
        data_sections: List of 2D arrays containing sensor data
        interrogation_rate: Sampling rate in Hz
        gauge_length: Gauge length in meters
        wavelength: Laser wavelength in meters (default: 1550e-9)
        refractive_index: Fiber refractive index (default: 1.4682)
        photoelastic: Photoelastic coefficient (default: 0.78)
        window_function: Window function for FFT (default: 'blackman-harris')
        data_type: 'phase' or 'dphase' - if 'dphase', will integrate data (default: 'phase')
    
    Returns:
        List of tuples (frequencies, RMS ASD)
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



def plot_combined_self_noise_db(results, title, test_sections, gauge_length, org_data_unit='unit', 
                                 sampling_freq=None, n_channels=None, duration=None):
    """
    Plot all sections on the same graph in dB relative to picostrain/√Hz
    
    Args:
        results: List of (frequencies, self_noise) tuples
        title: Plot title
        test_sections: Names of test sections
        gauge_length: Gauge length in meters
        org_data_unit: 'nano' or 'pico' - original data unit
        sampling_freq: Sampling frequency in Hz (optional, for metadata box)
        n_channels: Number of channels per section (optional, for metadata box)
        duration: Total duration of data in seconds (optional, for metadata box)
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
    main_ax.set_ylabel(f"Self-Noise (dB re. 1 {org_data_unit}/√Hz)", fontsize=label_fontsize)
    main_ax.set_title(title, fontsize=title_fontsize)
    
    # Configure grid
    main_ax.grid(True, which='both')
    main_ax.grid(True, which='major', linestyle='-')
    main_ax.grid(True, which='minor', linestyle='--', alpha=0.5)

    # Add typical frequency bands
    main_ax.axvspan(1, 100, alpha=0.3, color='lightgreen', label='Seismic Band')
    main_ax.axvspan(100, max(frequencies), alpha=0.3, color='lightblue', label='Acoustic Band')
    
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
                    fontsize=12,
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





def report_self_noise(results, gauge_length, test_sections=('Section 1', 'Section 2', 'Section 3'), band_frequencies=None, report_in_db=False,org_data_unit='unit'):
    """
    Report self-noise measurements in variable units.
    
    Args:
        results: List of frequency-magnitude pairs for each section
        gauge_length: Gauge length in meters
        test_sections: Names of test sections
        plot: Whether to plot results
        band_frequencies: List of (low, high) frequency bands to analyze
        report_in_db: Whether to report values in dB
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
            print(f"  Self-noise values (dB re 1 {org_data_unit}/rtHz) at key frequencies:")
            print(f"    Frequency (Hz) | Self-Noise (dB)")
        else:
            print(f"  Self-noise values ({org_data_unit} per root Hz) at key frequencies:")
            print(f"    Frequency (Hz) | Self-Noise ({org_data_unit}/rtHz)")
        print(f"    --------------- | --------------------------")

        for freq in [10, 100, 1000, 10000]:
            idx = np.argmin(np.abs(frequencies - freq))
            val = self_noise[idx]
            if report_in_db:
                val = 20 * np.log10(val)
            print(f"    {frequencies[idx]:<14.2f} | {val:<25.2e}")
            freq_values[freq].append(val)

        if band_frequencies:
            print("\n  Average self-noise in specified frequency bands:")
            print(f"    Frequency Band (Hz) | Avg Self-Noise (dB) | Avg Self-Noise ({org_data_unit}/√Hz)")
            print(f"    --------------- | ----------------- | -------------------")
            for band in band_frequencies:
                low, high = band
                band_mask = (frequencies >= low) & (frequencies <= high)
                if np.any(band_mask):
                    band_rms = np.sqrt(np.mean(self_noise[band_mask] ** 2))
                    band_rms_db = 20 * np.log10(band_rms)
                    print(f"    [{low:6.1f}, {high:6.1f}]   | {band_rms_db:12.2f} dB | {band_rms:12.2e}")
                    band_values[f"{low}-{high}"].append(band_rms)
