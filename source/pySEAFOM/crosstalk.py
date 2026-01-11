import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.fft import fft


def _add_brand_box(ax, text: str = "pySEAFOM 0.1"):
    bbox_props = dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.7)
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=20,
        fontweight="bold",
        bbox=bbox_props,
        alpha=1.0,
    )


def calculate_crosstalk(
    *,
    section_data,
    stimulus_freq,
    fs,
    fft_size=16384,
    gauge_length=10.0,
    stretcher_length=10.0,
    channel_spacing=1.0,
    gl_multiplier=50,
):
    """Compute crosstalk profile and maximum crosstalk (dB) for one section.

    Parameters
    ----------
    section_data : ndarray
        2D array shaped (n_ssl, n_samples). The stimulation point is assumed to
        be at the center SSL (n_ssl//2).
    stimulus_freq : float
        Stimulus frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    fft_size : int, optional
        FFT block size (default 16384).
    gauge_length : float
        Gauge length [m]. Used to define exclusion/analysis zones.
    stretcher_length : float
        Length of the stretcher region [m]. Used to define the reference zone.
    channel_spacing : float
        Channel spacing [m] (distance per SSL).
    gl_multiplier : int, optional
        Maximum GL multiple used for the crosstalk search region (default 50).

    Returns
    -------
    crosstalk_db : ndarray
        1D array length n_ssl, in dB re reference level (reference ~ stretcher zone).
        Values in center ±2GL are set to NaN.
    max_xt_db : float
        Maximum crosstalk (dB) within the region outside ±3GL and up to ±gl_multiplier*GL.
    magnitudes : ndarray
        Mean magnitude at the stimulus frequency for each SSL (linear units).
    reference_level : float
        Mean magnitude in the reference (stretcher) zone (linear units).
    """
    data = np.asarray(section_data, float)
    if data.ndim != 2:
        raise ValueError("section_data must be a 2D array (n_ssl, n_samples)")

    n_ssl, n_samples = data.shape
    if n_samples < fft_size:
        raise ValueError("section_data has fewer samples than fft_size")

    n_blocks = n_samples // int(fft_size)
    if n_blocks <= 0:
        raise ValueError("Not enough samples for even one FFT block")

    center_ssl = int(n_ssl // 2)

    gl_ssl = int(np.round(float(gauge_length) / float(channel_spacing)))
    if gl_ssl <= 0:
        gl_ssl = 1

    stretcher_half_ssl = int(np.round(float(stretcher_length) / float(channel_spacing) / 2.0))
    if stretcher_half_ssl <= 0:
        stretcher_half_ssl = 1

    magnitudes = np.zeros(n_ssl, dtype=float)

    win = windows.flattop(int(fft_size))
    freqs = np.fft.fftfreq(int(fft_size), d=1.0 / float(fs))[: int(fft_size) // 2]
    stim_idx = int(np.argmin(np.abs(freqs - float(stimulus_freq))))

    for ssl in range(n_ssl):
        signal = data[ssl]
        mag_list = []
        for i in range(n_blocks):
            block = signal[i * fft_size : (i + 1) * fft_size]
            spectrum = np.abs(fft(block * win))[: int(fft_size) // 2]
            mag_list.append(float(spectrum[stim_idx]))
        magnitudes[ssl] = float(np.mean(mag_list))

    ref_lo = max(0, center_ssl - stretcher_half_ssl)
    ref_hi = min(n_ssl - 1, center_ssl + stretcher_half_ssl)
    ref_slice = slice(ref_lo, ref_hi + 1)

    reference_level = float(np.mean(magnitudes[ref_slice]))
    if not np.isfinite(reference_level) or reference_level <= 0.0:
        raise ValueError("Reference level is invalid (<=0 or non-finite)")

    crosstalk_db = 20.0 * np.log10(magnitudes / reference_level)

    # Mask center ±2GL (exclude the drive region)
    exc_lo = max(0, center_ssl - 2 * gl_ssl)
    exc_hi = min(n_ssl - 1, center_ssl + 2 * gl_ssl)
    crosstalk_db[exc_lo : exc_hi + 1] = np.nan

    if exc_lo == 0 and exc_hi == n_ssl - 1:
        print(
            "Crosstalk exclusion region (±2GL) covers the entire section; "
            "crosstalk profile is all NaN. Increase section length or decrease gauge_length/channel_spacing."
        )

    # Compute max crosstalk outside ±3GL and within ±gl_multiplier*GL
    search_lo_gl = 3
    search_hi_gl_ssl = int(gl_multiplier*gauge_length / channel_spacing)

    # Build search regions WITHOUT clipping the ±3GL boundary inward.
    # If the boundary falls outside the section, the corresponding region is empty.
    left_hi = center_ssl - search_lo_gl * gl_ssl
    left_lo = max(0, center_ssl - search_hi_gl_ssl)
    if left_hi <= left_lo:
        left = np.array([], dtype=float)
    else:
        left = crosstalk_db[left_lo:left_hi]

    right_lo = center_ssl + search_lo_gl * gl_ssl
    right_hi = min(n_ssl - 1, center_ssl + search_hi_gl_ssl)
    if right_lo > right_hi:
        right = np.array([], dtype=float)
    else:
        right = crosstalk_db[right_lo : right_hi + 1]

    if left.size == 0 and right.size == 0:
        print(
            "Insufficient channels beyond ±3GL to compute max_xt_db; returning NaN. "
            f"n_ssl={n_ssl}, GL={gauge_length} m, channel_spacing={channel_spacing} m."
        )

    # Avoid NumPy "All-NaN slice" warnings by only taking maxima over finite values.
    finite_candidates = []
    if left.size:
        left_finite = left[np.isfinite(left)]
        if left_finite.size:
            finite_candidates.append(float(np.max(left_finite)))
    if right.size:
        right_finite = right[np.isfinite(right)]
        if right_finite.size:
            finite_candidates.append(float(np.max(right_finite)))

    max_xt_db = float(np.max(finite_candidates)) if finite_candidates else float("nan")

    return {
        "crosstalk_db": crosstalk_db,
        "max_xt_db": float(max_xt_db),
        "magnitudes": magnitudes,
        "reference_level": float(reference_level),
    }


def plot_crosstalk(crosstalk_db, *, channel_spacing=1.0, title="Crosstalk Profile", label=None):
    """Plot a single crosstalk profile (dB vs distance)."""
    crosstalk_db = np.asarray(crosstalk_db, float)
    center_index = int(crosstalk_db.size // 2)
    distances = (np.arange(crosstalk_db.size) - center_index) * float(channel_spacing)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.plot(distances, crosstalk_db, label=label)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Reference")

    label_fontsize = 16
    title_fontsize = 16
    legend_fontsize = 16

    ax.set_xlabel("Distance from stimulation point (m)", fontsize=label_fontsize)
    ax.set_ylabel("Amplitude (dB re. reference)", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    ax.grid(True, which="both")
    ax.grid(True, which="major", linestyle="-")
    ax.grid(True, which="minor", linestyle="--", alpha=0.5)

    if label is not None:
        ax.legend(fontsize=legend_fontsize)

    _add_brand_box(ax)

    plt.tight_layout()
    return fig, ax


def plot_crosstalk_sections(result, *, channel_spacing=1.0, title="Crosstalk Profile", section_label=None):
    """Plot a single crosstalk profile from a calculate_crosstalk() result.

    Note: Despite the historical name, this function now supports a single section only.
    """
    if not isinstance(result, dict):
        raise ValueError("result must be a dict returned by calculate_crosstalk()")
    return plot_crosstalk(
        result.get("crosstalk_db"),
        channel_spacing=channel_spacing,
        title=title,
        label=section_label,
    )


def report_crosstalk(result, *, section_label="Section 1"):
    """Print max crosstalk for a single section."""
    if not isinstance(result, dict):
        raise ValueError("result must be a dict returned by calculate_crosstalk()")

    mx = float(result.get("max_xt_db", float("nan")))
    ref = float(result.get("reference_level", float("nan")))

    print("\npySEAFOM — Crosstalk Report")
    print("---------------------------")
    print(f"{section_label}: max_xt = {mx:.2f} dB | reference_level = {ref:.3e}")
