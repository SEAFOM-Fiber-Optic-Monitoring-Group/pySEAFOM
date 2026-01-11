import numpy as np
from scipy.signal import windows
from scipy.fft import fft
def flattop_coherent_gain(N: int) -> float:
    """Return coherent gain of an N-point flattop window."""
    win = windows.flattop(N)
    return float(np.sum(win) / N)


def normalize_fft_magnitude(raw_fft, N: int) -> np.ndarray:
    """Single-sided amplitude spectrum, corrected for window coherent gain."""
    cg = flattop_coherent_gain(N)
    # 2/N for single-sided amplitude, divide by coherent gain
    return (2.0 / (N * cg)) * np.abs(raw_fft)

def is_good_quality_block(block, fs, stimulus_freq, snr_threshold_db: float = -40.0) -> bool:
    """Return True if the block meets a simple SNR quality gate."""
    block = np.asarray(block, float)
    if block.ndim != 1 or block.size < 4:
        return False

    N = block.size
    win = windows.flattop(N)
    windowed = block * win

    raw_fft = fft(windowed)[: N // 2]
    spectrum = normalize_fft_magnitude(raw_fft, N)

    freqs = np.fft.fftfreq(N, d=1.0 / fs)[: N // 2]
    stim_idx = int(np.argmin(np.abs(freqs - stimulus_freq)))

    signal_power = float(spectrum[stim_idx] ** 2)

    noise_spectrum = spectrum.copy()
    noise_spectrum[stim_idx] = 0.0
    if N // 2 > 0:
        noise_spectrum[0] = 0.0

    noise_power = float(np.mean(noise_spectrum**2))
    if noise_power <= 0.0:
        return False

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return snr_db >= snr_threshold_db


def compute_thd(
    signal,
    fs,
    stimulus_freq,
    *,
    fft_size=16384,
    harmonics=5,
    snr_threshold_db=-40.0,
    min_good_blocks=10,
    max_good_blocks=200,
):
    """Compute THD (%) and harmonics (dB re fundamental) using block FFT.

    The function splits the signal into consecutive non-overlapping blocks of
    length `fft_size`, screens blocks using `is_good_quality_block`, then
    averages harmonic magnitudes across accepted blocks.

    Parameters
    ----------
    signal : array-like
        1D time series.
    fs : float
        Sampling rate in Hz.
    stimulus_freq : float
        Fundamental frequency in Hz.
    fft_size : int, optional
        Block size for FFT (SEAFOM commonly uses 16384).
    harmonics : int, optional
        Number of harmonics to evaluate, including the fundamental (default 5).
    snr_threshold_db : float, optional
        SNR gate for accepting blocks.
    min_good_blocks : int, optional
        Minimum number of accepted blocks required.
    max_good_blocks : int, optional
        Stop early after this many good blocks.

    Returns
    -------
    thd_percent : float
    harmonics_db : ndarray
        Array length `harmonics`, in dB re fundamental (0 dB at index 0).
    n_good_blocks : int
    """
    sig = np.asarray(signal, float)
    if sig.ndim != 1:
        raise ValueError("signal must be a 1D array")

    if fft_size <= 0:
        raise ValueError("fft_size must be > 0")

    n_blocks = sig.size // int(fft_size)
    if n_blocks <= 0:
        raise ValueError("signal is too short for the chosen fft_size")

    harmonic_amplitudes = []
    good_blocks = 0

    for block_idx in range(n_blocks):
        block = sig[block_idx * fft_size : (block_idx + 1) * fft_size]

        if not is_good_quality_block(block, fs, stimulus_freq, snr_threshold_db=snr_threshold_db):
            continue

        win = windows.flattop(fft_size)
        spectrum = np.abs(fft(block * win))[: fft_size // 2]
        freqs = np.fft.fftfreq(fft_size, d=1.0 / fs)[: fft_size // 2]

        amps = []
        for h in range(1, int(harmonics) + 1):
            f_harm = h * stimulus_freq
            idx = int(np.argmin(np.abs(freqs - f_harm)))
            amps.append(float(spectrum[idx]))

        harmonic_amplitudes.append(amps)
        good_blocks += 1

        if max_good_blocks is not None and good_blocks >= int(max_good_blocks):
            break

    if good_blocks < int(min_good_blocks):
        raise ValueError(
            f"Only found {good_blocks} good blocks, need minimum {int(min_good_blocks)}"
        )

    harmonic_amplitudes = np.asarray(harmonic_amplitudes, float)
    mean_amps = np.mean(harmonic_amplitudes, axis=0)

    v1 = float(mean_amps[0])
    if v1 <= 0.0:
        raise ValueError("Fundamental amplitude is zero; cannot compute THD")

    distortion = float(np.sqrt(np.sum(mean_amps[1:] ** 2)))
    thd_percent = (distortion / v1) * 100.0

    harmonics_db = 20.0 * np.log10(mean_amps / v1)
    return thd_percent, harmonics_db, good_blocks


def report_fidelity_thd(results, *, show_harmonics=True, show_per_channel=True):
    """Improved reporting for SEAFOM‑aligned fidelity THD results."""

    print("\npySEAFOM — Fidelity (THD) Report")
    print("--------------------------------")
    print(f"fs: {results.get('fs')} Hz | stimulus_freq: {results.get('stimulus_freq')} Hz")
    print(f"fft_size: {results.get('fft_size')} | harmonics: {results.get('harmonics')}")
    print(f"snr_threshold_db: {results.get('snr_threshold_db')} dB")

    for sec in results.get("sections", []):
        print(f"\n{sec.get('name')}: channels {sec.get('channel_range')}")

        for lvl in sec.get("levels", []):
            t0, t1 = lvl.get("time_steps", [None, None])
            mean_thd = lvl.get("thd_percent")
            print(f"  Level {lvl.get('level_index')}:")
            print(f"    Mean THD: {mean_thd:.3f}% | samples [{t0}, {t1})")

            # Per‑channel THD
            if show_per_channel and "per_channel_thd" in lvl:
                pcs = lvl["per_channel_thd"]
                print(f"    Per‑channel THD (%):")
                for ch, thd in enumerate(pcs):
                    print(f"      Ch {ch}: {thd:.3f}%")

            # Harmonics (averaged)
            if show_harmonics:
                hdb = np.asarray(lvl.get("harmonics_db"), float)
                msg = ", ".join([f"H{i+1}:{hdb[i]:.1f} dB" for i in range(hdb.size)])
                print(f"    Harmonics (avg): {msg}")

            # Good blocks (min across channels)
            if "n_good_blocks" in lvl:
                print(f"    Good blocks (min across channels): {lvl['n_good_blocks']}")


def compute_thd_single_channel(
    signal,
    fs,
    stimulus_freq,
    *,
    fft_size: int = 16384,
    harmonics: int = 5,
    snr_threshold_db: float = -40.0,
    min_good_blocks: int = 10,
    max_good_blocks: int | None = 200,
):
    """Compute THD (%) and harmonics (dB re fundamental) for a single channel."""

    sig = np.asarray(signal, float)
    if sig.ndim != 1:
        raise ValueError("signal must be 1D")

    n_blocks = sig.size // int(fft_size)
    if n_blocks <= 0:
        raise ValueError("Signal too short for fft_size")

    harmonic_amplitudes = []
    good_blocks = 0

    for b in range(n_blocks):
        block = sig[b * fft_size : (b + 1) * fft_size]

        if not is_good_quality_block(block, fs, stimulus_freq, snr_threshold_db):
            continue

        N = int(fft_size)
        win = windows.flattop(N)
        raw_fft = fft(block * win)[: N // 2]
        spectrum = normalize_fft_magnitude(raw_fft, N)

        freqs = np.fft.fftfreq(N, d=1.0 / fs)[: N // 2]

        amps = []
        for h in range(1, int(harmonics) + 1):
            f_h = h * stimulus_freq
            idx = int(np.argmin(np.abs(freqs - f_h)))
            amps.append(float(spectrum[idx]))

        harmonic_amplitudes.append(amps)
        good_blocks += 1

        if max_good_blocks is not None and good_blocks >= int(max_good_blocks):
            break

    if good_blocks < int(min_good_blocks):
        raise ValueError(f"Only {good_blocks} good blocks found, need >= {int(min_good_blocks)}")

    harmonic_amplitudes = np.asarray(harmonic_amplitudes, float)
    mean_amps = np.mean(harmonic_amplitudes, axis=0)

    v1 = float(mean_amps[0])
    if not np.isfinite(v1) or v1 <= 0.0:
        raise ValueError("Fundamental amplitude is zero or invalid; cannot compute THD")

    distortion = float(np.sqrt(np.sum(mean_amps[1:] ** 2)))
    thd_percent = (distortion / v1) * 100.0

    harmonics_db = 20.0 * np.log10(mean_amps / v1)
    return thd_percent, harmonics_db, good_blocks


def calculate_fidelity_thd(
    time_series_data,
    *,
    fs=10000,
    levels_time_steps=None,
    stimulus_freq=500.0,
    snr_threshold_db=-40.0,
    fft_size=16384,
    harmonics=5,
    section_name="Section 1",
):
    """Compute THD across levels for a section (per-channel, then averaged)."""

    data = np.asarray(time_series_data, float)
    if data.ndim != 2:
        raise ValueError("time_series_data must be 2D (n_channels, n_samples)")

    n_channels, n_samples = data.shape

    if levels_time_steps is None:
        raise ValueError("levels_time_steps is required")

    # Allow [t0, t1] or [[t0, t1], ...]
    lts = levels_time_steps
    if (
        isinstance(lts, (list, tuple, np.ndarray))
        and len(lts) == 2
        and np.isscalar(lts[0])
        and np.isscalar(lts[1])
    ):
        levels_time_steps = [lts]

    results = {
        "fs": float(fs),
        "stimulus_freq": float(stimulus_freq),
        "snr_threshold_db": float(snr_threshold_db),
        "fft_size": int(fft_size),
        "harmonics": int(harmonics),
        "sections": [],
    }

    section_out = {
        "name": str(section_name),
        "channel_range": [0, n_channels - 1],
        "levels": [],
    }

    for level_idx, (t0, t1) in enumerate(levels_time_steps):
        t0 = int(t0)
        t1 = int(t1)
        if t0 < 0 or t1 <= t0 or t1 > n_samples:
            raise ValueError(f"Invalid levels_time_steps[{level_idx}] = [{t0}, {t1}]")

        per_channel_thd = []
        per_channel_harmonics = []
        per_channel_good_blocks = []

        for ch in range(n_channels):
            seg = data[ch, t0:t1]

            thd_percent, harmonics_db, n_good_blocks = compute_thd_single_channel(
                seg,
                fs,
                stimulus_freq,
                fft_size=int(fft_size),
                harmonics=int(harmonics),
                snr_threshold_db=float(snr_threshold_db),
            )

            per_channel_thd.append(float(thd_percent))
            per_channel_harmonics.append(np.asarray(harmonics_db, float))
            per_channel_good_blocks.append(int(n_good_blocks))

        mean_thd = float(np.mean(per_channel_thd))

        # Average harmonics across channels in linear domain
        h_db_stack = np.vstack(per_channel_harmonics)  # shape: (n_channels, n_harmonics)
        h_lin_stack = 10.0 ** (h_db_stack / 20.0)
        h_lin_mean = np.mean(h_lin_stack, axis=0)
        harmonics_db_mean = 20.0 * np.log10(h_lin_mean)

        level_entry = {
            "level_index": int(level_idx),
            "time_steps": [t0, t1],
            "thd_percent": mean_thd,  # same key as before
            "harmonics_db": harmonics_db_mean,
            "n_good_blocks": int(min(per_channel_good_blocks)),
            "per_channel_thd": per_channel_thd,
        }

        section_out["levels"].append(level_entry)

    results["sections"].append(section_out)
    return results
