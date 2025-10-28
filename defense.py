import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Parameters ---
print("Setting up simulation parameters...")

# Waveform Parameters
fs = 100e6         # Sampling frequency (100 MHz)
T_pulse = 20e-6    # Pulse duration (20 us)
B_chirp = 40e6     # Chirp bandwidth (40 MHz)
c = 3e8            # Speed of light
f_carrier = 10e9   # Carrier frequency (10 GHz, X-band)
lambda_c = c / f_carrier # Wavelength

# Target Parameters
target_range = 1500  # Target range in meters (1.5 km)
target_velocity = 200 # Target velocity in m/s (approx. 450 mph)
snr_db = -20         # Signal-to-Noise Ratio in dB (TARGET IS "INVISIBLE")

# Processing Parameters
num_pulses = 128     # Number of pulses to integrate (N)
T_pri = 100e-6       # Pulse Repetition Interval (100 us)
num_samples_pulse = int(T_pulse * fs)
num_samples_pri = int(T_pri * fs)

# --- 2. Generate Waveforms ---

# Create the time vector for a single pulse
t = np.linspace(0, T_pulse, num_samples_pulse, endpoint=False)

# Create the transmitted LFM (Linear Frequency Modulation) "chirp"
k = B_chirp / T_pulse  # Chirp rate
tx_pulse = np.exp(1j * np.pi * k * t**2)

# Create the Matched Filter (time-reversed conjugate of the pulse)
matched_filter = np.conj(np.flip(tx_pulse))

# --- 3. Simulate Received Data ---
print(f"Simulating {num_pulses} pulses... (Target SNR: {snr_db} dB)")

# Calculate target delay and Doppler
t_delay = 2 * target_range / c
delay_samples = int(np.round(t_delay * fs))
f_doppler = 2 * target_velocity / lambda_c

# Initialize the "raw data" matrix (Pulses x Range Samples)
# We simulate a range window up to T_pri
raw_data = np.zeros((num_pulses, num_samples_pri), dtype=complex)

# Calculate noise and signal power
noise_power_db = 0
noise_power = 10**(noise_power_db / 10)
noise = (np.random.randn(num_pulses, num_samples_pri) + 
         1j * np.random.randn(num_pulses, num_samples_pri)) * np.sqrt(noise_power / 2)

signal_power = noise_power * 10**(snr_db / 10)
signal_amplitude = np.sqrt(signal_power)

# Add noise to the raw data
raw_data += noise

# Add the target signal pulse by pulse
for i in range(num_pulses):
    # Calculate Doppler phase shift for this pulse
    doppler_phase_shift = np.exp(1j * 2 * np.pi * f_doppler * (i * T_pri))
    
    # Add the delayed pulse with Doppler shift
    start_idx = delay_samples
    end_idx = delay_samples + num_samples_pulse
    
    if end_idx < num_samples_pri:
        raw_data[i, start_idx:end_idx] += \
            tx_pulse * doppler_phase_shift * signal_amplitude

# --- 4. DSP Processing Chain ---

# === STEP 1: Pulse Compression (Matched Filter) ===
print("Step 1: Applying Pulse Compression...")
range_compressed_data = np.zeros_like(raw_data)
for i in range(num_pulses):
    # Convolve each pulse with the matched filter
    range_compressed_data[i, :] = np.convolve(raw_data[i, :], matched_filter, 'same')

# === STEP 2 & 3: Doppler Processing & Coherent Integration (FFT) ===
print("Step 2 & 3: Applying Doppler FFT (Coherent Integration)...")

# "Corner Turn" is implicit here. We apply FFT along the 'pulse' (slow-time) axis.
# This FFT performs the coherent integration (Gain = N)
range_doppler_map = np.fft.fft(range_compressed_data, axis=0)

# Shift the zero-Doppler bin to the center
range_doppler_map = np.fft.fftshift(range_doppler_map, axes=0)

# Convert to dB for visualization
range_doppler_map_db = 20 * np.log10(np.abs(range_doppler_map))
range_doppler_map_db_normalized = range_doppler_map_db - np.max(range_doppler_map_db)


# === STEP 4: CFAR Detection ===
print("Step 4: Applying CFAR Detection...")

def ca_cfar_1d(data_power, num_train, num_guard, pfa):
    """
    1D Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector.
    
    data_power: Input 1D array of power values (amplitude squared).
    num_train: Number of training cells (must be even).
    num_guard: Number of guard cells (must be even).
    pfa: Probability of False Alarm.
    """
    N = num_train
    alpha = N * (pfa**(-1/N) - 1)  # CFAR scaling factor
    
    detections = []
    threshold_line = []
    
    # Half-window size for one side
    half_window = num_train // 2 + num_guard // 2
    
    for i in range(half_window, len(data_power) - half_window):
        # Cell Under Test (CUT)
        cut = data_power[i]
        
        # Define training cell windows
        # Lagging window
        lag_start = i - half_window
        lag_end = i - num_guard // 2
        
        # Leading window
        lead_start = i + num_guard // 2 + 1
        lead_end = i + half_window + 1
        
        lagging_window = data_power[lag_start : lag_end]
        leading_window = data_power[lead_start : lead_end]
        
        # Calculate local noise power
        noise_power_estimate = (np.sum(lagging_window) + np.sum(leading_window)) / N
        
        # Calculate threshold
        threshold = alpha * noise_power_estimate
        threshold_line.append(threshold)
        
        # Compare CUT to threshold
        if cut > threshold:
            detections.append(i)
            
    return detections, threshold_line


# --- 5. Visualization ---
print("Generating plots...")
plt.style.use('seaborn-v0_8-darkgrid')

# === Plot 1: Raw Signal (Target Invisible) ===
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(np.abs(raw_data[0, :]))
plt.title(f"Raw Signal (First Pulse)\nTarget at {delay_samples} (Invisible, SNR {snr_db} dB)")
plt.xlabel("Range Sample")
plt.ylabel("Amplitude")

# === Plot 2: After Pulse Compression (Target Visible) ===
# Calculate Gain
gain_pc = T_pulse * B_chirp
snr_after_pc = snr_db + 10 * np.log10(gain_pc)

plt.subplot(1, 2, 2)
plt.plot(np.abs(range_compressed_data[0, :]))
plt.title(f"After Pulse Compression (Gain: {10*np.log10(gain_pc):.1f} dB)\nNew SNR: {snr_after_pc:.1f} dB")
plt.xlabel("Range Sample")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# === Plot 3: Range-Doppler Map (Target Detected) ===
plt.figure(figsize=(10, 7))
# Create axes
range_axis = np.linspace(0, (num_samples_pri/fs) * c / 2, num_samples_pri)
doppler_freq_axis = np.linspace(-fs/2 / (num_samples_pri/num_samples_pulse), fs/2 / (num_samples_pri/num_samples_pulse), num_pulses)
# A simpler Doppler velocity axis
velocity_axis = (doppler_freq_axis * lambda_c) / 2

plt.imshow(range_doppler_map_db_normalized, 
           aspect='auto', 
           extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
           cmap='jet')
plt.title(f"Range-Doppler Map (After FFT Gain: {10*np.log10(num_pulses):.1f} dB)")
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")
plt.colorbar(label="Power (dB, normalized)")
plt.show()

# === Plot 4: CFAR Detection ===
# Find the range bin with the target
target_range_bin = np.argmax(np.max(np.abs(range_doppler_map), axis=0))

# Get the Doppler slice at the target's range
doppler_slice = np.abs(range_doppler_map[:, target_range_bin])**2

# Run CFAR
detections, threshold = ca_cfar_1d(doppler_slice, 
                                     num_train=20, 
                                     num_guard=4, 
                                     pfa=1e-3)

plt.figure(figsize=(12, 6))
plt.plot(doppler_slice, label="Doppler Slice Power")
# Plot threshold (needs offset)
cfar_start_index = (20 // 2) + (4 // 2)
plt.plot(np.arange(cfar_start_index, cfar_start_index + len(threshold)), 
         threshold, 'r--', label="CFAR Adaptive Threshold")

# Plot detections
if detections:
    plt.plot(detections, doppler_slice[detections], 'go', 
             markersize=12, label="CFAR Detection!")

plt.title(f"CFAR Detection on Doppler Slice (at Range Bin {target_range_bin})")
plt.xlabel("Doppler Bin")
plt.ylabel("Power")
plt.legend()
plt.show()