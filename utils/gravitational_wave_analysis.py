import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import simps
from scipy.fft import fft, fftfreq
from astrobiology.directional_equations import schwarzschild_radius, planck_energy, hawking_temperature

def detect_gravitational_waves(signal, threshold=0.001):
    """
    Detect gravitational waves in a time-domain signal using a threshold-based approach.
    
    Parameters:
    signal (numpy array): The input time-domain signal.
    threshold (float): The threshold for peak detection.
    
    Returns:
    numpy array: Indices of detected peaks corresponding to gravitational wave events.
    """
    peaks, _ = find_peaks(signal, height=threshold)
    return peaks

def calculate_strain_amplitude(frequency, distance, chirp_mass):
    """
    Calculate the strain amplitude of a gravitational wave using the quadrupole formula.
    
    Parameters:
    frequency (float): The frequency of the gravitational wave (Hz).
    distance (float): The distance to the source (meters).
    chirp_mass (float): The chirp mass of the binary system (solar masses).
    
    Returns:
    float: The strain amplitude of the gravitational wave.
    """
    # Convert chirp mass from solar masses to kilograms
    chirp_mass_kg = chirp_mass * M_SUN
    
    # Gravitational wave strain amplitude
    strain = (4 * (G * chirp_mass_kg)**(5/3) * (np.pi * frequency)**(2/3)) / (C**4 * distance)
    return strain

def calculate_energy_spectrum(signal, sampling_rate):
    """
    Calculate the energy spectrum of a gravitational wave signal.
    
    Parameters:
    signal (numpy array): The input time-domain signal.
    sampling_rate (float): The sampling rate of the signal (Hz).
    
    Returns:
    tuple: Frequencies and corresponding energy spectrum values.
    """
    # Perform Fourier Transform
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)
    
    # Calculate energy spectrum
    energy_spectrum = np.abs(yf)**2 / N
    return xf, energy_spectrum

def calculate_total_energy(signal, sampling_rate):
    """
    Calculate the total energy of a gravitational wave signal.
    
    Parameters:
    signal (numpy array): The input time-domain signal.
    sampling_rate (float): The sampling rate of the signal (Hz).
    
    Returns:
    float: The total energy of the signal.
    """
    _, energy_spectrum = calculate_energy_spectrum(signal, sampling_rate)
    total_energy = simps(energy_spectrum, dx=1/sampling_rate)
    return total_energy

def apply_astrophysical_transformations(signal):
    """
    Apply astrophysical transformations to a gravitational wave signal.
    
    Parameters:
    signal (numpy array): The input time-domain signal.
    
    Returns:
    numpy array: The transformed signal.
    """
    # Apply Schwarzschild radius transformation
    schwarzschild_transformed = schwarzschild_radius(signal)
    
    # Apply Planck energy transformation
    planck_transformed = planck_energy(signal)
    
    # Apply Hawking temperature transformation
    hawking_transformed = hawking_temperature(signal)
    
    # Combine all transformations into a final signal
    transformed_signal = (schwarzschild_transformed + planck_transformed + hawking_transformed) / 3
    return transformed_signal

# Constants for the gravitational wave analysis
M_SUN = 1.989e30  # Solar mass in kilograms

# FOR TESTS
# if __name__ == "__main__":
#     # Generate a sample gravitational wave signal
#     time = np.linspace(0, 1, 1000)
#     signal = np.sin(2 * np.pi * 50 * time) * np.exp(-time)
    
#     # Detect gravitational waves in the signal
#     detected_peaks = detect_gravitational_waves(signal)
#     print("Detected peaks at indices:", detected_peaks)
    
#     # Calculate strain amplitude
#     strain = calculate_strain_amplitude(50, 1e22, 30)
#     print("Calculated strain amplitude:", strain)
    
#     # Calculate energy spectrum
#     frequencies, energy_spectrum = calculate_energy_spectrum(signal, 1000)
#     print("Energy spectrum calculated.")
    
#     # Calculate total energy
#     total_energy = calculate_total_energy(signal, 1000)
#     print("Total energy of the signal:", total_energy)
    
#     # Apply astrophysical transformations
#     transformed_signal = apply_astrophysical_transformations(signal)
#     print("Transformed signal obtained.")
