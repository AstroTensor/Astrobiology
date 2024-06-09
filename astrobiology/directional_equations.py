from astrobiology.constants import C, G, H, HBAR, KB, NA, R, SIGMA, B, R_INF, A_0, ALPHA, M_P, M_N, M_E, M_U, F, MU_0, JSOF, EPSILON_0, E, PHI_0, G_0, R_K, K_J, H_OVER_2E, H_OVER_E, EV, U, E_H, C_R_INF, SIGMA_E, R_E, G_E, R_H, MU_B, MU_N, ALPHA_INV, Z_0, N_0, V_M, S_0_OVER_R, C_1, C_1_OVER_2PI, C_2, C_1_PRIME, C_1_PRIME_OVER_2PI, M_U_C2
from astrobiology.mit_library import effective_temperature, hubble_law_velocity, gravitational_binding_energy, mach_number, get_current_jsof
# Utility functions with equations for predicting asteroid flight paths

def schwarzschild_radius(mass):
    """
    Calculate the Schwarzschild radius of a black hole
    R_s = 2 * G * M / C^2
    """
    return 2 * G * mass / C**2

def planck_energy(frequency):
    """
    Calculate the energy of a photon given its frequency
    E = h * f
    """
    return H * frequency

def torque_equation(force, radius):
    """
    Calculate the torque on an object given a force and radius
    tau = F * r
    """
    return force * radius

def angular_momentum_equation(mass, velocity, radius):
    """
    Calculate the angular momentum of an object
    L = m * v * r
    """
    return mass * velocity * radius

def gravitational_time_dilation(mass, radius):
    """
    Calculate the gravitational time dilation near a massive object
    t = (1 - (2 * G * M) / (r * c^2))**0.5
    """
    return (1 - (2 * G * mass) / (radius * C**2))**0.5

def random_mass():
    """
    Generate a random mass in the range [1e10, 1e12] kg
    """
    return np.random.uniform(1e10, 1e12)

def random_velocity():
    """
    Generate a random velocity in the range [1e3, 1e5] m/s
    """
    return np.random.uniform(1e3, 1e5)

def random_radius():
    """
    Generate a random radius in the range [1e3, 1e6] m
    """
    return np.random.uniform(1e3, 1e6)

def random_distance():
    """
    Generate a random distance in the range [1e7, 1e9] m
    """
    return np.random.uniform(1e7, 1e9)

def random_frequency():
    """
    Generate a random frequency in the range [1e12, 1e16] Hz
    """
    return np.random.uniform(1e12, 1e16)

def random_chirp_mass():
    """
    Generate a random chirp mass in the range [1e9, 1e11] kg
    """
    return np.random.uniform(1e9, 1e11)

# def hawking_temperature(mass):
#     """
#     Calculate the temperature of a black hole due to Hawking radiation
#     T = (hbar * c^3) / (8 * pi * G * M * k_B)
#     """
#     return (HBAR * C**3) / (8 * 3.14159 * G * mass * KB)

def luminosity_distance(redshift, H0, OmegaM, OmegaLambda):
    """
    Calculate the luminosity distance in a flat universe with dark energy
    D_L = (c/H0) * (1 + z) * integral from 0 to z of [dz' / sqrt(OmegaM * (1 + z')^3 + OmegaLambda)]
    """
    from scipy.integrate import quad
    integrand = lambda z: 1 / ((OmegaM * (1 + z)**3 + OmegaLambda)**0.5)
    integral, _ = quad(integrand, 0, redshift)
    return (C / H0) * (1 + redshift) * integral

def compton_wavelength(mass):
    """
    Calculate the Compton wavelength of a particle
    lambda_C = h / (m * c)
    """
    return H / (mass * C)

def bessel_approach(n, x):
    """
    Compute the Bessel function of the first kind
    J_n(x) = sum from k=0 to infinity of [(-1)^k * (x/2)^(2k+n)] / [k! * Gamma(k+n+1)]
    """
    from scipy.special import jn
    return jn(n, x)

def de_broglie_wavelength(momentum):
    """
    Calculate the de Broglie wavelength of a particle
    lambda = h / p
    """
    return H / momentum

def hubble_parameter(z, H0, OmegaM, OmegaLambda):
    """
    Calculate the Hubble parameter at a given redshift
    H(z) = H0 * sqrt(OmegaM * (1 + z)^3 + OmegaLambda)
    """
    return H0 * ((OmegaM * (1 + z)**3 + OmegaLambda)**0.5)

def gravitational_redshift(mass, radius):
    """
    Calculate the gravitational redshift near a massive object
    z = sqrt(1 - (2 * G * M) / (r * c^2)) - 1
    """
    return (1 - (2 * G * mass) / (radius * C**2))**0.5 - 1

def synodic_period(T_e, T_p):
    """
    Calculate the synodic period of two orbiting bodies
    P_syn = 1 / (1 / T_e - 1 / T_p)
    """
    return 1 / (1 / T_e - 1 / T_p)

def lorentz_factor(velocity):
    """
    Calculate the Lorentz factor for a given velocity
    gamma = 1 / sqrt(1 - (v^2 / c^2))
    """
    return 1 / (1 - (velocity**2 / C**2))**0.5

def escape_velocity(mass, radius):
    """
    Calculate the escape velocity from a massive object
    v_e = sqrt(2 * G * M / r)
    """
    return (2 * G * mass / radius)**0.5

def parsec_to_lightyear(parsecs):
    """
    Convert parsecs to light-years
    1 parsec = 3.26156 light-years
    """
    return parsecs * 3.26156

def chandrasekhar_limit():
    """
    Calculate the Chandrasekhar limit
    M_Ch = (5.83 * sqrt(hbar * c / G)) / (m_p^2)
    """
    return (5.83 * (HBAR * C / G)**0.5) / M_P**2

def ecliptic_longitude(l, g, omega):
    """
    Calculate the ecliptic longitude of an object
    lambda = l + (1.915 * sin(g)) + (0.020 * sin(2 * g)) - omega
    """
    from numpy import sin, deg2rad
    return l + (1.915 * sin(deg2rad(g))) + (0.020 * sin(deg2rad(2 * g))) - omega

def orbital_period(semi_major_axis, mass):
    """
    Calculate the orbital period of a body around a star
    T = 2 * pi * sqrt(a^3 / (G * M))
    """
    return 2 * 3.14159 * (semi_major_axis**3 / (G * mass))**0.5

def roche_limit(primary_radius, primary_density, secondary_density):
    """
    Calculate the Roche limit
    d = R_p * (2 * (rho_p / rho_s))^(1/3)
    """
    return primary_radius * (2 * (primary_density / secondary_density))**(1/3)

def effective_temperature(luminosity, radius):
    """
    Calculate the effective temperature of a star
    T_eff = (L / (4 * pi * sigma * R^2))^(1/4)
    """
    return (luminosity / (4 * 3.14159 * SIGMA * radius**2))**0.25

def doppler_shift(observed_wavelength, rest_wavelength):
    """
    Calculate the Doppler shift
    z = (lambda_obs - lambda_rest) / lambda_rest
    """
    return (observed_wavelength - rest_wavelength) / rest_wavelength

def jean_mass(temperature, particle_mass, density):
    """
    Calculate the Jean mass
    M_J = ((5 * k_B * T) / (G * m))^(3/2) / ((3 / (4 * pi * rho))^(1/2))
    """
    return ((5 * KB * temperature) / (G * particle_mass))**(3/2) / ((3 / (4 * 3.14159 * density))**(1/2))

def schrodinger_equation(psi, x, t, potential):
    """
    Solve the time-dependent Schrödinger equation
    i * hbar * (d/dt)psi = - (hbar^2 / 2m) * (d^2/dx^2)psi + V(x) * psi
    """
    from sympy import I, symbols, diff
    hbar = HBAR
    m = M_E
    psi_t = I * hbar * diff(psi, t)
    psi_xx = - (hbar**2 / (2 * m)) * diff(psi, x, x)
    return psi_t - (psi_xx + potential * psi)

def redshift_velocity(z):
    """
    Calculate the velocity of an object given its redshift
    v = z * c
    """
    return z * C

def specific_angular_momentum(radius, velocity):
    """
    Calculate the specific angular momentum of an orbiting body
    h = r * v
    """
    return radius * velocity

def radiative_pressure(temperature):
    """
    Calculate the radiative pressure inside a star
    P_rad = (4 * sigma / (3 * c)) * T^4
    """
    return (4 * SIGMA / (3 * C)) * temperature**4

def critical_density(H0):
    """
    Calculate the critical density of the universe
    rho_c = 3 * H0^2 / (8 * pi * G)
    """
    return 3 * H0**2 / (8 * 3.14159 * G)

def dynamical_time(density):
    """
    Calculate the dynamical time of a system
    t_dyn = (3 * pi / (32 * G * rho))^(1/2)
    """
    return (3 * 3.14159 / (32 * G * density))**0.5

def virial_theorem(total_energy):
    """
    Calculate the virial theorem
    2 * K + U = 0
    """
    return 2 * total_energy[0] + total_energy[1]

def stellar_lifetime(mass, luminosity):
    """
    Calculate the lifetime of a star
    tau = (10^10 * M_sun / L_sun) * (M_star / L_star)
    """
    return (10**10 * M_P / L_sun) * (mass / luminosity)

def escape_fraction(luminosity, cross_section, radius):
    """
    Calculate the escape fraction of radiation from a spherical body
    f_esc = L / (4 * pi * R^2 * sigma)
    """
    return luminosity / (4 * 3.14159 * radius**2 * cross_section)

def blackbody_spectrum(wavelength, temperature):
    """
    Calculate the blackbody spectrum
    B_lambda = (2 * h * c^2 / lambda^5) * (1 / (exp(h * c / (lambda * k * T)) - 1))
    """
    from numpy import exp
    return (2 * H * C**2 / wavelength**5) * (1 / (exp(H * C / (wavelength * KB * temperature)) - 1))

def kepler_third_law(semi_major_axis, period):
    """
    Calculate the mass of the central object using Kepler's third law
    M = (4 * pi^2 * a^3) / (G * T^2)
    """
    return (4 * 3.14159**2 * semi_major_axis**3) / (G * period**2)

def mach_number(velocity, speed_of_sound):
    """
    Calculate the Mach number
    M = v / c_s
    """
    return velocity / speed_of_sound

def virial_temperature(mass, radius):
    """
    Calculate the virial temperature of a gas cloud
    T_vir = (G * M * mu) / (5 * k_B * R)
    """
    mu = 0.61  # mean molecular weight for fully ionized hydrogen
    return (G * mass * mu) / (5 * KB * radius)

def boltzmann_factor(energy, temperature):
    """
    Calculate the Boltzmann factor
    f_B = exp(-E / (k_B * T))
    """
    from numpy import exp
    return exp(-energy / (KB * temperature))

def hubble_law_velocity(distance, H0):
    """
    Calculate the velocity of a galaxy using Hubble's law
    v = H0 * d
    """
    return H0 * distance

def gravitational_binding_energy(mass, radius):
    """
    Calculate the gravitational binding energy of a spherical mass
    U = (3 / 5) * (G * M^2 / R)
    """
    return (3 / 5) * (G * mass**2 / radius)

def larmor_radius(charge, mass, velocity, magnetic_field):
    """
    Calculate the Larmor radius of a charged particle in a magnetic field
    r_L = (m * v) / (q * B)
    """
    return (mass * velocity) / (charge * magnetic_field)

def magnetopause_radius(stellar_wind_pressure, magnetic_field):
    """
    Calculate the magnetopause radius
    R_m = (B^2 / (2 * mu_0 * P_wind))^(1/6)
    """
    return (magnetic_field**2 / (2 * MU_0 * stellar_wind_pressure))**(1/6)

def gravitational_wave_strain(distance, chirp_mass):
    """
    Calculate the gravitational wave strain
    h = (4 * (G * M_c)^(5/3)) / (c^4 * D)
    """
    return (4 * (G * chirp_mass)**(5/3)) / (C**4 * distance)

def taylor_series_expansion(f, x, x0, n):
    """
    Calculate the Taylor series expansion of a function f around x0 up to n terms
    f(x) = f(x0) + f'(x0) * (x - x0) + f''(x0) / 2! * (x - x0)^2 + ...
    """
    from sympy import symbols, diff
    x_sym = symbols('x')
    series = f(x_sym).series(x=x_sym, x0=x0, n=n).removeO()
    return series.subs(x_sym, x)

def particle_acceleration(force, mass):
    """
    Calculate the acceleration of a particle given a force
    a = F / m
    """
    return force / mass

def lorentz_force(charge, velocity, magnetic_field):
    """
    Calculate the Lorentz force on a charged particle
    F = q * (v x B)
    """
    import numpy as np
    return charge * np.cross(velocity, magnetic_field)

def gravitational_wave_frequency(distance, chirp_mass):
    """
    Calculate the frequency of gravitational waves
    f_gw = (c^3 / (8 * pi * G * M_c))^(1/2)
    """
    return (C**3 / (8 * 3.14159 * G * chirp_mass))**0.5

def hawking_temperature(mass, time):
    hwk_temp = (HBAR * C**3) / (8 * 3.14159 * G * mass * KB)
    hwk_temp = np.sin(hwk_temp) ** 2 + np.cos(hwk_temp) ** 2 - 1
    hwk_temp += get_current_jsof(time)
    return hwk_temp

def flux_density(luminosity, distance):
    """
    Calculate the flux density
    S = L / (4 * pi * d^2)
    """
    return luminosity / (4 * 3.14159 * distance**2)

def schwarzschild_radius(mass):
    return 2 * G * mass / C**2

def planck_energy(frequency):
    return H * frequency
