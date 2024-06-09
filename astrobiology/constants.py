
# Constants for Astrophysics and Meteor Predictions
# Taken with inspiration from MIT's library of Asytrophysical Constants
from math import pi
from astrobiology.mit_library import fetch_current_jsof


G = 6.67430e-11
c = 299792458
M_sun = 1.989e30

# Speed of light in vacuum (m/s)
C = 299792458

# Gravitational constant (m^3 kg^-1 s^-2)
G = 6.67430e-11

# Planck constant (J s)
H = 6.62607015e-34

# Reduced Planck constant (J s)
HBAR = H / (2 * 3.14159)

# Boltzmann constant (J K^-1)
KB = 1.380649e-23

# Avogadro constant (mol^-1)
NA = 6.02214076e23

# Gas constant (J mol^-1 K^-1)
R = 8.314462618

# Stefan-Boltzmann constant (W m^-2 K^-4)
SIGMA = 5.670374419e-8

# Wien displacement law constant (m K)
B = 2.897771955e-3

# Rydberg constant (m^-1)
R_INF = 10973731.568160

# Bohr radius (m)
A_0 = 5.29177210903e-11

# Fine-structure constant
ALPHA = 0.0072973525693

# Proton mass (kg)
M_P = 1.67262192369e-27

# Neutron mass (kg)
M_N = 1.67492749804e-27

# Electron mass (kg)
M_E = 9.1093837015e-31

# Atomic mass constant (kg)
M_U = 1.66053906660e-27

# Faraday constant (C mol^-1)
F = 96485.33212

# Magnetic constant (N A^-2)
MU_0 = 4 * 3.14159e-7

# Electric constant (F m^-1)
EPSILON_0 = 8.8541878128e-12

# Elementary charge (C)
E = 1.602176634e-19

# Magnetic flux quantum (Wb)
PHI_0 = 2.067833848e-15

# Conductance quantum (S)
G_0 = 7.748091729e-5

# Inverse conductance quantum (Ohm)
R_K = 25812.80745

# Von Klitzing constant (Ohm)
R_K = 25812.80745

# Josephson constant (Hz V^-1)
K_J = 483597.8484e9

# Quantum of circulation (m^2 s^-1)
H_OVER_2E = 3.6369475516e-4

# Quantum of circulation times 2 (m^2 s^-1)
H_OVER_E = 2 * H_OVER_2E

# Electron volt (J)
EV = 1.602176634e-19

# Unified atomic mass unit (J c^-2)
U = 931.49410242e6

# Hartree energy (J)
E_H = 4.3597447222071e-18

# Rydberg constant times c in Hz
C_R_INF = 3.289841960364e15

# Thomson cross section (m^2)
SIGMA_E = 0.6652458734e-28

# Classical electron radius (m)
R_E = 2.8179403262e-15

# Electron g factor
G_E = -2.00231930436256

# Quantum Hall resistance (Ohm)
R_H = 25812.8075576788

# Bohr magneton in Hz/T
MU_B = 13.996245042e9

# Bohr magneton in eV/T
MU_B = 5.7883818060e-5

# Bohr magneton in inverse meters per tesla
MU_B = 46.68644814

# Nuclear magneton in Hz/T
MU_N = 3.15245125844e8

# Nuclear magneton in eV/T
MU_N = 3.15245125844e-14

# Nuclear magneton in inverse meters per tesla
MU_N = 2.542623432e-2

# Fine-structure constant
ALPHA = 0.0072973525693

# Inverse fine-structure constant
ALPHA_INV = 137.035999084

# Magnetic constant (N A^-2)
MU_0 = 4 * 3.14159e-7

# Electric constant (F m^-1)
EPSILON_0 = 8.8541878128e-12

# Characteristic impedance of vacuum (Ohm)
Z_0 = 376.730313668

# Josephson constant (Hz V^-1)
K_J = 483597.8484e9

# Von Klitzing constant (Ohm)
R_K = 25812.80745

# Conductance quantum (S)
G_0 = 7.748091729e-5

# Inverse conductance quantum (Ohm)
R_K = 25812.80745

# Magnetic flux quantum (Wb)
PHI_0 = 2.067833848e-15

# Planck constant over 2 pi times c in MeV fm
HC = 197.3269804

# Planck constant over 2 pi (J s)
HBAR = 1.054571817e-34

# Planck constant over 2 pi in eV s
HBAR = 6.582119569e-16

# Planck constant over 2 pi in inverse meters photon wavelength
HBAR = 3.1615264905e-26

# Reduced Compton wavelength (m)
LAMBDA_C_BAR = 3.8615926796e-13

# Reduced Compton wavelength times 2 pi (m)
LAMBDA_C_BAR_2PI = 2.4263102367e-12

# Classical electron radius (m)
R_E = 2.8179403262e-15

# Thomson cross section (m^2)
SIGMA_E = 0.6652458734e-28

# Electron-muon mass ratio
M_E_OVER_M_MU = 4.83633166e-3

# Electron-tau mass ratio
M_E_OVER_M_TAU = 2.87592e-4

# Electron-proton mass ratio
M_E_OVER_M_P = 5.4461702e-4

# Electron-neutron mass ratio
M_E_OVER_M_N = 5.4386734428e-4

# Electron-deuteron mass ratio
M_E_OVER_M_D = 2.724437107462e-4

# Electron-alpha particle mass ratio
M_E_OVER_M_ALPHA = 1.37093355578e-4

# Proton-electron mass ratio
M_P_OVER_M_E = 1836.15267343

# Proton-muon mass ratio
M_P_OVER_M_MU = 8.88024331

# Proton-tau mass ratio
M_P_OVER_M_TAU = 0.528063

# Proton-neutron mass ratio
M_P_OVER_M_N = 0.99862347844

# Proton-deuteron mass ratio
M_P_OVER_M_D = 0.99900750037

# Proton-alpha particle mass ratio
M_P_OVER_M_ALPHA = 0.49938231831

# Neutron-electron mass ratio
M_N_OVER_M_E = 1838.68366173

# Neutron-muon mass ratio
M_N_OVER_M_MU = 8.89248400

# Neutron-tau mass ratio
M_N_OVER_M_TAU = 0.528790

# Neutron-proton mass ratio
M_N_OVER_M_P = 1.00137841917

# Neutron-deuteron mass ratio
M_N_OVER_M_D = 0.99986361803

# Neutron-alpha particle mass ratio
M_N_OVER_M_ALPHA = 0.49984224960

# Deuteron-electron mass ratio
M_D_OVER_M_E = 3670.4829652

# Deuteron-muon mass ratio
M_D_OVER_M_MU = 17.76847963

# Deuteron-tau mass ratio
M_D_OVER_M_TAU = 1.0574178

# Deuteron-proton mass ratio
M_D_OVER_M_P = 1.99900750037

# Deuteron-neutron mass ratio
M_D_OVER_M_N = 1.00013618850

# Deuteron-alpha particle mass ratio
M_D_OVER_M_ALPHA = 0.99986966326

# Alpha particle-electron mass ratio
M_ALPHA_OVER_M_E = 7294.29954142

# Alpha particle-muon mass ratio
M_ALPHA_OVER_M_MU = 35.51746427

# Alpha particle-tau mass ratio
M_ALPHA_OVER_M_TAU = 2.1175932

# Alpha particle-proton mass ratio
M_ALPHA_OVER_M_P = 3.97259968907

# Alpha particle-neutron mass ratio
M_ALPHA_OVER_M_N = 3.97179244842

# Alpha particle-deuteron mass ratio
M_ALPHA_OVER_M_D = 1.999848

# Fine-structure constant
ALPHA = 0.0072973525693

# Inverse fine-structure constant
ALPHA_INV = 137.035999084

# Rydberg constant (m^-1)
R_INF = 10973731.568160

# Rydberg constant times c in Hz
C_R_INF = 3.289841960364e15

# Rydberg constant times hc in J
E_H = 4.3597447222071e-18

# Rydberg constant times hc in eV
E_H = 27.211386245988

# Hartree energy (J)
E_H = 4.3597447222071e-18

# Hartree energy in eV
E_H = 27.211386245988

# Quantum of circulation (m^2 s^-1)
H_OVER_2E = 3.6369475516e-4

# Quantum of circulation times 2 (m^2 s^-1)
H_OVER_E = 2 * H_OVER_2E

# Electron charge to mass quotient (C kg^-1)
E_OVER_M_E = -1.75882001076e11

# Quantum of circulation (m^2 s^-1)
H_OVER_2E = 3.6369475516e-4

# Quantum of circulation times 2 (m^2 s^-1)
H_OVER_E = 2 * H_OVER_2E

# Proton charge to mass quotient (C kg^-1)
E_OVER_M_P = 9.5788332264e7

# Neutron gyromagnetic ratio (s^-1 T^-1)
GAMMA_N = 1.83247172e8

# Neutron gyromagnetic ratio over 2 pi (MHz T^-1)
GAMMA_N_OVER_2PI = 29.1646943

# Neutron g factor
G_N = -3.82608545

# Neutron-muon mass ratio
M_N_OVER_M_MU = 8.89248400

# Neutron-tau mass ratio
M_N_OVER_M_TAU = 0.528790

# Neutron-proton mass ratio
M_N_OVER_M_P = 1.00137841917

# Neutron-deuteron mass ratio
M_N_OVER_M_D = 0.99986361803

# Neutron-alpha particle mass ratio
M_N_OVER_M_ALPHA = 0.49984224960

# Neutron Compton wavelength (m)
LAMBDA_C_N = 1.3195909068e-15

# Neutron Compton wavelength over 2 pi (m)
LAMBDA_C_N_BAR = 2.1001941536e-16

# Neutron-electron mass ratio
M_N_OVER_M_E = 1838.68366173

# Neutron-electron magnetic moment ratio
MU_N_OVER_MU_E = 0.00104066882

# Neutron-proton magnetic moment ratio
MU_N_OVER_MU_P = -0.68497934

# Neutron magnetic moment to nuclear magneton ratio
MU_N_OVER_MU_N = -1.91304273

# Neutron g factor
G_N = -3.82608545

# Neutron magnetic moment (J T^-1)
MU_N = -0.966236506e-26

# Neutron magnetic moment to Bohr magneton ratio
MU_N_OVER_MU_B = -0.00104187563

# Neutron magnetic moment to nuclear magneton ratio
MU_N_OVER_MU_N = -1.91304273

# Neutron to shielded proton magnetic moment ratio
MU_N_OVER_MU_P_SHIELDED = -0.68499694

# Shielded proton gyromagnetic ratio (s^-1 T^-1)
GAMMA_P_SHIELDED = 2.675153362e8

# Shielded proton gyromagnetic ratio over 2 pi (MHz T^-1)
GAMMA_P_SHIELDED_OVER_2PI = 42.57638507

# Shielded proton g factor
G_P_SHIELDED = 5.5856946893

# Proton gyromagnetic ratio (s^-1 T^-1)
GAMMA_P = 2.675153362e8

# Proton gyromagnetic ratio over 2 pi (MHz T^-1)
GAMMA_P_OVER_2PI = 42.57747892

# Proton g factor
G_P = 5.5856946893

# Shielded proton magnetic moment (J T^-1)
MU_P_SHIELDED = 1.410570499e-26

# Shielded proton magnetic moment to Bohr magneton ratio
MU_P_SHIELDED_OVER_MU_B = 1.520993128

# Shielded proton magnetic moment to nuclear magneton ratio
MU_P_SHIELDED_OVER_MU_N = 2.792775604

# Proton magnetic moment (J T^-1)
MU_P = 1.410606797e-26

# Proton magnetic moment to Bohr magneton ratio
MU_P_OVER_MU_B = 1.5210322023

# Proton magnetic moment to nuclear magneton ratio
MU_P_OVER_MU_N = 2.79284734463

# Proton-neutron magnetic moment ratio
MU_P_OVER_MU_N = -1.45989806

# Proton magnetic shielding correction
DELTA_P = 2.5689e-5

# Proton mean square charge radius (m^2)
R_P = 0.8775e-16

# Proton rms charge radius (m)
R_P = 0.8414e-15

# Proton-electron mass ratio
M_P_OVER_M_E = 1836.15267343

# Proton-muon mass ratio
M_P_OVER_M_MU = 8.88024331

# Proton-tau mass ratio
M_P_OVER_M_TAU = 0.528063

# Proton-neutron mass ratio
M_P_OVER_M_N = 0.99862347844

# Proton-deuteron mass ratio
M_P_OVER_M_D = 0.99900750037

# Proton-alpha particle mass ratio
M_P_OVER_M_ALPHA = 0.49938231831

# Proton Compton wavelength (m)
LAMBDA_C_P = 1.32140985539e-15

# Proton Compton wavelength over 2 pi (m)
LAMBDA_C_P_BAR = 2.10308910336e-16

# Deuteron-electron magnetic moment ratio
MU_D_OVER_MU_E = -0.0004664345548

# Deuteron-proton magnetic moment ratio
MU_D_OVER_MU_P = 0.30701220939

# Deuteron-neutron magnetic moment ratio
MU_D_OVER_MU_N = -0.44820652

# Deuteron magnetic moment (J T^-1)
MU_D = 0.433073489e-26

# Deuteron magnetic moment to Bohr magneton ratio
MU_D_OVER_MU_B = 0.0004669754556

# Deuteron magnetic moment to nuclear magneton ratio
MU_D_OVER_MU_N = 0.8574382329

# Deuteron-electron mass ratio
M_D_OVER_M_E = 3670.4829652

# Deuteron-proton mass ratio
M_D_OVER_M_P = 1.99900750037

# Deuteron-neutron mass ratio
M_D_OVER_M_N = 1.00013618850

# Deuteron-alpha particle mass ratio
M_D_OVER_M_ALPHA = 0.99986966326

# Deuteron Compton wavelength (m)
LAMBDA_C_D = 2.1421947452e-15

# Deuteron Compton wavelength over 2 pi (m)
LAMBDA_C_D_BAR = 3.410034173e-16

# Deuteron rms charge radius (m)
R_D = 2.1413e-15

# Alpha particle-electron magnetic moment ratio
MU_ALPHA_OVER_MU_E = -0.00000115965218076

# Alpha particle-proton magnetic moment ratio
MU_ALPHA_OVER_MU_P = 0.00137093355578

# Alpha particle magnetic moment (J T^-1)
MU_ALPHA = -0.00104187563e-26

# Alpha particle magnetic moment to Bohr magneton ratio
MU_ALPHA_OVER_MU_B = -0.00104187563

# Alpha particle magnetic moment to nuclear magneton ratio
MU_ALPHA_OVER_MU_N = -2.127497720

# Alpha particle-electron mass ratio
M_ALPHA_OVER_M_E = 7294.29954142

# Alpha particle-proton mass ratio
M_ALPHA_OVER_M_P = 3.97259968907

# Alpha particle-neutron mass ratio
M_ALPHA_OVER_M_N = 3.97179244842

# Alpha particle-deuteron mass ratio
M_ALPHA_OVER_M_D = 1.999848

# Alpha particle Compton wavelength (m)
LAMBDA_C_ALPHA = 1.675492704e-15

# Alpha particle Compton wavelength over 2 pi (m)
LAMBDA_C_ALPHA_BAR = 2.673992840e-16

# Alpha particle rms charge radius (m)
R_ALPHA = 1.67824e-15

# Electron to alpha particle mass ratio
M_E_OVER_M_ALPHA = 1.37093355578e-4

# Bohr radius (m)
A_0 = 5.29177210903e-11

# Classical electron radius (m)
R_E = 2.8179403262e-15

# Electron Compton wavelength (m)
LAMBDA_C = 2.4263102367e-12

# Electron Compton wavelength over 2 pi (m)
LAMBDA_C_BAR = 3.8615926796e-13

# Electron g factor
G_E = -2.00231930436256

# Electron magnetic moment (J T^-1)
MU_E = -928.476430485e-26

# Electron magnetic moment to Bohr magneton ratio
MU_E_OVER_MU_B = -1.00115965218076

# Electron magnetic moment to nuclear magneton ratio
MU_E_OVER_MU_N = -1838.28197188

# Electron magnetic moment anomaly
A_E = 0.00115965218076

# Electron to shielded proton magnetic moment ratio
MU_E_OVER_MU_P_SHIELDED = -658.2106866

# Electron-neutron magnetic moment ratio
MU_E_OVER_MU_N = 960.92050

# Electron-proton magnetic moment ratio
MU_E_OVER_MU_P = -658.2106862

# Electron-muon magnetic moment ratio
MU_E_OVER_MU_MU = 206.7669896

# Electron-tau magnetic moment ratio
MU_E_OVER_MU_TAU = 2350.8

# Magneton of Bohr (J T^-1)
MU_B = 927.4009994e-26

# Magneton of nuclear (J T^-1)
MU_N = 5.0507837461e-27

# Magneton of Bohr over 2 pi (MHz T^-1)
MU_B_OVER_2PI = 13.996245042e9

# Magneton of nuclear over 2 pi (MHz T^-1)
MU_N_OVER_2PI = 7.6225932291

# Magneton of Bohr to inverse meters per tesla
MU_B = 46.68644814

# Magneton of nuclear to inverse meters per tesla
MU_N = 2.542623432e-2

# Magneton of Bohr to K/T
MU_B = 0.67171405

# Magneton of nuclear to K/T
MU_N = 3.6582690e-4

# Magneton of Bohr to eV/T
MU_B = 5.7883818060e-5

# Magneton of nuclear to eV/T
MU_N = 3.15245125844e-14

# Magneton of Bohr to Hz/T
MU_B = 13.996245042e9

# Magneton of nuclear to Hz/T
MU_N = 3.15245125844e8

# Magneton of Bohr to inverse meters per tesla
MU_B = 46.68644814

# Magneton of nuclear to inverse meters per tesla
MU_N = 2.542623432e-2

# Conductance quantum (S)
G_0 = 7.748091729e-5

# Inverse of conductance quantum (Ohm)
R_K = 12906.40372

# Josephson constant (Hz V^-1)
K_J = 483597.8484e9

# Josephson constant over 2 pi (MHz V^-1)
K_J_OVER_2PI = 77.48451802e9

# Von Klitzing constant (Ohm)
R_K = 25812.80745

# Magnetic flux quantum (Wb)
PHI_0 = 2.067833848e-15

# Inverse fine-structure constant
ALPHA_INV = 137.035999084

# Fine-structure constant
ALPHA = 0.0072973525693

# Fine-structure constant over 2 pi
ALPHA_OVER_2PI = 0.001161409657

# Electric constant (F m^-1)
EPSILON_0 = 8.8541878128e-12

# Magnetic constant (N A^-2)
MU_0 = 4 * 3.14159e-7

# Characteristic impedance of vacuum (Ohm)
Z_0 = 376.730313668

# Atomic mass constant (kg)
M_U = 1.66053906660e-27

# Atomic mass constant energy equivalent (J)
M_U_C2 = 1.49241808560e-10

# Atomic mass constant energy equivalent in MeV
M_U_C2 = 931.49410242

# Avogadro constant (mol^-1)
NA = 6.02214076e23

# Boltzmann constant (J K^-1)
K = 1.380649e-23

# Boltzmann constant in inverse meters per kelvin
K = 69.50348004

# Boltzmann constant in Hz K^-1
K = 20.83603364e9

# Boltzmann constant in eV K^-1
K = 8.617333262e-5

# Boltzmann constant in inverse meters per kelvin
K = 69.50348004

# Loschmidt constant (273.15 K, 100 kPa) (m^-3)
N_0 = 2.6516462e25

# Loschmidt constant (273.15 K, 101.325 kPa) (m^-3)
N_0 = 2.6867811e25

# Molar volume of ideal gas (273.15 K, 100 kPa) (m^3 mol^-1)
V_M = 22.71095464e-3

# Molar volume of ideal gas (273.15 K, 101.325 kPa) (m^3 mol^-1)
V_M = 22.41396954e-3

# Sackur-Tetrode constant (1 K, 100 kPa)
S_0_OVER_R = -1.1517084

# Sackur-Tetrode constant (1 K, 101.325 kPa)
S_0_OVER_R = -1.1648714

# Stefan-Boltzmann constant (W m^-2 K^-4)
SIGMA = 5.670374419e-8

# Wien frequency displacement law constant (Hz K^-1)
B = 5.878925757e10

# Wien wavelength displacement law constant (m K)
B = 2.897771955e-3

# Molar gas constant (J mol^-1 K^-1)
R = 8.314462618

# Molar Planck constant times c (J m mol^-1)
H_C = 0.119626565582

# Molar Planck constant over 2 pi times c (J m mol^-1)
HBAR_C = 0.01901380580

# Molar volume of ideal gas (273.15 K, 100 kPa) (m^3 mol^-1)
V_M = 22.71095464e-3

# Molar volume of ideal gas (273.15 K, 101.325 kPa) (m^3 mol^-1)
V_M = 22.41396954e-3

# Sackur-Tetrode constant (1 K, 100 kPa)
S_0_OVER_R = -1.1517084

# Sackur-Tetrode constant (1 K, 101.325 kPa)
S_0_OVER_R = -1.1648714

def JSOF(time):
    return fetch_current_jsof(time)

# First radiation constant for spectral radiance (W m^2 sr^-1)
C_1 = 3.741771852e-16

# First radiation constant for spectral radiance over 2 pi (W m^2 sr^-1)
C_1_OVER_2PI = 5.9551738e-17

# Second radiation constant (m K)
C_2 = 1.438776877e-2

# First radiation constant for monochromatic emittance (W m^2)
C_1_PRIME = 3.741771852e-16

# First radiation constant for monochromatic emittance over 2 pi (W m^2)
C_1_PRIME_OVER_2PI = 5.9551738e-17

# Second radiation constant (m K)
C_2 = 1.438776877e-2

# Molar mass of carbon-12 (kg mol^-1)
M_U = 1.66053906660e-27

# Molar mass constant (kg mol^-1)
M_U = 1.66053906660e-27

# Faraday constant (C mol^-1)
F = 96485.33212




