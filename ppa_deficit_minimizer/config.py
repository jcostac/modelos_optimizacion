
# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
# PATHS
# ------------------------------------------------------------------------------
INPUTS_PATH = 'inputs'
OUTPUTS_PATH = 'outputs'
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# BATTERY PARAMETERS
# ------------------------------------------------------------------------------
CAPACITY = 40  # MWh
MAX_SOC_PERCENT = 1.0  # Percentage of total capacity (e.g., 1.0 for 100%)
MIN_SOC_PERCENT = 0.0  # Percentage of total capacity (e.g., 0.1 for 10%)
MAX_CHARGE_POWER = 10  # MW
MAX_DISCHARGE_POWER = 10  # MW

# Round-trip efficiency is charge_efficiency * discharge_efficiency.
# For 85% round-trip, sqrt(0.85) = 0.922. Set both to 0.922.
CHARGE_EFFICIENCY = 0.92  # As a decimal
DISCHARGE_EFFICIENCY = 0.92  # As a decimal
INITIAL_ENERGY_MWH = 0  # Starting energy in the battery [MWh]

# ------------------------------------------------------------------------------
# GRID AND PPA PARAMETERS
# ------------------------------------------------------------------------------
CAPACITY_POI = 43.2  # Max grid injection limit [MW]
TRANSFORMER_LOSSES = 0.004 # As a decimal (e.g., 0.4% is 0.004). Set to 0 or None for no losses.

# PPA Profile Type Selection - Choose exactly ONE of the following:
GENERATE_FLAT_PPA_PROFILE = False  # Generate constant baseload profiles
GENERATE_SEASONAL_PPA_PROFILE = True  # Generate seasonal profiles from JSON
USE_PRELOADED_PPA_PROFILES = False # Use pre-existing CSV profiles from inputs folder

# ------------------------------------------------------------------------------
# DEGRADATION PARAMETERS
# ------------------------------------------------------------------------------
# If using a degradation profile from 'battery_degradation_profile.csv', set this to None.
# Otherwise, provide an annual degradation factor as a decimal (e.g., 0.02 for 2%).
BATTERY_DEGRADATION_FACTOR = None

# ------------------------------------------------------------------------------
# EXECUTION PARAMETERS
# ------------------------------------------------------------------------------
# List of baseload scenarios to run in MW, if GENERATE_FLAT_PPA_PROFILE is True.
BASELOAD_MW_LIST = [5,10]

# List of seasonal PPA scenarios to run, if GENERATE_SEASONAL_PPA_PROFILE is True.
SEASONAL_PPA_JSON_LIST = ['ppa_input_1.json', "ppa_input_2.json", "ppa_input_3.json", "ppa_input_4.json", "ppa_input_5.json", "ppa_input_6.json"] #list of json files with the same structure as ppa_input_example.json

#default is 8-20 which are solar hours 
PEAK_START_HOUR = 8 #hour of the day to start the peak period (inclusive)
PEAK_END_HOUR = 20 #hour of the day to end the peak period (NOT inclusive)

# List of preloaded PPA CSV files to run scenarios for, if USE_PRELOADED_PPA_PROFILES is True.
# These files should be located in the inputs folder and follow the same format as other CSV profiles.
PRELOADED_PPA_PROFILES_LIST = ['ppa_profile.csv']


# If True, consolidates all hourly CSV results into a single Excel file.
CONSOLIDATE_EXCEL = True

# Optional start and end dates for filtering input data (format 'YYYY-MM-DD').
# Set to None to use all available data.
START_DATE = None
END_DATE = None

# If True, prints detailed progress and summary information during the run.
VERBOSE = False

