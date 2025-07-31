
# ==============================================================================
# SCRIPT CONFIGURATION
# ==============================================================================
# PATHS
# ------------------------------------------------------------------------------
INPUTS_PATH = 'ppa_deficit_minimizer/inputs'
OUTPUTS_PATH = 'ppa_deficit_minimizer/outputs'
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

# If True, a baseload PPA profile will be generated.
# If False, a 'ppa_profile.csv' must be present in the inputs folder.
GENERATE_PPA_PROFILE = True

# ------------------------------------------------------------------------------
# DEGRADATION PARAMETERS
# ------------------------------------------------------------------------------
# If using a degradation profile from 'battery_degradation_profile.csv', set this to None.
# Otherwise, provide an annual degradation factor as a decimal (e.g., 0.02 for 2%).
BATTERY_DEGRADATION_FACTOR = None

# ------------------------------------------------------------------------------
# EXECUTION PARAMETERS
# ------------------------------------------------------------------------------
# List of baseload scenarios to run in MW.
BASELOAD_MW_LIST = [5]

# If True, consolidates all hourly results into a single Excel file.
CONSOLIDATE_EXCEL = True

# Optional start and end dates for filtering input data (format 'YYYY-MM-DD').
# Set to None to use all available data.
START_DATE = '2004-01-01'
END_DATE = '2004-12-31'

# If True, prints detailed progress and summary information during the run.
VERBOSE = False

