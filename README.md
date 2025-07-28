# PPA Deficit Minimizer (`v2_modelo_ppa_deficit.py`)

This script optimizes the operation of a hybrid power plant, consisting of wind and solar generation sources along with a battery energy storage system (BESS). The primary goal is to minimize the deficit against a Power Purchase Agreement (PPA) target, while a secondary goal is to maximize the total energy injected into the grid.

## Model Overview

The optimization is a two-phase linear programming problem:

1.  **Phase 1: Minimize PPA Deficit**: The model first finds the minimum possible total energy deficit (unmet PPA) over the entire simulation period.
2.  **Phase 2: Maximize Grid Injection**: With the total deficit fixed at the optimal value from Phase 1, the model then maximizes the total energy delivered to the grid. This ensures that any excess generation is used to charge the battery or sold to the grid, rather than being unnecessarily curtailed, without worsening the PPA deficit.

### System Parameters

The following system parameters are currently hardcoded within the `main_baseload` function. You can modify them there to suit different simulation scenarios.

**Battery Storage (BESS) :**
*   **Total Capacity:** 40 MWh
*   **Maximum State of Charge (SoC):** 95% (38 MWh)
*   **Minimum State of Charge (SoC):** 5% (2 MWh)
*   **Max Charge/Discharge Power:** 9 MW
*   **Round-trip Efficiency:** 90% (modeled as `sqrt(0.9)` for charge and discharge efficiencies)
*   **Initial Energy:** 0 MWh

**Grid Connection:**
*   **Point of Interconnection (POI) Capacity:** 43.2 MW

## Inputs

The model requires time-series data for generation profiles. By default, it looks for these files in the `inputs/` directory.

### Required Files

The script requires at least one of the following generation profiles. If one is missing, it will assume zero generation from that source.

*   `inputs/wind_profile.csv`
*   `inputs/solar_profile.csv`

### Optional File

*   `inputs/ppa_profile.csv`: This file is only used if you decide not to generate a flat baseload PPA profile within the script.

### File Format

All input CSV files must follow this format:
*   **Column 1:** Datetime information (e.g., `YYYY-MM-DD HH:MM:SS`)
*   **Column 2:** Power value in Megawatts (MW)

Example `wind_profile.csv`:
```csv
datetime,MW
2023-01-01 00:00:00,15.3
2023-01-01 01:00:00,16.1
2023-01-01 02:00:00,14.8
...
```

## Outputs

The script generates several output files in the `outputs/` directory for each PPA baseload scenario it runs. The filenames are suffixed with the baseload value (e.g., `_15MW`).

*   `consolidated_results_hourly_{baseload}MW.csv`: A detailed hourly timeseries of all model variables, including generation, PPA targets, battery state, grid injection, curtailment, and deficits.
*   `consolidated_results_monthly_{baseload}MW.csv`: A summary of the PPA deficit percentage for each month.
*   `results_summary_{baseload}MW.csv`: High-level summary of the optimization results, including total PPA deficit and total grid injection over the period.

### Consolidated Excel File

*   `consolidated_baseload_results.xlsx`: If enabled, the script will gather all the `consolidated_results_hourly_...csv` files and compile them into a single Excel workbook, with each scenario in a separate sheet.

## How to Run

To execute the model, configure the parameters within the `if __name__ == "__main__":` block at the end of the script and run it from your terminal.

```python
if __name__ == "__main__":
    main_baseload(
        baseload_mw_list=[5, 10, 15], 
        consolidate_excel=True, 
        start_date='2000-01-01', 
        end_date='2000-12-31', 
        verbose=False
    )
```

### `main_baseload` Function Parameters

*   `baseload_mw_list` (list[float]): A list of flat, constant PPA targets (in MW) to simulate. The script will run once for each value in the list.
*   `consolidate_excel` (bool): If `True`, the script will create the consolidated Excel file after all scenarios are complete (usefule for multiple scenarios)
*   `start_date` (str | None): The start date for filtering the input data. Format: `'YYYY-MM-DD'`. If `None`, we use the entire dataset time period
*   `end_date` (str | None): The end date for filtering the input data. Format: `'YYYY-MM-DD'`. If `None`, we use the entire dataset time period
*   `verbose` (bool): If `True`, the script will print detailed logs, model parameters, and optimization progress. Recommended for debugging or single runs. Set to `False` for faster batch processing.
*   `generate_ppa_profile` (bool): This is set to `True` inside `main_baseload`. This tells the script to generate a flat PPA profile based on the values in `baseload_mw_list`. If you set it to `False`, the script will attempt to load `inputs/ppa_profile.csv` instead.

## Prerequisites

Ensure you have the required Python packages installed:
```sh
pip install -r requirements.txt
```

The model also requires a linear programming solver. It will automatically try to find one of the following: **HiGHS**, **GLPK**, or **CBC**. Please ensure at least one of these is installed and accessible in your environment.