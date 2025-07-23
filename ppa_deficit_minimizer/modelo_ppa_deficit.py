import pulp
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import datetime
import pretty_errors
import glob
import os
import openpyxl

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import INPUTS_PATH, OUTPUTS_PATH

class PPADeficitMinimizer:
    def __init__(self, max_capacity, min_capacity, max_charge_power, max_discharge_power, charge_efficiency, discharge_efficiency, initial_energy, capacity_poi):
        """
        Initialize PPA deficit minimizer with battery parameters.
        
        Args:
            max_capacity (float): Maximum storage capacity [MWh]
            min_capacity (float): Minimum storage capacity [MWh]
            max_charge_power (float): Maximum charge rate [MW]
            max_discharge_power (float): Maximum discharge rate [MW]
            charge_efficiency (float): Charge efficiency
            discharge_efficiency (float): Discharge efficiency
            initial_energy (float): Initial state-of-charge [MWh]
        """
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.max_charge_power = max_charge_power
        self.max_discharge_power = max_discharge_power
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.initial_energy = initial_energy
        self.capacity_poi = capacity_poi

    def optimize_with_surplus_deficit(self, wind: list[float], solar: list[float], ppa: list[float]) -> dict:
        """
        Optimize battery charge/discharge to minimize deviation from PPA.
        
        Args:
            wind: List of wind production [MW] per hour
            solar: List of solar production [MW] per hour
            ppa: List of PPA targets [MW] per hour
            
        Returns:
            dict: Optimization results including charge, discharge, state_of_charge, surplus, and deficit.
        """
        T = len(ppa)
        assert len(wind) == T and len(solar) == T, "Input series must have same length"

        model = pulp.LpProblem("PPA_Deficit_Minimization", pulp.LpMinimize)

        # Decision variables
        charge = pulp.LpVariable.dicts("charge", range(T), lowBound=0, upBound=self.max_charge_power)
        discharge = pulp.LpVariable.dicts("discharge", range(T), lowBound=0, upBound=self.max_discharge_power)
        state_of_charge = pulp.LpVariable.dicts("state_of_charge", range(T), lowBound=0, upBound=self.max_capacity)  # Changed lowBound to 0
        surplus = pulp.LpVariable.dicts("surplus", range(T), lowBound=0)
        deficit = pulp.LpVariable.dicts("deficit", range(T), lowBound=0)

        # Objective: Minimize total absolute deviation
        model += pulp.lpSum(surplus[t] + deficit[t] for t in range(T))

        # Constraints for the entire optimization period
        for t in range(T):
            supply = wind[t] + solar[t] + discharge[t] - charge[t]
            model += (supply - ppa[t]) == (surplus[t] - deficit[t]), f"supply_deficit_balance_{t}"

        # Energy balance at the beginning of the optimization
        model += (state_of_charge[0] == self.initial_energy + self.charge_efficiency * charge[0] - (1 / self.discharge_efficiency) * discharge[0]), f"energy_balance_t0"
        # Energy balance for all hours except the first one
        for t in range(1, T): 
            model += (state_of_charge[t] == state_of_charge[t-1] + self.charge_efficiency * charge[t] - (1 / self.discharge_efficiency) * discharge[t]), f"energy_balance_t{t}"
            model += (state_of_charge[t] >= self.min_capacity), f"min_capacity_constraint_t{t}"

        # Solve model
        model.solve()

        # Extract results
        results = {
            'charge': [charge[t].value() for t in range(T)],
            'discharge': [discharge[t].value() for t in range(T)],
            'state_of_charge': [state_of_charge[t].value() for t in range(T)],
            'surplus': [surplus[t].value() for t in range(T)],
            'deficit': [deficit[t].value() for t in range(T)],
            'status': pulp.LpStatus[model.status],
            'total_deviation': pulp.value(model.objective)
        }
        return {'results': results}

    def compute_deficits(self, results: dict, ppa: list[float], timestamps: list[datetime.datetime] = None) -> dict:
        """
        Compute hourly and monthly % deficits.
        
        Args:
            results: Output from optimize()
            ppa: PPA targets
            timestamps: Optional list of datetimes for monthly grouping
            
        Returns:
            dict: Hourly deficits and monthly deficits in percentage %
        """
        T = len(ppa)
        deficit_values = results['deficit']

        # Hourly deficits 
        hourly = [(deficit_values[t] / ppa[t]) * 100 if ppa[t] > 0 else 0 for t in range(T)]

        if timestamps is None:
            return {'hourly': hourly, 'monthly': 'Timestamps required for monthly computation'}
        else: 
            timestamps = pd.to_datetime(timestamps)
        
        # Monthly deficits
        df = pd.DataFrame({
            'timestamp': timestamps,
            'deficit': deficit_values,
            'ppa': ppa
        })
        df['month'] = df['timestamp'].dt.to_period('M')
        monthly = df.groupby('month').agg({'deficit': 'sum', 'ppa': 'sum'})
        monthly['%_deficit'] = monthly.apply(lambda row: (row['deficit'] / row['ppa']) * 100 if row['ppa'] > 0 else 0, axis=1)
        
        return {'deficits': {'hourly': hourly, 'monthly': monthly['%_deficit'].to_dict()}}

    def load_data(self, generate_ppa_profile: bool = False, baseload_mw: float = None):
        """
        Load data from the inputs folder.
        Assumes CSVs have a datetime column as the first column and a value column as the second.

        Args:
            generate_ppa_profile (bool): Whether to generate a baseload profile or use a specific PPA profile
            baseload_mw (float): Constant baseload value in MW
        """
        PATHS = {
                'wind_profile': f'{INPUTS_PATH}/wind_profile.csv',
                'solar_profile': f'{INPUTS_PATH}/solar_profile.csv',
                'ppa_profile': f'{INPUTS_PATH}/ppa_profile.csv',
            }

        
        wind_df = pd.read_csv(PATHS['wind_profile'])
        solar_df = pd.read_csv(PATHS['solar_profile'])

        self.wind_profile = wind_df.iloc[:, 1].tolist()
        self.solar_profile = solar_df.iloc[:, 1].tolist()

        if generate_ppa_profile:
            self.ppa_profile = self.create_baseload_ppa(baseload_mw)
        else:
            ppa_df = pd.read_csv(PATHS['ppa_profile'])
            self.ppa_profile = ppa_df.iloc[:, 1].tolist()

        # Check if wind and solar profiles have the same length
        length_wind = len(self.wind_profile)
        length_solar = len(self.solar_profile)
        length_ppa = len(self.ppa_profile)  
        if length_wind != length_solar or length_wind != length_ppa or length_solar != length_ppa:
            print(f"Length of wind profile: {length_wind}")
            print(f"Length of solar profile: {length_solar}")
            print(f"Length of PPA profile: {length_ppa}")
            raise ValueError("Wind, solar, and PPA profiles must have the same length. Check the input files.")

        else: 
            self.timestamps = pd.to_datetime(wind_df.iloc[:, 0]).tolist()
            
    def create_baseload_ppa(self, baseload_mw):
        """
        Create a baseload PPA profile with constant value.
        
        Args:
            baseload_mw (float): Constant baseload value in MW
            
        Returns:
            list: PPA profile with constant baseload value
            
        Note:
            This method requires wind and solar profiles to be loaded first
            to determine the correct length of the PPA profile.
        """
        if not hasattr(self, 'wind_profile') or not hasattr(self, 'solar_profile'):
            raise ValueError("Wind and solar profiles must be loaded first. Call load_data() before creating baseload PPA.")

        if baseload_mw is None:
            raise ValueError("Baseload value must be provided. Call load_data() with baseload_mw parameter, and a generate_ppa_profile=True flag.")
        
        # Use wind profile length to determine the size (could also use solar_profile)
        profile_length = len(self.wind_profile)
        
        # Create baseload PPA profile
        baseload_ppa = [baseload_mw] * profile_length
            
        return baseload_ppa

    def save_results(self, results: dict, deficits: dict, save_as_csv: bool = True, save_as_excel: bool = False):
        """
        Save results and deficits to the outputs folder.
        - A consolidated hourly time-series is saved to consolidated_results_hourly.csv.
        - A consolidated monthly summary is saved to consolidated_results_monthly.csv.
        - Summary data from results is saved to results_summary.csv.
        """
        results_data = results.get('results', {})
        deficits_data = deficits.get('deficits', {})

        # Prepare time-series data for hourly results
        df_hourly_data = {
            'timestamp': self.timestamps,
            'wind': self.wind_profile,
            'solar_pv': self.solar_profile,
            'ppa_profile': self.ppa_profile
        }

        # Add results to the dataframe
        for key, value in results_data.items():
            if isinstance(value, list):
                df_hourly_data[key] = value
        
        if 'hourly' in deficits_data:
            df_hourly_data['%_deficit'] = deficits_data['hourly']

        df_hourly = pd.DataFrame(df_hourly_data)

        # Rename columns to be more user-friendly
        df_hourly = df_hourly.rename(columns={
            'state_of_charge': 'soc',
        })

        df_hourly['net_charge'] = df_hourly['charge'] - df_hourly['discharge']

        # Select and order columns for the final hourly CSV
        final_hourly_cols = [
            'timestamp', 'wind', 'solar_pv', 'ppa_profile', 'soc',  'charge', 'discharge',
            'net_charge', 'deficit', '%_deficit'
        ]
        # Filter to only include columns that exist
        final_hourly_cols = [col for col in final_hourly_cols if col in df_hourly.columns]
        df_hourly_final = df_hourly[final_hourly_cols].copy()

        # Round numeric columns to 4 decimal places
        for col in df_hourly_final.columns:
            if pd.api.types.is_numeric_dtype(df_hourly_final[col]):
                df_hourly_final.loc[:, col] = df_hourly_final[col].round(4)
        
        df_hourly_final.loc[:, 'timestamp'] = pd.to_datetime(df_hourly_final['timestamp'])

        avg_ppa_profile = str(int(round(np.mean(self.ppa_profile))))
        
        self._save_to_csv(df_hourly_final, f"consolidated_results_hourly_{avg_ppa_profile}MW.csv")

        # Create and save monthly summary
        numeric_cols = [col for col in df_hourly_final.columns if col != 'timestamp']
        df_monthly_summary = df_hourly_final.set_index('timestamp')[numeric_cols].resample('ME').mean().round(4)

        # Save summary data (non-timeseries results)
        summary_data = {k: v for k, v in results_data.items() if not isinstance(v, list)}
        if summary_data:
            df_summary = pd.DataFrame([summary_data])
            if save_as_csv:
                self._save_to_csv(df_summary, f"results_summary_{avg_ppa_profile}MW.csv")
            if save_as_excel:
                self._save_to_excel(df_summary, avg_ppa_profile)
   
    def _save_to_excel(self, df_hourly_final, avg_ppa_profile):
        """
        Save individual results to Excel (for single baseload case).
        """
        excel_path = f"{OUTPUTS_PATH}/results_baseload_{avg_ppa_profile}MW.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            sheet_name = f"Baseload {avg_ppa_profile}MW"
            df_hourly_final.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"Excel file saved: {excel_path}")

    def _save_to_csv(self, data, filename: str):
        """
        Helper to save data to CSV, handling lists, dicts, and pandas objects.
        """
        path = f"{OUTPUTS_PATH}/{filename}"
        if isinstance(data, pd.DataFrame):
            # For monthly summary, the index (month) is meaningful
            data.to_csv(path, index=not ('hourly' in filename), float_format='%.4f')
        elif isinstance(data, dict):
            # Convert Period keys to string for CSV if they are present
            df = pd.DataFrame(list(data.items()), columns=['key', 'value'])
            if not df.empty and hasattr(df['key'].iloc[0], 'to_timestamp'):
                 df['key'] = df['key'].astype(str)
            df.to_csv(path, index=False)
        else:
            # Fallback for other data types, though not expected with new save_results
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)

    @staticmethod
    def consolidate_all_to_excel(excel_filename: str = "consolidated_baseload_results.xlsx"):
        """
        Consolidate all existing hourly CSV files in outputs folder into a single Excel file.
        Each CSV becomes a sheet named 'Baseload XMW' where X is the average PPA value.
        """
        # Get all consolidated_results_hourly CSV files
        csv_pattern = f"{OUTPUTS_PATH}/consolidated_results_hourly_*MW.csv"
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print("No consolidated hourly CSV files found in outputs folder.")
            return
        
        excel_path = f"{OUTPUTS_PATH}/{excel_filename}"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for csv_file in sorted(csv_files):
                # Extract baseload value from filename
                filename = os.path.basename(csv_file)
                # Extract MW value from filename like "consolidated_results_hourly_15MW.csv"
                baseload_mw = filename.split('_')[-1].replace('MW.csv', '')
                
                # Read CSV
                df = pd.read_csv(csv_file)
                
                # Create sheet name
                sheet_name = f"Baseload {baseload_mw}MW"
                
                # Write to Excel sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Added sheet: {sheet_name}")
        
        print(f"Consolidated Excel file saved: {excel_path}")

def main_baseload(baseload_mw_list: list[float], consolidate_excel: bool = True):
    #variables
    capacity = 40
    max_soc = 0.8 #set to 1 for full capacity
    min_soc = 0.2 #set to 0 for no minimum capacity
    max_capacity = capacity * max_soc
    min_capacity = capacity * min_soc
    capacity_poi = 43.2
    max_charge_power = 9
    max_discharge_power = 9
    charge_efficiency = 0.85
    discharge_efficiency = 0.85
    initial_energy = 0
    
    # Load data variables
    generate_ppa_profile = True
   
    for baseload_mw in baseload_mw_list:
            # Create minimizer
            minimizer = PPADeficitMinimizer(max_capacity=max_capacity, min_capacity=min_capacity, max_charge_power=max_charge_power, max_discharge_power=max_discharge_power, charge_efficiency=charge_efficiency, discharge_efficiency=discharge_efficiency, initial_energy=initial_energy, capacity_poi=capacity_poi)
            minimizer.load_data(generate_ppa_profile, baseload_mw)

            results = minimizer.optimize_with_surplus_deficit(minimizer.wind_profile, minimizer.solar_profile, minimizer.ppa_profile)

            deficits = minimizer.compute_deficits(results['results'], minimizer.ppa_profile, minimizer.timestamps)

            # Save results
            minimizer.save_results(results=results, deficits=deficits, save_as_csv=True, save_as_excel=False)

    # After all baseloads are processed, consolidate to Excel
    if consolidate_excel and baseload_mw_list:
        PPADeficitMinimizer.consolidate_all_to_excel()

if __name__ == "__main__":
    main_baseload(baseload_mw_list=[15, 20, 25, 30, 35, 40], consolidate_excel=True)
    
