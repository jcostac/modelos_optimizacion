import pandas as pd
import numpy as np
import glob
import os
import sys
from pathlib import Path
import datetime

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import OUTPUTS_PATH

class ResultsHandler:
    """
    Handles calculation of deficits, and saving all results to CSV and Excel files.
    """
    def __init__(self, data_handler):
        self.data = data_handler

    def compute_deficits(self, results: dict, ppa: list[float], timestamps: list[datetime] = None) -> dict:
        """
        Compute hourly and monthly % deficits (unmet PPA).
        
        Args:
            results: Output from optimize()
            ppa: PPA targets
            timestamps: Optional list of datetimes for monthly grouping
            
        Returns:
            dict: Hourly deficits and monthly deficits in percentage %
        """
        T = len(ppa)
        unmet_values = results['ppa_deficit']

        # Hourly deficits 
        hourly = [(unmet_values[t] / ppa[t]) * 100 if ppa[t] > 0 else 0 for t in range(T)]

        if timestamps is None:
            return {'hourly': hourly, 'monthly': 'Timestamps required for monthly computation'}
        
        timestamps_pd = pd.to_datetime(timestamps)
        
        # Monthly deficits
        df = pd.DataFrame({
            'datetime': timestamps_pd,
            'ppa_deficit': unmet_values,
            'ppa': ppa
        })
        df['month'] = df['datetime'].dt.to_period('M')
        monthly_grouped = df.groupby('month').agg({'ppa_deficit': 'sum', 'ppa': 'sum'})
        monthly_grouped['%_deficit'] = monthly_grouped.apply(
            lambda row: (row['ppa_deficit'] / row['ppa']) * 100 if row['ppa'] > 0 else 0, 
            axis=1
        )
        
        return {'deficits': {'hourly': hourly, 'monthly': monthly_grouped['%_deficit'].to_dict()}}

    def save_results(self, results: dict, deficits: dict, capacity_poi: float, scenario_name: str = None):
        """
        Save results and deficits to the outputs folder.
        - Consolidated hourly time-series to consolidated_results_hourly.csv.
        - Monthly summary (with summed % deficit) to consolidated_results_monthly.csv.
        - Summary data to results_summary.csv.
        """
        results_data = results.get('results', {})
        deficits_data = deficits.get('deficits', {})

        # Prepare hourly DataFrame
        df_hourly_data = {
            'datetime': self.data.timestamps,
            'wind': self.data.wind_profile,
            'solar_pv': self.data.solar_profile,
            'ppa_profile': self.data.ppa_profile
        }

        # Add results
        for key, value in results_data.items():
            if isinstance(value, list):
                df_hourly_data[key] = value
        
        if 'hourly' in deficits_data:
            df_hourly_data['%_deficit'] = deficits_data['hourly']

        df_hourly = pd.DataFrame(df_hourly_data)

        # Rename and add columns
        df_hourly = df_hourly.rename(columns={'state_of_charge': 'soc'})
        df_hourly['net_charge'] = df_hourly['charge'] - df_hourly['discharge']

        df_hourly['wind + pv'] = df_hourly['wind'] + df_hourly['solar_pv']
        df_hourly['vertido_before'] = df_hourly['wind'] + df_hourly['solar_pv'] - capacity_poi
        df_hourly['vertido_before'] = df_hourly['vertido_before'].clip(lower=0)
        df_hourly['total_generation'] = df_hourly['wind'] + df_hourly['solar_pv'] + df_hourly['net_charge']
        df_hourly = df_hourly.rename(columns={'curtailment': 'vertido_after'})

        # Select columns
        final_hourly_cols = [
            'datetime', 'wind', 'solar_pv', "wind + pv", 'ppa_profile', 'vertido_before',  'soc', 
            'net_charge', "total_generation", 'vertido_after', 'total_grid_injection', 'ppa_deficit', 'curtailment', '%_deficit', 
            'total_ppa_deficit'
        ]
        final_hourly_cols = [col for col in final_hourly_cols if col in df_hourly.columns]
        df_hourly_final = df_hourly[final_hourly_cols].copy()

        # Round numerics
        for col in df_hourly_final.columns:
            if pd.api.types.is_numeric_dtype(df_hourly_final[col]):
                df_hourly_final[col] = df_hourly_final[col].round(4)
        
        df_hourly_final['datetime'] = pd.to_datetime(df_hourly_final['datetime'])

        if scenario_name is None:
            peak_ppa_profile = str(int(round(np.max(self.data.ppa_profile))))
            filename_suffix = f"{peak_ppa_profile}MW"
        else:
            filename_suffix = scenario_name
        
        self._save_to_csv(df_hourly_final, f"results_hourly_{filename_suffix}.csv")

        # Create and save monthly summary (with sums for accurate % deficit)
        if 'monthly' in deficits_data:
            df_monthly = pd.DataFrame(list(deficits_data['monthly'].items()), columns=['month', '%_deficit'])
            df_monthly['month'] = df_monthly['month'].astype(str)
            self._save_to_csv(df_monthly, f"results_monthly_{filename_suffix}.csv")

        # Save summary data
        summary_data = {k: v for k, v in results_data.items() if not isinstance(v, list)}
        if summary_data:
            df_summary = pd.DataFrame([summary_data])
            self._save_to_csv(df_summary, f"results_summary_{filename_suffix}.csv")

    def _save_to_csv(self, data, filename: str):
        """Helper to save to CSV."""
        path = f"{OUTPUTS_PATH}/{filename}"
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False, float_format='%.4f')
        else:
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
            
    @staticmethod
    def consolidate_all_to_excel(excel_filename: str = "consolidated_baseload_results.xlsx"):
        """
        Consolidate all existing hourly CSV files in outputs folder into a single Excel file.
        Each CSV becomes a sheet named with type and profile identifier.
        """
        csv_pattern = f"{OUTPUTS_PATH}/results_hourly_*MW.csv"
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print("No consolidated hourly CSV files found in outputs folder.")
            return
        
        excel_path = f"{OUTPUTS_PATH}/{excel_filename}"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for csv_file in sorted(csv_files, key=lambda x: os.path.basename(x)):
                
                filename = os.path.basename(csv_file)
                scenario_name = filename.replace('results_hourly_', '').replace('.csv', '')
                
                # Parse scenario name to create sheet name
                if scenario_name.startswith('flat_'):
                    # Format: flat_5MW -> flat_5MW
                    parts = scenario_name.split('_')
                    peak_part = parts[-1]  # e.g., "5MW"
                    peak_value = float(peak_part.replace('MW', ''))
                    peak_rounded = int(round(peak_value))
                    sheet_name = f"flat_{peak_rounded}MW"
                    
                elif scenario_name.startswith('seasonal_'):
                    # New format with custom profile name
                    profile_name = scenario_name.replace('seasonal_', '')
                    sheet_name = f"seasonal_{profile_name}"
                        
                elif scenario_name.startswith('custom_'):
                    # Format: custom_ppa_profile_15.3MW -> custom_15MW
                    parts = scenario_name.split('_')
                    peak_part = parts[-1]  # e.g., "15.3MW"
                    peak_value = float(peak_part.replace('MW', ''))
                    peak_rounded = int(round(peak_value))
                    sheet_name = f"custom_{peak_rounded}MW"
            
                
                df = pd.read_csv(csv_file)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Added sheet: {sheet_name}")
        
        print(f"Consolidated Excel file saved: {excel_path}") 