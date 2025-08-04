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

        df_hourly['produccion'] = df_hourly['wind'] + df_hourly['solar_pv']
        #df_hourly['vertido_before'] = df_hourly['wind'] + df_hourly['solar_pv'] - capacity_poi
        #df_hourly['vertido_before'] = df_hourly['vertido_before'].clip(lower=0)
        df_hourly['wind+solar+bat'] = df_hourly['wind'] + df_hourly['solar_pv'] + df_hourly['net_charge']
        df_hourly["ppa_excess"] = df_hourly['wind+solar+bat'] - df_hourly['ppa_profile']
        df_hourly["ppa_excess"] = df_hourly["ppa_excess"].clip(lower=0)

        # Select columns
        final_hourly_cols = [
            'datetime', 'wind', 'solar_pv', "produccion", 'ppa_profile',  'soc', 
            'net_charge', 'vertido_after', 'total_grid_injection', 'ppa_deficit', 'ppa_excess', 'curtailment', '%_deficit'
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

        self._calculate_and_save_yearly_deficit_summary(df_hourly_final, filename_suffix)
        self._calculate_and_save_per_hour_summary(df_hourly_final, filename_suffix)

    def _calculate_and_save_yearly_deficit_summary(self, df_hourly: pd.DataFrame, filename_suffix: str):
        """
        Calculates and saves a yearly summary of deficit metrics.
        """
        df = df_hourly.copy()
        if 'datetime' not in df.columns:
            print("Yearly summary not created: 'datetime' column not found.")
            return

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = df['datetime'].dt.year

        yearly_summary = []
        
        for year in sorted(df['year'].unique()):
            df_year = df[df['year'] == year]
            
            hours_in_year = len(df_year)
            if hours_in_year == 0:
                continue
                
            n_hours_deficit = (df_year['ppa_deficit'] > 1e-4).sum()
            pct_hours_deficit = (n_hours_deficit / hours_in_year) * 100
            
            total_energy_deficit = df_year['ppa_deficit'].sum()
            
            if 'net_charge' in df_year.columns:
                total_production = (df_year['wind'] + df_year['solar_pv'] + df_year['net_charge']).sum()
            else:
                total_production = (df_year['wind'] + df_year['solar_pv']).sum()

            pct_energy_deficit = (total_energy_deficit / total_production) * 100 if total_production > 0 else 0
            
            yearly_summary.append({
                'Year': year,
                'Deficit Hours': n_hours_deficit,
                'Deficit Hours %': pct_hours_deficit,
                'Deficit Energy (MWh)': total_energy_deficit,
                'Deficit Energy %': pct_energy_deficit
            })

        if not yearly_summary:
            print("No data to generate yearly deficit summary.")
            return
            
        df_summary = pd.DataFrame(yearly_summary)
        
        # Rounding for presentation
        df_summary['Deficit Hours %'] = df_summary['Deficit Hours %'].round(1)
        df_summary['Deficit Energy (MWh)'] = df_summary['Deficit Energy (MWh)'].round(2)
        df_summary['Deficit Energy %'] = df_summary['Deficit Energy %'].round(1)

        self._save_to_csv(df_summary, f"results_yearly_deficit_summary_{filename_suffix}.csv")

    def _calculate_and_save_per_hour_summary(self, df_hourly: pd.DataFrame, filename_suffix: str):
        """
        Calculates and saves a summary of deficit and excess per hour of the day.
        """
        df = df_hourly.copy()
        if 'datetime' not in df.columns:
            print("Per-hour summary not created: 'datetime' column not found.")
            return

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour

        if 'net_charge' in df.columns:
            df['produccion'] = df['wind'] + df['solar_pv'] + df['net_charge']
        else:
            df['produccion'] = df['wind'] + df['solar_pv']

        # Group by hour of the day and calculate metrics
        hourly_summary = df.groupby('hour').apply(self._calculate_hourly_metrics).reset_index()

        if hourly_summary.empty:
            print("No data to generate per-hour deficit/excess summary.")
            return
        
        # Format for presentation
        hourly_summary = hourly_summary.round({
            'production': 2,
            ('deficit', '%'): 2,
            ('deficit', 'energy'): 2,
            ('deficit', '%_energy'): 1,
            ('excess', '%'): 2,
            ('excess', 'energy'): 2,
            ('excess', '%_energy'): 1,
        })
        
        self._save_to_csv(hourly_summary, f"results_per_hour_summary_{filename_suffix}.csv")

    @staticmethod
    def _calculate_hourly_metrics(group):
        """Helper to calculate metrics for each hourly group."""
        total_hours_of_type = len(group)
        if total_hours_of_type == 0:
            return None
        
        total_production = group['produccion'].sum()

        # Deficit
        deficit_hours = (group['ppa_deficit'] > 1e-4).sum()
        pct_deficit_hours = (deficit_hours / total_hours_of_type) * 100
        deficit_energy = group['ppa_deficit'].sum()
        pct_deficit_energy = (deficit_energy / total_production) * 100 if total_production > 0 else 0

        # Excess
        excess_hours = (group['ppa_excess'] > 1e-4).sum()
        pct_excess_hours = (excess_hours / total_hours_of_type) * 100
        excess_energy = group['ppa_excess'].sum()
        pct_excess_energy = (excess_energy / total_production) * 100 if total_production > 0 else 0

        data = {
            'production': total_production,
            'total_hours': total_hours_of_type,
            ('deficit', 'hours'): deficit_hours,
            ('deficit', '%'): pct_deficit_hours,
            ('deficit', 'energy'): deficit_energy,
            ('deficit', '%_energy'): pct_deficit_energy,
            ('excess', 'hours'): excess_hours,
            ('excess', '%'): pct_excess_hours,
            ('excess', 'energy'): excess_energy,
            ('excess', '%_energy'): pct_excess_energy,
        }
        return pd.Series(data)

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