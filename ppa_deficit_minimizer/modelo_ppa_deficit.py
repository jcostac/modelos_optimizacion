import pulp
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import datetime
import pretty_errors

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import INPUTS_PATH, OUTPUTS_PATH

class PPADeficitMinimizer:
    def __init__(self, max_capacity, min_capacity, max_charge_power, max_discharge_power, charge_efficiency, discharge_efficiency, initial_energy):
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

        if self.discharge_efficiency != 1:
            self.max_discharge_power = self.max_discharge_power * self.discharge_efficiency
        if self.charge_efficiency != 1:
            self.max_charge_power = self.max_charge_power * self.charge_efficiency

        # Decision variables
        charge = pulp.LpVariable.dicts("charge", range(T), lowBound=0, upBound=self.max_charge_power)
        discharge = pulp.LpVariable.dicts("discharge", range(T), lowBound=0, upBound=self.max_discharge_power)
        state_of_charge = pulp.LpVariable.dicts("state_of_charge", range(T), lowBound=self.min_capacity, upBound=self.max_capacity)
        surplus = pulp.LpVariable.dicts("surplus", range(T), lowBound=0)
        deficit = pulp.LpVariable.dicts("deficit", range(T), lowBound=0)

        # Objective: Minimize total absolute deviation
        model += pulp.lpSum(surplus[t] + deficit[t] for t in range(T))

        # Constraints
        for t in range(T):
            supply = wind[t] + solar[t] + discharge[t] - charge[t]
            model += supply - ppa[t] == surplus[t] - deficit[t]

        if self.min_capacity > 0:
            for t in range(1, T):
                model += state_of_charge[t] >= self.min_capacity

        # Energy balance at the beginning of the optimization
        model += state_of_charge[0] == self.initial_energy + self.charge_efficiency * charge[0] - (1 / self.discharge_efficiency) * discharge[0]
        for t in range(1, T): # Energy balance for all hours except the first one
            model += state_of_charge[t] == state_of_charge[t-1] + self.charge_efficiency * charge[t] - (1 / self.discharge_efficiency) * discharge[t]

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

    def optimize_without_surplus_deficit(self, wind: list[float], solar: list[float], ppa: list[float]) -> dict:
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

        model = pulp.LpProblem("PPA_Surplus_Maximizer", pulp.LpMaximize)

        if self.discharge_efficiency != 1:
            self.max_discharge_power = self.max_discharge_power * self.discharge_efficiency
        if self.charge_efficiency != 1:
            self.max_charge_power = self.max_charge_power * self.charge_efficiency

        # Decision variables
        charge = pulp.LpVariable.dicts("charge", range(T), lowBound=0, upBound=self.max_charge_power)
        discharge = pulp.LpVariable.dicts("discharge", range(T), lowBound=0, upBound=self.max_discharge_power)
        state_of_charge = pulp.LpVariable.dicts("state_of_charge", range(T), lowBound=self.min_capacity, upBound=self.max_capacity)

        # Compute deficit (PPA target minus renewables)
        surplus = [ppa[t] - wind[t] - solar[t] for t in range(T)]

        # Objective: Mainimize "surplus" to encourage matching PPA
        model += pulp.lpSum((discharge[t] - charge[t]) * surplus[t] for t in range(T))

    
        # Energy balance at the beginning of the optimization
        model += state_of_charge[0] == self.initial_energy + self.charge_efficiency * charge[0] - (1 / self.discharge_efficiency) * discharge[0]
        for t in range(1, T): # Energy balance for all hours except the first one
            model += state_of_charge[t] == state_of_charge[t-1] + self.charge_efficiency * charge[t] - (1 / self.discharge_efficiency) * discharge[t]

        # Add constraint for state of charge >= min capacity for all t > 0
        if self.min_capacity > 0:
            for t in range(1, T):
                model += state_of_charge[t] >= self.min_capacity

        # Solve model
        model.solve()

        # Extract results
        results = {
            'charge': [charge[t].value() for t in range(T)],
            'discharge': [discharge[t].value() for t in range(T)],
            'state_of_charge': [state_of_charge[t].value() for t in range(T)],
            'status': pulp.LpStatus[model.status],
            'total_deviation': 0 #placeholder for now
        }

        # Compute surplus and deficit outside the model based on optimized supply
        total_generation = [wind[t] + solar[t] + results['discharge'][t] - results['charge'][t] for t in range(T)]
        results['surplus'] = [max(0, total_generation[t] - ppa[t]) for t in range(T)]
        results['deficit'] = [max(0, ppa[t] - total_generation[t]) for t in range(T)]
        results['total_deviation'] = sum(results['surplus']) + sum(results['deficit'])

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

    def load_data(self):
        """
        Load data from the inputs folder.
        Assumes CSVs have a datetime column as the first column and a value column as the second.
        """
        PATHS = {
                'wind_profile': f'{INPUTS_PATH}/wind_profile.csv',
                'solar_profile': f'{INPUTS_PATH}/solar_profile.csv',
                'ppa_profile': f'{INPUTS_PATH}/ppa_profile.csv',
            }
        
        wind_df = pd.read_csv(PATHS['wind_profile'])
        solar_df = pd.read_csv(PATHS['solar_profile'])
        ppa_df = pd.read_csv(PATHS['ppa_profile'])

        self.wind_profile = wind_df.iloc[:, 1].tolist()
        self.solar_profile = solar_df.iloc[:, 1].tolist()
        self.ppa_profile = ppa_df.iloc[:, 1].tolist()
        
        # Use timestamps from ppa_profile, assuming they are consistent across files
        self.timestamps = pd.to_datetime(ppa_df.iloc[:, 0]).tolist()

    def save_results(self, results: dict, deficits: dict):
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

        # Select and order columns for the final hourly CSV
        final_hourly_cols = [
            'timestamp', 'wind', 'solar_pv', 'ppa_profile', 'soc', 
            'charge', 'discharge', 'deficit', '%_deficit'
        ]
        # Filter to only include columns that exist
        final_hourly_cols = [col for col in final_hourly_cols if col in df_hourly.columns]
        df_hourly_final = df_hourly[final_hourly_cols].copy()  # <-- Make a copy to avoid SettingWithCopyWarning

        # Round numeric columns to 4 decimal places
        for col in df_hourly_final.columns:
            if pd.api.types.is_numeric_dtype(df_hourly_final[col]):
                df_hourly_final.loc[:, col] = df_hourly_final[col].round(4)
        
        df_hourly_final.loc[:, 'timestamp'] = pd.to_datetime(df_hourly_final['timestamp'])
        
        self._save_to_csv(df_hourly_final, "consolidated_results_hourly.csv")

        # Create and save monthly summary
        numeric_cols = [col for col in df_hourly_final.columns if col != 'timestamp']
        df_monthly_summary = df_hourly_final.set_index('timestamp')[numeric_cols].resample('ME').mean().round(4)  # <-- Use 'ME' instead of 'M'
        
        self._save_to_csv(df_monthly_summary, "consolidated_results_monthly.csv")

        # Save summary data (non-timeseries results)
        summary_data = {k: v for k, v in results_data.items() if not isinstance(v, list)}
        if summary_data:
            df_summary = pd.DataFrame([summary_data])
            self._save_to_csv(df_summary, "results_summary.csv")


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

def main():
    #variables
    max_capacity = 40 * 0.8
    min_capacity = 40 * 0.2
    max_charge_power = 9
    max_discharge_power = 9
    charge_efficiency = 0.85
    discharge_efficiency = 0.85
    initial_energy = min_capacity

    with_surplus_deficit = False


    # Create minimizer
    minimizer = PPADeficitMinimizer(max_capacity=max_capacity, min_capacity=min_capacity, max_charge_power=max_charge_power, max_discharge_power=max_discharge_power, charge_efficiency=charge_efficiency, discharge_efficiency=discharge_efficiency, initial_energy=initial_energy)
    minimizer.load_data()

    if with_surplus_deficit:
        # Optimize
        results = minimizer.optimize_with_surplus_deficit(minimizer.wind_profile, minimizer.solar_profile, minimizer.ppa_profile)
    else:
        # Optimize
        results = minimizer.optimize_without_surplus_deficit(minimizer.wind_profile, minimizer.solar_profile, minimizer.ppa_profile)

    deficits = minimizer.compute_deficits(results['results'], minimizer.ppa_profile, minimizer.timestamps)

    # Save results
    minimizer.save_results(results=results, deficits=deficits)

if __name__ == "__main__":
    main()

# Example usage (comment out or adapt as needed):
# minimizer = PPADeficitMinimizer(max_capacity=20, min_capacity=0, max_charge_power=5, max_discharge_power=5, charge_efficiency=0.9, discharge_efficiency=0.9)
# results = minimizer.optimize(wind_list, solar_list, ppa_list)
# deficits = minimizer.compute_deficits(results, ppa_list, timestamp_list)