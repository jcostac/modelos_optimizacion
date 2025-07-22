import pulp
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import datetime

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import INPUTS_PATH, OUTPUTS_PATH

class PPADeficitMinimizer:
    def __init__(self, max_capacity, max_charge_power, max_discharge_power, charge_efficiency=0.9, discharge_efficiency=0.9, initial_energy=0, min_capacity=0):
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

    def optimize(self, wind: list[float], solar: list[float], ppa: list[float]) -> dict:
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
        state_of_charge = pulp.LpVariable.dicts("state_of_charge", range(T), lowBound=self.min_capacity, upBound=self.max_capacity)
        surplus = pulp.LpVariable.dicts("surplus", range(T), lowBound=0)
        deficit = pulp.LpVariable.dicts("deficit", range(T), lowBound=0)

        # Objective: Minimize total absolute deviation
        model += pulp.lpSum(surplus[t] + deficit[t] for t in range(T))

        # Constraints
        for t in range(T):
            supply = wind[t] + solar[t] + discharge[t] - charge[t]
            model += supply - ppa[t] == surplus[t] - deficit[t]

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
        return results

    def compute_deficits(self, results: dict, ppa: list[float], timestamps: list[datetime.datetime] = None) -> dict:
        """
        Compute hourly and monthly % deficits.
        
        Args:
            results: Output from optimize()
            ppa: PPA targets
            timestamps: Optional list of datetimes for monthly grouping
            
        Returns:
            dict: Hourly deficits and monthly deficits
        """
        T = len(ppa)
        deficit_values = results['deficit']

        # Hourly deficits
        hourly = [deficit_values[t] / ppa[t] if ppa[t] > 0 else 0 for t in range(T)]

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
        monthly['deficit_pct'] = monthly.apply(lambda row: row['deficit'] / row['ppa'] if row['ppa'] > 0 else 0, axis=1)
        
        return {'hourly': hourly, 'monthly': monthly['deficit_pct'].to_dict()}

    def load_data(self):
        """
        Load data from the inputs folder.
        """
        PATHS = {
                'wind_profile': f'{INPUTS_PATH}/wind_profile.csv',
                'solar_profile': f'{INPUTS_PATH}/solar_profile.csv',
                'ppa_profile': f'{INPUTS_PATH}/ppa_profile.csv',
                'timestamp_profile': f'{INPUTS_PATH}/timestamp_profile.csv'
            }
        
        self.wind_profile = pd.read_csv(PATHS['wind_profile'])
        self.solar_profile = pd.read_csv(PATHS['solar_profile'])
        self.ppa_profile = pd.read_csv(PATHS['ppa_profile'])
        self.timestamp_profile = pd.read_csv(PATHS['timestamp_profile'])


def main():
    #variables
    max_capacity = 20
    min_capacity = 0
    max_charge_power = 5
    max_discharge_power = 5
    charge_efficiency = 0.9
    discharge_efficiency = 0.9
    initial_energy = 0


    # Load data
    minimizer = PPADeficitMinimizer()
    minimizer.load_data()

    # Create minimizer
    minimizer = PPADeficitMinimizer(max_capacity, min_capacity, max_charge_power, max_discharge_power, charge_efficiency, discharge_efficiency, initial_energy)

    # Optimize
    results = minimizer.optimize(minimizer.wind_profile, minimizer.solar_profile, minimizer.ppa_profile)
    deficits = minimizer.compute_deficits(results, minimizer.ppa_profile, minimizer.timestamp_profile)



# Example usage (comment out or adapt as needed):
# minimizer = PPADeficitMinimizer(max_capacity=20, min_capacity=0, max_charge_power=5, max_discharge_power=5, charge_efficiency=0.9, discharge_efficiency=0.9)
# results = minimizer.optimize(wind_list, solar_list, ppa_list)
# deficits = minimizer.compute_deficits(results, ppa_list, timestamp_list)