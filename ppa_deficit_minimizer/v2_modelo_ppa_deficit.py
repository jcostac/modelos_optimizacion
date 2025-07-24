import pyomo.environ as pyo
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import datetime
import glob
import os

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import INPUTS_PATH, OUTPUTS_PATH

class PPADeficitMinimizer:
    def __init__(self, max_capacity, min_capacity, max_charge_power, max_discharge_power, charge_efficiency, discharge_efficiency, initial_energy, capacity_poi):
        """
        Initialize ppa_deficit minimizer for a hybrid system with wind and solar generation.
        
        Args:
            max_capacity (float): Maximum storage capacity [MWh]
            min_capacity (float): Minimum storage capacity [MWh]
            max_charge_power (float): Maximum charge rate [MW]
            max_discharge_power (float): Maximum discharge rate [MW]
            charge_efficiency (float): Charge efficiency
            discharge_efficiency (float): Discharge efficiency
            initial_energy (float): Initial state-of-charge [MWh]
            capacity_poi (float): Max grid injection limit [MW]
        """
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.max_charge_power = max_charge_power
        self.max_discharge_power = max_discharge_power
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.initial_energy = initial_energy
        self.capacity_poi = capacity_poi

    def optimize(self, wind: list[float], solar: list[float], ppa: list[float], verbose: bool = False) -> dict:
        """
        Optimize battery charge/discharge to minimize unmet PPA.
        
        Args:
            wind: List of wind production [MW] per hour
            solar: List of solar production [MW] per hour
            ppa: List of PPA targets [MW] per hour
            verbose: If True, print detailed model info and solving progress
            
        Returns:
            dict: Optimization results including charge, discharge, state_of_charge, unmet, and curtailment.
        """
        T = len(ppa)
        assert len(wind) == T and len(solar) == T, "Input series must have same length"

        if verbose:
            self.print_model_info(wind, solar, ppa)

        model = pyo.ConcreteModel()
        model.T = pyo.RangeSet(0, T - 1)

        # Parameters
        model.W = pyo.Param(model.T, initialize=dict(enumerate(wind)), doc='Wind production [MW]')
        model.S = pyo.Param(model.T, initialize=dict(enumerate(solar)), doc='Solar production [MW]')
        model.PPA = pyo.Param(model.T, initialize=dict(enumerate(ppa)), doc='PPA profile [MW]')
        model.P_grid_max = pyo.Param(initialize=self.capacity_poi, doc='Max grid injection limit [MW]')
        model.P_chg_max = pyo.Param(initialize=self.max_charge_power, doc='Max charge power [MW]')
        model.P_dis_max = pyo.Param(initialize=self.max_discharge_power, doc='Max discharge power [MW]')
        model.eta_c = pyo.Param(initialize=self.charge_efficiency, doc='Charge efficiency')
        model.eta_d = pyo.Param(initialize=self.discharge_efficiency, doc='Discharge efficiency')
        model.SoC_min = pyo.Param(initialize=self.min_capacity, doc='Min state-of-charge [MWh]')
        model.SoC_max = pyo.Param(initialize=self.max_capacity, doc='Max state-of-charge [MWh]')
        model.SoC_0 = pyo.Param(initialize=self.initial_energy, doc='Initial state-of-charge [MWh]')
        model.Delta_t = pyo.Param(initialize=1, doc='Time step [h]') #used to convert charge and discharge rates in MW to energy rates in MWh

        # Decision variables
        model.c = pyo.Var(model.T, bounds=(0, self.max_charge_power), doc='Charge power [MW]')
        model.d = pyo.Var(model.T, bounds=(0, self.max_discharge_power), doc='Discharge power [MW]')
        model.SoC = pyo.Var(model.T, within=pyo.NonNegativeReals, doc='State-of-charge [MWh]')
        model.uPPA = pyo.Var(model.T, bounds=(0, None), doc='Unmet PPA [MW]')
        model.curt = pyo.Var(model.T, bounds=(0, None), doc='Curtailment [MW]')

        # Constraints
        def soc_dynamics_rule(m, t):
            """
            State-of-charge dynamics. Links SoC from one hour to the next.
            """
            if t == 0:
                # The SoC at the end of the first hour is the initial SoC + net charging in that hour.
                return m.SoC[t] == m.SoC_0 + m.eta_c * m.c[t] * m.Delta_t - (m.d[t] / m.eta_d) * m.Delta_t
            # For all other hours, it's based on the previous hour's SoC.
            return m.SoC[t] == m.SoC[t-1] + m.eta_c * m.c[t] * m.Delta_t - (m.d[t] / m.eta_d) * m.Delta_t
        model.soc_dynamics = pyo.Constraint(model.T, rule=soc_dynamics_rule)

        def soc_limits_rule(m, t):
            """
            Enforces min/max SoC bounds, skipping the first hour to avoid infeasibility.  Inequality = min SOC < SoC < max SOC
            """
            if t == 0:
                return pyo.Constraint.Skip
            return pyo.inequality(m.SoC_min, m.SoC[t], m.SoC_max)
        model.soc_limits = pyo.Constraint(model.T, rule=soc_limits_rule)

        def power_balance_rule(m, t):
            """
            Power balance: generation + discharge >= PPA target + charging - unmet
            """
            return m.W[t] + m.S[t] - m.curt[t] + m.d[t] >= m.PPA[t] - m.uPPA[t] + m.c[t]
        model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

        def poi_constraint_rule(m, t):
            """
            Grid injection limit: total power sent to grid cannot exceed the POI capacity.
            """
            return m.W[t] + m.S[t] - m.curt[t] + m.d[t] - m.c[t] <= m.P_grid_max
        model.poi_constraint = pyo.Constraint(model.T, rule=poi_constraint_rule)
    
        # Phase 1: Minimize total unmet PPA
        model.obj = pyo.Objective(expr=sum(model.uPPA[t] for t in model.T), sense=pyo.minimize)

        # Select solver
        solver = self.select_solver(verbose=verbose)

        if verbose:
            print("\n🎯 Phase 1: Minimizing total unmet PPA")
            self.print_model_setup(model, T, solver)
        else:
            print("Solving Phase 1...")
            
        results1 = solver.solve(model, tee=verbose)
        
        if verbose:
            self.print_optimization_completion(results1, model)

        # Check if Phase 1 solve was successful
        if (results1.solver.status != pyo.SolverStatus.ok or
            results1.solver.termination_condition != pyo.TerminationCondition.optimal):
            raise RuntimeError(f"Phase 1 solve failed: {results1.solver.status}")

        # Store optimal unmet PPA from Phase 1
        U_star = pyo.value(model.obj)

        # Phase 2: Maximize total grid injection while fixing unmet PPA
        model.obj.deactivate()
        #turn the unmet PPA into a constraint
        model.unmet_fix = pyo.Constraint(expr=sum(model.uPPA[t] for t in model.T) == U_star)
        #maximize grid injection
        model.obj2 = pyo.Objective(
            expr=sum(model.W[t] + model.S[t] - model.curt[t] + model.d[t] - model.c[t] for t in model.T),
            sense=pyo.maximize
        )

        if verbose:
            print(f"\n⚡ Phase 2: Maximizing grid injection (fixed unmet PPA = {U_star:.2f} MW)")
        else:
            print("Solving Phase 2...")

        results2 = solver.solve(model, tee=verbose)

        if verbose:
            print(f"\n✅ Phase 2 completed!")
            print(f"  • Status:           {results2.solver.status}")
            print(f"  • Termination:      {results2.solver.termination_condition}")
            if hasattr(results2.solver, 'time'):
                print(f"  • Solve time:       {results2.solver.time:.2f} seconds")
            print(f"  • Objective value:  {pyo.value(model.obj2):.2f} MWh (total grid injection)")

        # Check if Phase 2 solve was successful
        if (results2.solver.status != pyo.SolverStatus.ok or
            results2.solver.termination_condition != pyo.TerminationCondition.optimal):
            raise RuntimeError(f"Phase 2 solve failed: {results2.solver.status}")

        # Extract results from Phase 2 solution
        res = {
            'charge': [pyo.value(model.c[t]) for t in model.T],
            'discharge': [pyo.value(model.d[t]) for t in model.T],
            'state_of_charge': [pyo.value(model.SoC[t]) for t in model.T],
            'ppa_deficit': [pyo.value(model.uPPA[t]) for t in model.T],
            'curtailment': [pyo.value(model.curt[t]) for t in model.T],
            'status': results2.solver.status,
            'total_ppa_deficit': U_star,
            'total_grid_injection': pyo.value(model.obj2)
        }
        
        if verbose:
            self.print_results_summary(res)
            
        return {'results': res}
    
    def select_solver(self, verbose=False):
            """
            Try different solvers in order of preference and return the first available one.
            """
            solver_options = ['highs', 'glpk', 'cbc']
            for solver_name in solver_options:
                try:
                    solver = pyo.SolverFactory(solver_name)
                    if solver.available():
                        if verbose:
                            print(f"Using solver: {solver_name}")
                        return solver
                except Exception:
                    continue
            raise RuntimeError("No suitable open-source LP solver found. Please install 'highs', 'glpk', or 'cbc'.")
            
    def compute_deficits(self, results: dict, ppa: list[float], timestamps: list[datetime.datetime] = None) -> dict:
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
        else: 
            timestamps = pd.to_datetime(timestamps)
        
        # Monthly deficits
        df = pd.DataFrame({
            'datetime': timestamps,
            'ppa_deficit': unmet_values,
            'ppa': ppa
        })
        df['month'] = df['datetime'].dt.to_period('M')
        monthly = df.groupby('month').agg({'ppa_deficit': 'sum', 'ppa': 'sum'})
        monthly['%_deficit'] = monthly.apply(lambda row: (row['ppa_deficit'] / row['ppa']) * 100 if row['ppa'] > 0 else 0, axis=1)
        
        return {'deficits': {'hourly': hourly, 'monthly': monthly['%_deficit'].to_dict()}}

    def load_data(self, generate_ppa_profile: bool = False, baseload_mw: float = None, start_date: str = None, end_date: str = None):
        """
        Load data from the inputs folder. Can handle missing wind or solar profiles.
        Assumes CSVs have a datetime column as the first column and a value column as the second.
        If start_date and end_date are provided, the data will be filtered to only include data between those dates.

        Args:
            generate_ppa_profile (bool): Whether to generate a baseload profile or use a specific PPA profile
            baseload_mw (float): Constant baseload value in MW for generated profile.
            start_date (str): Start date for filtering data (e.g., 'YYYY-MM-DD').
            end_date (str): End date for filtering data (e.g., 'YYYY-MM-DD').
        """
        PATHS = {
            'wind_profile': f'{INPUTS_PATH}/wind_profile.csv',
            'solar_profile': f'{INPUTS_PATH}/solar_profile.csv',
            'ppa_profile': f'{INPUTS_PATH}/ppa_profile.csv',
        }

        def _load_profile(path):
            if os.path.exists(path):
                df = pd.read_csv(path)
                df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
                df['datetime'] = pd.to_datetime(df['datetime'])
                return df
            print(f"Info: '{os.path.basename(path)}' not found. Assuming zero generation for this source.")
            return None

        wind_df = _load_profile(PATHS['wind_profile'])
        solar_df = _load_profile(PATHS['solar_profile'])

        if wind_df is None and solar_df is None:
            raise FileNotFoundError("No generation data found. At least 'wind_profile.csv' or 'solar_profile.csv' must exist in the inputs folder.")

        # Filter by date
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            if wind_df is not None:
                wind_df = wind_df[(wind_df['datetime'] >= start) & (wind_df['datetime'] <= end)].reset_index(drop=True)
            if solar_df is not None:
                solar_df = solar_df[(solar_df['datetime'] >= start) & (solar_df['datetime'] <= end)].reset_index(drop=True)

        if wind_df is not None and solar_df is not None and len(wind_df) != len(solar_df):
            raise ValueError("Wind and solar profiles have different lengths after date filtering. Please check input files.")

        ref_df = wind_df if wind_df is not None else solar_df
        if ref_df is None or ref_df.empty:
            raise ValueError("No data available for the specified date range or in the provided files.")

        self.timestamps = ref_df['datetime'].tolist()
        profile_length = len(self.timestamps)

        self.wind_profile = wind_df.iloc[:, 1].tolist() if wind_df is not None else [0.0] * profile_length
        self.solar_profile = solar_df.iloc[:, 1].tolist() if solar_df is not None else [0.0] * profile_length

        if generate_ppa_profile:
            self.ppa_profile = self.create_baseload_ppa(baseload_mw, profile_length)
        else:
            if not os.path.exists(PATHS['ppa_profile']):
                raise FileNotFoundError(f"PPA profile not found at {PATHS['ppa_profile']}. To generate a baseload profile, set generate_ppa_profile=True.")
            
            ppa_df = _load_profile(PATHS['ppa_profile'])
            
            # Align PPA with generation data timestamps
            ppa_df = ppa_df.set_index('datetime').reindex(ref_df['datetime']).reset_index()
            # Check for NaNs after reindexing if PPA file doesn't cover the full range
            if ppa_df.iloc[:, 1].isnull().any():
                print("Warning: PPA profile has missing values for some timestamps. Filling with 0.")
                ppa_df.iloc[:, 1].fillna(0, inplace=True)
            self.ppa_profile = ppa_df.iloc[:, 1].tolist()

        # Final length check
        length_wind = len(self.wind_profile)
        length_solar = len(self.solar_profile)
        length_ppa = len(self.ppa_profile)
        if length_wind != length_solar or length_wind != length_ppa:
            print(f"Length of wind profile: {length_wind}")
            print(f"Length of solar profile: {length_solar}")
            print(f"Length of PPA profile: {length_ppa}")
            raise ValueError("Wind, solar, and PPA profiles must have the same length after processing.")

        self.timestamps = ref_df['datetime'].tolist()

    def create_baseload_ppa(self, baseload_mw, profile_length):
        """
        Create a baseload PPA profile with constant value.
        
        Args:
            baseload_mw (float): Constant baseload value in MW
            profile_length (int): The length of the profile to generate.
            
        Returns:
            list: PPA profile with constant baseload value
        """
        if not hasattr(self, 'wind_profile') or not hasattr(self, 'solar_profile'):
            raise ValueError("Wind and solar profiles must be loaded first. Call load_data() before creating baseload PPA.")

        if baseload_mw is None:
            raise ValueError("Baseload value must be provided when generating a PPA profile.")
        
        return [baseload_mw] * profile_length

    def save_results(self, results: dict, deficits: dict):
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
            'datetime': self.timestamps,
            'wind': self.wind_profile,
            'solar_pv': self.solar_profile,
            'ppa_profile': self.ppa_profile
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

        # Select columns
        final_hourly_cols = [
            'datetime', 'wind', 'solar_pv', 'ppa_profile', 'soc',
            'net_charge', 'ppa_deficit', 'curtailment', '%_deficit', 'total_ppa_deficit', 'total_grid_injection'
        ]
        final_hourly_cols = [col for col in final_hourly_cols if col in df_hourly.columns]
        df_hourly_final = df_hourly[final_hourly_cols].copy()

        # Round numerics
        for col in df_hourly_final.columns:
            if pd.api.types.is_numeric_dtype(df_hourly_final[col]):
                df_hourly_final[col] = df_hourly_final[col].round(4)
        
        df_hourly_final['datetime'] = pd.to_datetime(df_hourly_final['datetime'])

        avg_ppa_profile = str(int(round(np.mean(self.ppa_profile))))
        
        self._save_to_csv(df_hourly_final, f"consolidated_results_hourly_{avg_ppa_profile}MW.csv")

        # Create and save monthly summary (with sums for accurate % deficit)
        if 'monthly' in deficits_data:
            df_monthly = pd.DataFrame(list(deficits_data['monthly'].items()), columns=['month', '%_deficit'])
            df_monthly['month'] = df_monthly['month'].astype(str)
            self._save_to_csv(df_monthly, f"consolidated_results_monthly_{avg_ppa_profile}MW.csv")

        # Save summary data
        summary_data = {k: v for k, v in results_data.items() if not isinstance(v, list)}
        if summary_data:
            df_summary = pd.DataFrame([summary_data])
            self._save_to_csv(df_summary, f"results_summary_{avg_ppa_profile}MW.csv")

    def _save_to_csv(self, data, filename: str):
        """
        Helper to save to CSV.
        """
        path = f"{OUTPUTS_PATH}/{filename}"
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False, float_format='%.4f')
        else:
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)

    def print_model_info(self, wind: list[float], solar: list[float], ppa: list[float]):
        """
        Print detailed model parameters and input data statistics.
        
        Args:
            wind: List of wind production [MW] per hour
            solar: List of solar production [MW] per hour  
            ppa: List of PPA targets [MW] per hour
        """
        print("\n" + "="*60)
        print("PPA DEFICIT MINIMIZER - MODEL CONFIGURATION")
        print("="*60)
        
        # Battery Parameters
        print("\n📦 BATTERY SYSTEM PARAMETERS:")
        print(f"  • Max Capacity:           {self.max_capacity:.1f} MWh")
        print(f"  • Min Capacity:           {self.min_capacity:.1f} MWh")
        print(f"  • Usable Capacity:        {self.max_capacity - self.min_capacity:.1f} MWh")
        print(f"  • Max Charge Power:       {self.max_charge_power:.1f} MW")
        print(f"  • Max Discharge Power:    {self.max_discharge_power:.1f} MW")
        print(f"  • Charge Efficiency:      {self.charge_efficiency*100:.1f}%")
        print(f"  • Discharge Efficiency:   {self.discharge_efficiency*100:.1f}%")
        print(f"  • Initial Energy:         {self.initial_energy:.1f} MWh")
        
        # Grid Parameters
        print(f"\n⚡ GRID CONNECTION:")
        print(f"  • POI Capacity Limit:     {self.capacity_poi:.1f} MW")
        
        # Input Data Statistics
        T = len(ppa)
        print(f"\n📊 INPUT DATA STATISTICS (T = {T:,} hours):")
        
        print(f"\n  🌪️  Wind Generation:")
        print(f"     • Average:             {np.mean(wind):.2f} MW")
        print(f"     • Max:                 {np.max(wind):.2f} MW")
        print(f"     • Min:                 {np.min(wind):.2f} MW")
        print(f"     • Total Energy:        {np.sum(wind):.0f} MWh")
        
        print(f"\n  ☀️  Solar Generation:")
        print(f"     • Average:             {np.mean(solar):.2f} MW")
        print(f"     • Max:                 {np.max(solar):.2f} MW")
        print(f"     • Min:                 {np.min(solar):.2f} MW")
        print(f"     • Total Energy:        {np.sum(solar):.0f} MWh")
        
        total_gen = np.array(wind) + np.array(solar)
        print(f"\n  ⚡ Combined Generation:")
        print(f"     • Average:             {np.mean(total_gen):.2f} MW")
        print(f"     • Max:                 {np.max(total_gen):.2f} MW")
        print(f"     • Min:                 {np.min(total_gen):.2f} MW")
        print(f"     • Total Energy:        {np.sum(total_gen):.0f} MWh")
        
        print(f"\n  🎯 PPA Target:")
        print(f"     • Average:             {np.mean(ppa):.2f} MW")
        print(f"     • Max:                 {np.max(ppa):.2f} MW")
        print(f"     • Min:                 {np.min(ppa):.2f} MW")
        print(f"     • Total Target:        {np.sum(ppa):.0f} MWh")
        
        # Energy balance analysis
        total_deficit = np.sum(ppa) - np.sum(total_gen)
        print(f"\n  📈 ENERGY BALANCE ANALYSIS:")
        print(f"     • Total PPA Target:    {np.sum(ppa):.0f} MWh")
        print(f"     • Total Generation:    {np.sum(total_gen):.0f} MWh")
        if total_deficit > 0:
            print(f"     • Energy Deficit:      {total_deficit:.0f} MWh ⚠️")
        else:
            print(f"     • Energy Surplus:      {abs(total_deficit):.0f} MWh ✅")
        
        capacity_factor = (np.sum(total_gen) / (np.max(total_gen) * T)) * 100 if np.max(total_gen) > 0 else 0
        print(f"     • Capacity Factor:     {capacity_factor:.1f}%")
        
        print("\n" + "="*60 + "\n")

    def print_model_setup(self, model, T, solver):
        """Print optimization model setup information."""
        print("🔧 OPTIMIZATION MODEL SETUP:")
        print(f"  • Variables:        {len(list(model.component_objects(pyo.Var))):,}")
        print(f"  • Constraints:      {len(list(model.component_objects(pyo.Constraint))):,}")
        print(f"  • Time periods:     {T:,}")
        print(f"  • Solver:           {solver.name}")
        print("\n🚀 Starting optimization...")

    def print_optimization_completion(self, results, model):
        """Print optimization completion information."""
        print(f"\n✅ OPTIMIZATION COMPLETED!")
        print(f"  • Status:           {results.solver.status}")
        print(f"  • Termination:      {results.solver.termination_condition}")
        if hasattr(results.solver, 'time'):
            print(f"  • Solve time:       {results.solver.time:.2f} seconds")
        print(f"  • Objective value:  {pyo.value(model.obj):.2f} MW (total unmet PPA)")

    def print_results_summary(self, results):
        """Print detailed optimization results summary."""
        total_charge = sum(results['charge'])
        total_discharge = sum(results['discharge'])
        total_curtailment = sum(results['curtailment'])

        #min and max soc after hour 0 (initial energy)
        min_soc = min(results['state_of_charge'][1:]) if len(results['state_of_charge']) > 1 else results['state_of_charge'][0]
        max_soc = max(results['state_of_charge'][1:]) if len(results['state_of_charge']) > 1 else results['state_of_charge'][0]
        
        print(f"\n📊 OPTIMIZATION RESULTS SUMMARY:")
        print(f"  • Total Energy Charged:    {total_charge:.1f} MWh")
        print(f"  • Total Energy Discharged: {total_discharge:.1f} MWh")
        print(f"  • Total Curtailment:       {total_curtailment:.1f} MWh")
        print(f"  • Max SoC Reached:         {max_soc:.1f} MWh")
        print(f"  • Min SoC Reached:         {min_soc:.1f} MWh")
        print(f"  • Battery Utilization:     {((max_soc - min_soc) / (self.max_capacity - self.min_capacity) * 100):.1f}%")
        print("="*60 + "\n")

    @staticmethod
    def consolidate_all_to_excel(excel_filename: str = "consolidated_baseload_results.xlsx"):
        """
        Consolidate all existing hourly CSV files in outputs folder into a single Excel file.
        Each CSV becomes a sheet named 'Baseload XMW' where X is the average PPA value.
        """
        csv_pattern = f"{OUTPUTS_PATH}/consolidated_results_hourly_*MW.csv"
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print("No consolidated hourly CSV files found in outputs folder.")
            return
        
        excel_path = f"{OUTPUTS_PATH}/{excel_filename}"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for csv_file in sorted(csv_files):
                filename = os.path.basename(csv_file)
                baseload_mw = filename.split('_')[-1].replace('MW.csv', '')
                df = pd.read_csv(csv_file)
                sheet_name = f"Baseload {baseload_mw}MW"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"Added sheet: {sheet_name}")
        
        print(f"Consolidated Excel file saved: {excel_path}")

def main_baseload(baseload_mw_list: list[float], consolidate_excel: bool = True, start_date: str = None, end_date: str = None, verbose: bool = True):
    # Variables
    capacity = 40
    max_soc = 0.95  # Set to 1 for full capacity
    min_soc = 0.05  # Set to 0 for no minimum capacity
    max_capacity = capacity * max_soc
    min_capacity = capacity * min_soc
    capacity_poi = 43.2
    max_charge_power = 9
    max_discharge_power = 9
    charge_efficiency = np.sqrt(0.9)
    discharge_efficiency = np.sqrt(0.9)
    initial_energy = 0
    
    # Load data variables
    generate_ppa_profile = True
   
    for baseload_mw in baseload_mw_list:
        if verbose:
            print(f"\n🎯 Processing baseload scenario: {baseload_mw} MW")
            
        # Create minimizer
        minimizer = PPADeficitMinimizer(
            max_capacity=max_capacity,
            min_capacity=min_capacity,
            max_charge_power=max_charge_power,
            max_discharge_power=max_discharge_power,
            charge_efficiency=charge_efficiency,
            discharge_efficiency=discharge_efficiency,
            initial_energy=initial_energy,
            capacity_poi=capacity_poi
        )
        minimizer.load_data(generate_ppa_profile, baseload_mw, start_date, end_date)

        results = minimizer.optimize(minimizer.wind_profile, minimizer.solar_profile, minimizer.ppa_profile, verbose=verbose)

        deficits = minimizer.compute_deficits(results['results'], minimizer.ppa_profile, minimizer.timestamps)

        # Save results
        minimizer.save_results(results=results, deficits=deficits)
        
        if verbose:
            print(f"✅ Results saved for {baseload_mw} MW baseload scenario\n")

    # Consolidate to Excel
    if consolidate_excel and baseload_mw_list:
        PPADeficitMinimizer.consolidate_all_to_excel()

if __name__ == "__main__":
    main_baseload(baseload_mw_list=[15], consolidate_excel=False, start_date='2000-01-01', end_date='2000-12-31', verbose=True)