import pyomo.environ as pyo
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import datetime
# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

import config
from ppa_deficit_minimizer.data_handler import DataHandler
from ppa_deficit_minimizer.results_handler import ResultsHandler

class PPADeficitMinimizer:
    
    def __init__(self, battery_params, capacity_poi, battery_degradation_factor=None, battery_degradation_profile=None):
        """
        Initialize ppa_deficit minimizer for a hybrid system with wind and solar generation.
        
        Args:
            battery_params (dict): Dictionary with battery parameters.
            capacity_poi (float): Max grid injection limit [MW]
            battery_degradation_factor (float, optional): Annual degradation factor. Defaults to None.
            battery_degradation_profile (dict, optional): Degradation profile by year. Defaults to None.
        """
        self.max_capacity = battery_params['max_capacity']
        self.min_capacity = battery_params['min_capacity']
        self.max_charge_power = battery_params['max_charge_power']
        self.max_discharge_power = battery_params['max_discharge_power']
        self.charge_efficiency = battery_params['charge_efficiency']
        self.discharge_efficiency = battery_params['discharge_efficiency']
        self.initial_energy = battery_params['initial_energy']
        self.capacity_poi = capacity_poi 
        self.battery_degradation_factor = battery_degradation_factor
        self.battery_degradation_profile = battery_degradation_profile

    def optimize(self, wind: list[float], solar: list[float], ppa: list[float], timestamps: list[datetime.datetime] = None, verbose: bool = False) -> dict:
        """
        Optimize battery charge/discharge to minimize unmet PPA.
        
        Args:
            wind: List of wind production [MW] per hour
            solar: List of solar production [MW] per hour
            ppa: List of PPA targets [MW] per hour
            timestamps (list, optional): List of datetimes for degradation calculation. Defaults to None.
            verbose: If True, print detailed model info and solving progress
            
        Returns:
            dict: Optimization results including charge, discharge, state_of_charge, unmet, and curtailment.
        """
        T = len(ppa)
        assert len(wind) == T and len(solar) == T, "Input series must have same length"

        def _clean_data(data_list: list[float], name: str) -> list[float]:
            """Replaces NaNs with 0.0 and prints a warning."""
            nan_count = sum(1 for x in data_list if pd.isna(x))
            if nan_count > 0:
                print(f"Warning: Found and replaced {nan_count} NaN value(s) with 0 in '{name}' data.")
                return [0.0 if pd.isna(x) else x for x in data_list]
            return data_list

        wind = _clean_data(wind, 'wind')
        solar = _clean_data(solar, 'solar')
        ppa = _clean_data(ppa, 'ppa')

        if verbose:
            self.print_model_info(wind, solar, ppa)

        model = pyo.ConcreteModel()
        model.T = pyo.RangeSet(0, T - 1)

        #scalar parameters
        model.P_grid_max = pyo.Param(initialize=self.capacity_poi, within=pyo.NonNegativeReals, doc='Max grid injection limit [MW]')
        model.P_chg_max = pyo.Param(initialize=self.max_charge_power, within=pyo.NonNegativeReals, doc='Max charge power [MW]')
        model.P_dis_max = pyo.Param(initialize=self.max_discharge_power, within=pyo.NonNegativeReals, doc='Max discharge power [MW]')
        model.eta_c = pyo.Param(initialize=self.charge_efficiency, within=pyo.NonNegativeReals, doc='Charge efficiency')
        model.eta_d = pyo.Param(initialize=self.discharge_efficiency, within=pyo.NonNegativeReals, doc='Discharge efficiency')
        model.SoC_min = pyo.Param(initialize=self.min_capacity, within=pyo.NonNegativeReals, doc='Min state-of-charge [MWh]')
        model.SoC_0 = pyo.Param(initialize=self.initial_energy, within=pyo.NonNegativeReals, doc='Initial state-of-charge [MWh]')
        model.Delta_t = pyo.Param(initialize=1, within=pyo.NonNegativeReals, doc='Time step [h]') #used to convert charge and discharge rates in MW to energy rates in MWh (useful to convert ranularity of the data)
        
        # Validate that Delta_t matches the input data granularity
        if timestamps is not None:
            self._validate_granularity(timestamps, pyo.value(model.Delta_t), verbose=verbose)
        
        # Pre-calculate dict of degraded capacity values (year -> degraded capacity)
        soc_max_values = self._calculate_degraded_capacity(timestamps, T)

        #time-dependent parameters
        model.SoC_max = pyo.Param(model.T, initialize=soc_max_values, within=pyo.NonNegativeReals, doc='Max state-of-charge [MWh]')
        model.W = pyo.Param(model.T, initialize=dict(enumerate(wind)), within=pyo.NonNegativeReals, doc='Wind production [MW]')
        model.S = pyo.Param(model.T, initialize=dict(enumerate(solar)), within=pyo.NonNegativeReals, doc='Solar production [MW]')
        model.PPA = pyo.Param(model.T, initialize=dict(enumerate(ppa)), within=pyo.NonNegativeReals, doc='PPA profile [MW]')

        
        # Decision variables
        model.c = pyo.Var(model.T, bounds=(0, self.max_charge_power), within=pyo.NonNegativeReals, doc='Charge power [MW]')
        model.d = pyo.Var(model.T, bounds=(0, self.max_discharge_power), within=pyo.NonNegativeReals, doc='Discharge power [MW]')
        model.SoC = pyo.Var(model.T, within=pyo.NonNegativeReals, doc='State-of-charge [MWh]')  
        model.uPPA = pyo.Var(model.T, bounds=(0, None), within=pyo.NonNegativeReals, doc='Unmet PPA [MW]')
        model.curt = pyo.Var(model.T, bounds=(0, None), within=pyo.NonNegativeReals, doc='Curtailment [MW]')

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
            return pyo.inequality(m.SoC_min, m.SoC[t], m.SoC_max[t])
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
    
    def _calculate_degraded_capacity(self, timestamps: list[datetime.datetime], T: int) -> dict:
        """
        Calculates time-dependent maximum capacity after applying annual degradation.
        This is more efficient than updating a mutable Pyomo parameter in a loop.

        Args:
            timestamps: List of timestamps
            T: Length of the time series

        Returns:
            dict: Mapping of time step (year) to degraded capacity ie {0: 100, 1: 99, 2: 98, ...}
        """
        if self.battery_degradation_profile is not None:
            if timestamps is None:
                raise ValueError("Timestamps must be provided when using degradation profile.")

            timestamps_pd = pd.to_datetime(timestamps)
            if len(timestamps_pd) != T:
                raise ValueError("Timestamps must match the length of input data.")

            years = timestamps_pd.year
            start_year = years.min()

            # Get the last year in profile for defaulting beyond range
            if self.battery_degradation_profile:
                last_year = max(self.battery_degradation_profile.keys())
                default_remaining = self.battery_degradation_profile[last_year]
            else:
                default_remaining = 1.0

            soc_max_values = {}
            for t in range(T):
                relative_year = years[t] - start_year + 1
                remaining = self.battery_degradation_profile.get(relative_year, default_remaining)
                soc_max_values[t] = self.max_capacity * remaining
            return soc_max_values
        
        else:
            if self.battery_degradation_factor is None or self.battery_degradation_factor == 0.0:
                return {t: self.max_capacity for t in range(T)}

            if not timestamps:
                raise ValueError("Timestamps must be provided for battery degradation.")

            timestamps_pd = pd.to_datetime(timestamps)
            if len(timestamps_pd) != T:
                raise ValueError("Timestamps must match the length of input data.")

            years = timestamps_pd.year
            start_year = years.min()

            degradation_map = {
                year: (1 - self.battery_degradation_factor) ** (year - start_year)
                for year in years.unique()
            }

            degraded_capacity_dict = {
                t: self.max_capacity * degradation_map[years[t]]
                for t in range(T)
            }
            
            return degraded_capacity_dict

    def _validate_granularity(self, timestamps, delta_t_hours, verbose=False):
        """
        Validates that the Delta_t parameter matches the actual time granularity of the input data.
        
        Args:
            timestamps: List of timestamps from input data
            delta_t_hours: Delta_t value in hours
            
        Raises:
            ValueError: If Delta_t doesn't match the data granularity
        """
        if len(timestamps) < 2:
            print("Warning: Cannot validate granularity with less than 2 timestamps.")
            return
            
        timestamps_pd = pd.to_datetime(timestamps)
        
        # Calculate the actual time differences between consecutive timestamps
        time_diffs = timestamps_pd[1:] - timestamps_pd[:-1]
        
        # Convert to hours. .dt accessor is not needed for TimedeltaIndex.
        actual_delta_hours = time_diffs.total_seconds() / 3600

        # Find the most frequent time step (mode) to handle minor data gaps gracefully.
        delta_series = pd.Series(actual_delta_hours).round(6) # Round to avoid float precision issues
        
        if delta_series.empty:
            print("Warning: Not enough data to determine granularity.")
            return

        actual_granularity = delta_series.mode()[0]
        
        # Warn if multiple time steps are found, but proceed with the mode.
        unique_deltas = delta_series.unique()
        if len(unique_deltas) > 1 and verbose:
            print(f"Warning: Inconsistent time steps found. Proceeding with most frequent: {actual_granularity:.4f} hours.")
            print(f"  Detected steps (hours): {np.round(unique_deltas, 4)}")
        
        # Allow small floating point tolerance (1 second = ~0.0003 hours)
        tolerance = 0.001  # hours
        
        if abs(actual_granularity - delta_t_hours) > tolerance:
            raise ValueError(
                f"Granularity mismatch!\n"
                f"  • Input data granularity: {actual_granularity:.4f} hours ({actual_granularity*60:.1f} minutes)\n"
                f"  • Model Delta_t setting:  {delta_t_hours:.4f} hours ({delta_t_hours*60:.1f} minutes)\n"
                f"  • Please adjust Delta_t to match your input data granularity."
            )
        
        if verbose:
            print(f"✅ Granularity check passed: {actual_granularity:.4f} hours ({actual_granularity*60:.1f} minutes)")

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