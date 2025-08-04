import pyomo.environ as pyo
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import datetime
# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

import config
from data_handler import DataHandler
from results_handler import ResultsHandler
from models.ppa_deficit_minimizer_v3 import PPADeficitMinimizer

def main(baseload_mw_list: list[float] = None, seasonal_json_list: list[str] = None, preloaded_ppa_list: list[str] = None, peak_start: int = 8, peak_end: int = 20, consolidate_excel: bool = True, start_date: str = None, end_date: str = None, verbose: bool = True):
    """
    Main function to run PPA scenarios across multiple profile types.
    
    Args:
        baseload_mw_list (list[float]): List of baseload MW values to run scenarios for.
        seasonal_json_list (list[str]): List of seasonal PPA JSON paths to run scenarios for.
        preloaded_ppa_list (list[str]): List of preloaded PPA CSV files to run scenarios for.
        peak_start (int): Start hour for peak period (inclusive).
        peak_end (int): End hour for peak period (inclusive).
        consolidate_excel (bool, optional): Consolidate all results into one Excel file. Defaults to True.
        start_date (str, optional): Start date for data filtering ('YYYY-MM-DD'). Defaults to None.
        end_date (str, optional): End date for data filtering ('YYYY-MM-DD'). Defaults to None.
        verbose (bool, optional): Enable detailed print statements. Defaults to True.
    """
    # Load configuration from config.py
    battery_params = {
        'max_capacity': config.CAPACITY * config.MAX_SOC_PERCENT,
        'min_capacity': config.CAPACITY * config.MIN_SOC_PERCENT,
        'max_charge_power': config.MAX_CHARGE_POWER,
        'max_discharge_power': config.MAX_DISCHARGE_POWER,
        'charge_efficiency': config.CHARGE_EFFICIENCY,
        'discharge_efficiency': config.DISCHARGE_EFFICIENCY,
        'initial_energy': config.INITIAL_ENERGY_MWH,
    }

    # Build scenarios from all enabled profile types in order: flat → seasonal → preloaded
    scenarios = []
    
    # Add flat profile scenarios
    if config.GENERATE_FLAT_PPA_PROFILE:
        flat_scenarios = [('flat', mw) for mw in (baseload_mw_list or config.BASELOAD_MW_LIST)]
        scenarios.extend(flat_scenarios)
        if verbose:
            print(f"📋 Added {len(flat_scenarios)} flat baseload scenarios")
    
    # Add seasonal profile scenarios
    if config.GENERATE_SEASONAL_PPA_PROFILE:
        seasonal_scenarios = [('seasonal', json_p) for json_p in (seasonal_json_list or config.SEASONAL_PPA_JSON_LIST)]
        scenarios.extend(seasonal_scenarios)
        if verbose:
            print(f"📋 Added {len(seasonal_scenarios)} seasonal PPA scenarios")
    
    # Add preloaded profile scenarios
    if config.USE_PRELOADED_PPA_PROFILES:
        preloaded_scenarios = [('preloaded', csv_file) for csv_file in (preloaded_ppa_list or config.PRELOADED_PPA_PROFILES_LIST)]
        scenarios.extend(preloaded_scenarios)
        if verbose:
            print(f"📋 Added {len(preloaded_scenarios)} preloaded PPA scenarios")

    if not scenarios:
        raise ValueError("No scenarios to run. Please enable at least one profile type in config.py")

    if verbose:
        print(f"🚀 Total scenarios to process: {len(scenarios)}")

    for scenario_type, scenario_value in scenarios:
        
        # This logic should be carefully reviewed.
        # It permanently modifies the config values for subsequent runs in the same session.
        capacity_poi = config.CAPACITY_POI
        transformer_losses = config.TRANSFORMER_LOSSES if config.TRANSFORMER_LOSSES else 0

        if transformer_losses > 0:
            capacity_poi *= (1 + transformer_losses)

        data_handler = DataHandler(start_date=start_date, end_date=end_date, verbose=verbose)

        if scenario_type == 'flat':
            baseload_mw = scenario_value
            if verbose:
                print(f"\n🎯 Processing flat baseload scenario: {baseload_mw} MW")
            data_handler.load_data(
                generate_ppa_profile=True,
                baseload_mw=baseload_mw,  # Pass original value, transformer losses applied in data_handler
                transformer_losses=transformer_losses
            )
            # For flat profiles, peak MW includes transformer losses
            peak_mw = baseload_mw * (1 + transformer_losses) if transformer_losses > 0 else baseload_mw
            scenario_name = f"flat_{baseload_mw}MW"
            
        elif scenario_type == 'seasonal':
            json_filename = scenario_value
            if verbose:
                print(f"\n🎯 Processing seasonal PPA scenario: {json_filename}")
            data_handler.load_data(
                generate_ppa_profile=False,
                generate_seasonal_ppa=True,
                seasonal_json_filename=json_filename,
                peak_start=peak_start,
                peak_end=peak_end,
                transformer_losses=transformer_losses
            )
            # Calculate peak MW from the generated profile
            peak_mw = max(data_handler.ppa_profile)
            scenario_name = f"seasonal_{data_handler.profile_name}"
            
        elif scenario_type == 'preloaded':
            csv_filename = scenario_value
            if verbose:
                print(f"\n🎯 Processing preloaded PPA scenario: {csv_filename}")
            data_handler.load_data(
                generate_ppa_profile=False,
                preloaded_ppa_filename=csv_filename,
                transformer_losses=transformer_losses
            )
            # Calculate peak MW from the loaded profile
            peak_mw = max(data_handler.ppa_profile)
            scenario_name = f"custom_{Path(csv_filename).stem}_{peak_mw:.1f}MW"
        
        if verbose:
            print(f"   📊 Peak MW: {peak_mw:.1f}")
            print(f"   📝 Scenario name: {scenario_name}")
        
        # 2. Initialize and run optimizer
        minimizer = PPADeficitMinimizer(
            battery_params=battery_params,
            capacity_poi=capacity_poi,
            battery_degradation_factor=config.BATTERY_DEGRADATION_FACTOR,
            battery_degradation_profile=data_handler.battery_degradation_profile
        )
        
        results = minimizer.optimize(
            wind=data_handler.wind_profile, 
            solar=data_handler.solar_profile, 
            ppa=data_handler.ppa_profile, 
            timestamps=data_handler.timestamps, 
            verbose=verbose
        )

        # 3. Process and save results
        results_handler = ResultsHandler(data_handler=data_handler)
        deficits = results_handler.compute_deficits(
            results=results['results'], 
            ppa=data_handler.ppa_profile, 
            timestamps=data_handler.timestamps
        )
        results_handler.save_results(results=results, deficits=deficits, capacity_poi=capacity_poi, scenario_name=scenario_name)
        
        if verbose:
            print(f"✅ Results saved for {scenario_name} scenario\n")

    # Consolidate all results to a single Excel file
    if consolidate_excel and scenarios:
        ResultsHandler.consolidate_all_to_excel()

if __name__ == "__main__":
    main(
        baseload_mw_list=config.BASELOAD_MW_LIST, 
        seasonal_json_list=config.SEASONAL_PPA_JSON_LIST,
        preloaded_ppa_list=config.PRELOADED_PPA_PROFILES_LIST,
        peak_start=config.PEAK_START_HOUR,
        peak_end=config.PEAK_END_HOUR,
        consolidate_excel=config.CONSOLIDATE_EXCEL, 
        start_date=config.START_DATE, 
        end_date=config.END_DATE, 
        verbose=config.VERBOSE
    ) 

