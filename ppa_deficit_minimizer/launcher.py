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
from ppa_deficit_minimizer.ppa_deficit_minimizer_v3 import PPADeficitMinimizer

def main_baseload(baseload_mw_list: list[float], consolidate_excel: bool = True, start_date: str = None, end_date: str = None, verbose: bool = True):
    """
    Main function to run baseload scenarios.
    
    Args:
        baseload_mw_list (list[float]): List of baseload MW values to run scenarios for.
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

    for baseload_mw in baseload_mw_list:
        
        # This logic should be carefully reviewed.
        # It permanently modifies the config values for subsequent runs in the same session.
        capacity_poi = config.CAPACITY_POI
        current_baseload_mw = baseload_mw

        if config.TRANSFORMER_LOSSES and config.TRANSFORMER_LOSSES > 0:
            capacity_poi = capacity_poi * (1 + config.TRANSFORMER_LOSSES)
            current_baseload_mw = current_baseload_mw * (1 + config.TRANSFORMER_LOSSES)

        if verbose:
            print(f"\n🎯 Processing baseload scenario: {baseload_mw} MW")
            
        # 1. Load data
        data_handler = DataHandler(start_date=start_date, end_date=end_date, verbose=verbose)
        data_handler.load_data(
            generate_ppa_profile=config.GENERATE_PPA_PROFILE,
            baseload_mw=current_baseload_mw,
            transformer_losses=config.TRANSFORMER_LOSSES
        )
        
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
        results_handler.save_results(results=results, deficits=deficits, capacity_poi=capacity_poi)
        
        if verbose:
            print(f"✅ Results saved for {baseload_mw} MW baseload scenario\n")

    # Consolidate all results to a single Excel file
    if consolidate_excel and baseload_mw_list:
        ResultsHandler.consolidate_all_to_excel()

if __name__ == "__main__":
    main_baseload(
        baseload_mw_list=config.BASELOAD_MW_LIST, 
        consolidate_excel=config.CONSOLIDATE_EXCEL, 
        start_date=config.START_DATE, 
        end_date=config.END_DATE, 
        verbose=config.VERBOSE
    ) 