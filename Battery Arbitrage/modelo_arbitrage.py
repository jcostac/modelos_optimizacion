import pulp
import pandas as pd
import numpy as np
import sys
from pathlib import Path
# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import PATHS

class BatteryArbitrageOptimizer:
    def __init__(self, charge_rate, energy_capacity, initial_storage = 0, efficiency=0.9):
        """
        Initialize battery optimizer with parameters from mathematical formulation
        
        Args:
            max_power (float): Maximum power that can be sold/bought [MW]
            max_storage (float): Maximum energy storage capacity [MWh]
            initial_storage (float): Initial storage charge state [MWh]
        """
        self.max_power = charge_rate * efficiency   # Maximum power sold/bought = charge rate * efficiency (assume same efficiency for charge and discharge)
        self.max_storage = energy_capacity  # Maximum storage capacity = energy capacity
        self.initial_storage = initial_storage      # Initial storage state

    def optimize(self, prices: list[float]) -> dict:
        """
        Optimize battery operation for given price series
        
        Args:
            prices (list[float]): Price at time t ∈ T [€/MWh]
            
        Returns:
            dict: Optimization results containing power sold and energy stored profiles
        """
        T = len(prices)  # Set de horas
        
        # Create optimization model
        model = pulp.LpProblem("Battery_Arbitrage", pulp.LpMaximize)
        
        # Variables as defined in mathematical formulation
        # power at time t (can be negative, positive or zero --> charge/discharge)
        power = pulp.LpVariable.dicts("power", range(T), lowBound=-self.max_power, upBound=self.max_power)
        # energy stored at time t (can be positive or zero)
        stored_energy = pulp.LpVariable.dicts("stored_energy", range(T), lowBound=0, upBound=self.max_storage)
        
        ### Objective Function: Maximize revenue from power sold
        model += pulp.lpSum([power[t] * prices[t] for t in range(T)])
        
        ### Constraints ####
        # Initial storage state for hour 0
        model += stored_energy[0] == self.initial_storage - power[0]  # Energy balance at t=0
        
        # Storage dynamics for t > 0
        for t in range(1, T): #for all hours except the first one
            # Energy balance: stored energy at t is the stored energy at t-1 minus the power at t
            model += stored_energy[t] == stored_energy[t-1] - power[t]
            #other constraints should be that battery soc cannot be under 20% or over 80% (maintain battery health)
            model += stored_energy[t] >= 0.2 * self.max_storage
            model += stored_energy[t] <= 0.8 * self.max_storage
        
        # Solve the model
        model.solve()
        
        # Extract results
        results = {
            'power_sold': [power[t].value() for t in range(T)],  # power at t
            'energy_stored': [stored_energy[t].value() for t in range(T)],  # stored energy at t
            'revenue': pulp.value(model.objective), # revenue from power sold
            'status': pulp.LpStatus[model.status]
        }
        
        return results

    def calculate_metrics(self, results: dict, prices: list[float]) -> dict:
        """
        Calculate performance metrics including revenue per MWh and MW
        """
        # Calculate buying and selling separately
        power_bought = [-min(0, w) for w in results['power_sold']]  # When w_t < 0 --> charging the battery (buying)
        power_sold = [max(0, w) for w in results['power_sold']]     # When w_t > 0 --> discharging the battery (selling)
        
        buying_cost = sum(b * p for b, p in zip(power_bought, prices))
        selling_revenue = sum(s * p for s, p in zip(power_sold, prices))
        net_profit = selling_revenue - buying_cost
        
        # Calculate revenue per unit of capacity
        profit_per_mwh = net_profit / self.max_storage  # Revenue per MWh of storage
        profit_per_mw = net_profit / self.max_power   # Revenue per MW of power capacity
        
        cycles = sum(power_bought) / self.max_storage #full charge discharge cycles 
        
        return {
            'net_profit': net_profit,
            'buying_cost': buying_cost,
            'selling_revenue': selling_revenue,
            'profit_per_mwh': profit_per_mwh,
            'profit_per_mw': profit_per_mw,
            'cycles': cycles,
            'average_storage': np.mean(results['energy_stored'])
        }

def load_prices(file_path: str) -> pd.DataFrame:
    """
    Load prices from CSV file
    
    Args:
        file_path (str): Path to CSV file with columns ['FECHA', 'HORA', 'PRECIO']
        
    Returns:
        pd.DataFrame: DataFrame with datetime index and prices
    """
    df = pd.read_csv(file_path)
    df['FECHA'] = pd.to_datetime(df['FECHA']) #convert the fecha column to a datetime object
    #create a new column with the datetime of the fecha and hora in format YYYY-MM-DD HH:MM:SS
    df['DATETIME'] = pd.to_datetime(df['FECHA'].astype(str) + ' ' + df['HORA'].astype(str) + ':00:00')
    df = df.sort_values('DATETIME') #sort the dataframe by the datetime column

    return df

def calculate_revenue_metrics(optimizer: BatteryArbitrageOptimizer, prices_df: pd.DataFrame, window_size: int = 24) -> pd.DataFrame:
    """Calculate revenue metrics (net revenue, costs, cycles etc.) for each time window of prices data.
    
    Args:
        optimizer (BatteryArbitrageOptimizer): Battery optimizer instance to run simulations
        prices_df (pd.DataFrame): DataFrame with price data and datetime index
        window_size (int, optional): Size of each time window in hours. Defaults to 24.
        
    Returns:
        pd.DataFrame: DataFrame with revenue metrics for each time window
    """
    
    results_list = []
    
    total_hours = len(prices_df)
    n_windows = total_hours // window_size
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        
        spliced_prices = prices_df.iloc[start_idx:end_idx]['PRECIO'].tolist()
        start_datetime = prices_df.iloc[start_idx]['DATETIME']
        
        # Run optimization for window
        results = optimizer.optimize(spliced_prices)
        
        metrics = optimizer.calculate_metrics(results, spliced_prices)
        
      
        results_list.append({
            'datetime': start_datetime,
            'net_profit': metrics['net_profit'],
            'buying_cost': metrics['buying_cost'],
            'selling_revenue': metrics['selling_revenue'],
            'profit_per_mwh': metrics['profit_per_mwh'],
            'profit_per_mw': metrics['profit_per_mw'],
            'cycles': metrics['cycles']
        })
    
    # Create DataFrame with results
    results_df = pd.DataFrame(results_list)
    results_df.set_index('datetime', inplace=True)
    results_df['year'] = results_df.index.year
    results_df['month'] = results_df.index.month
    
    return results_df

if __name__ == "__main__":
    
    prices_file = PATHS['downloads']['price_data']
    
    # Load prices into DataFrame and process for correct datetime format
    prices_df = load_prices(prices_file)
    
    # Initialize optimizer with parameters matching mathematical formulation
    optimizer = BatteryArbitrageOptimizer(
        charge_rate=5,  # charge and discharge power of the battery [MW]
        energy_capacity=20, # Maximum storage capacity [MWh]
    ) # default efficiency of the battery is 0.9 and initial storage is 0 MWh
    
    # Calculate revenue metrics
    print("Optimizing and calculating revenue metrics...")
    revenue_df = calculate_revenue_metrics(optimizer, prices_df)
    
    # Display results
    print("\nMonthly Revenue Statistics:")
    monthly_stats = revenue_df.groupby(['year', 'month']).agg({
        'net_profit': 'sum',
        'profit_per_mwh': 'sum',
        'profit_per_mw': 'sum',
        'cycles': 'sum'
    }).round(2)
    print(monthly_stats)
    
    print("\nYearly Revenue Statistics:")
    yearly_stats = revenue_df.groupby('year').agg({
        'net_profit': 'sum',
        'profit_per_mwh': 'sum',
        'profit_per_mw': 'sum',
        'cycles': 'sum'
    }).round(2)
    print(yearly_stats)
    
    # Save results
    revenue_df.to_csv(PATHS['optimization']['results'])
