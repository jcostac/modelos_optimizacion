import os
import pandas as pd
import numpy as np
import calendar
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import INPUTS_PATH, BATTERY_DEGRADATION_FACTOR


class DataHandler:
    """
    Handles loading, processing, and validation of all input data for the optimization model.
    """
    def __init__(self, start_date=None, end_date=None, verbose=False):
        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose
        
        self.wind_profile = None
        self.solar_profile = None
        self.ppa_profile = None
        self.timestamps = None
        self.battery_degradation_profile = None

    def load_data(self, generate_ppa_profile: bool = False, baseload_mw: float = None, transformer_losses: float = 0):
        """
        Load data from the inputs folder. Can handle missing wind or solar profiles.
        Assumes CSVs have a datetime column as the first column and a value column as the second.
        If start_date and end_date are provided, the data will be filtered to only include data between those dates.

        Args:
            generate_ppa_profile (bool): Whether to generate a baseload profile or use a specific PPA profile
            baseload_mw (float): Constant baseload value in MW for generated profile.
            transformer_losses (float): Transformer losses to be added to the PPA profile.
        """
        paths = {
            'wind_profile': f'{INPUTS_PATH}/wind_profile.csv',
            'solar_profile': f'{INPUTS_PATH}/solar_profile.csv',
            'ppa_profile': f'{INPUTS_PATH}/ppa_profile.csv',
        }

        def _load_profile(path):
            if os.path.exists(path):
                df = pd.read_csv(path)
                df.rename(columns={df.columns[0]: 'datetime'}, inplace=True)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['datetime'] = df['datetime'].dt.floor('h') #extract hour
                return df
            print(f"Info: '{os.path.basename(path)}' not found. Assuming zero generation for this source.")
            return None

        wind_df = _load_profile(paths['wind_profile'])
        solar_df = _load_profile(paths['solar_profile'])

        if wind_df is None and solar_df is None:
            raise FileNotFoundError("No generation data found. At least 'wind_profile.csv' or 'solar_profile.csv' must exist in the inputs folder.")

        # Filter by date
        if self.start_date and self.end_date:
            start = pd.to_datetime(self.start_date)
            # Add a day to end_date to include all hours of the last day.
            end = pd.to_datetime(self.end_date) + pd.Timedelta(days=1)
            if wind_df is not None:
                wind_df = wind_df[(wind_df['datetime'] >= start) & (wind_df['datetime'] < end)].reset_index(drop=True)
            if solar_df is not None:
                solar_df = solar_df[(solar_df['datetime'] >= start) & (solar_df['datetime'] < end)].reset_index(drop=True)

        if wind_df is not None:
            self.check_complete_years(wind_df, 'wind', verbose=self.verbose)
        if solar_df is not None:
            self.check_complete_years(solar_df, 'solar', verbose=self.verbose)

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
            if not os.path.exists(paths['ppa_profile']):
                raise FileNotFoundError(f"PPA profile not found at {paths['ppa_profile']}. To generate a baseload profile, set generate_ppa_profile=True.")
            
            ppa_df = _load_profile(paths['ppa_profile'])

            if transformer_losses > 0 and transformer_losses is not None:
                ppa_df.iloc[:, 1] = ppa_df.iloc[:, 1] + (1*transformer_losses)
            
            # Align PPA with generation data timestamps
            ppa_df = ppa_df.set_index('datetime').reindex(ref_df['datetime']).reset_index()
            # Check for NaNs after reindexing if PPA file doesn't cover the full range
            if ppa_df.iloc[:, 1].isnull().any():
                print("Warning: PPA profile has missing values for some timestamps. Filling with 0.")
                ppa_df.iloc[:, 1].fillna(0, inplace=True)
            self.ppa_profile = ppa_df.iloc[:, 1].tolist()

        # Final length check
        if len(self.wind_profile) != len(self.solar_profile) or len(self.wind_profile) != len(self.ppa_profile):
            raise ValueError(f"Profile length mismatch after processing: Wind={len(self.wind_profile)}, Solar={len(self.solar_profile)}, PPA={len(self.ppa_profile)}")

        self._load_degradation_profile()
        self.timestamps = ref_df['datetime'].tolist()

    def _load_degradation_profile(self):
        """Loads and processes the battery degradation profile from a CSV file."""
        degradation_path = f'{INPUTS_PATH}/battery_degradation_profile.csv'
        if BATTERY_DEGRADATION_FACTOR is not None:
            if os.path.exists(degradation_path):
                raise ValueError("Cannot use both fixed BATTERY_DEGRADATION_FACTOR and degradation profile file.")
            print(f"Using fixed annual degradation factor: {BATTERY_DEGRADATION_FACTOR:.2%}")
            self.battery_degradation_profile = None # Explicitly set to None
            return

        if os.path.exists(degradation_path):
            df_deg = pd.read_csv(degradation_path)
            # Use first column as year, second as degradation factor
            year_col = df_deg.columns[0]
            deg_col = df_deg.columns[1]
            # Clean degradation factor: remove %, strip, handle blanks/dashes, convert to numeric
            df_deg[deg_col] = df_deg[deg_col].astype(str).str.replace('%', '').str.strip()
            # Treat '0%' as '-' (missing value)
            df_deg[deg_col] = df_deg[deg_col].replace('0', '-')
            df_deg[deg_col] = df_deg[deg_col].replace(['-', ' - ', ''], np.nan)
            df_deg[deg_col] = pd.to_numeric(df_deg[deg_col], errors='coerce')
            # If any value > 1, treat as percent and convert to fraction
            if not df_deg[deg_col].dropna().empty and df_deg[deg_col].max() > 1:
                df_deg[deg_col] = df_deg[deg_col] / 100
            # Forward fill and fill remaining NaNs with 1.0
            df_deg[deg_col] = df_deg[deg_col].ffill().fillna(1.0)
            # Ensure years are integers
            df_deg[year_col] = df_deg[year_col].astype(int)
            self.battery_degradation_profile = dict(zip(df_deg[year_col], df_deg[deg_col]))
            print(f"Loaded battery degradation profile with {len(self.battery_degradation_profile)} entries.")
        else:
            print("No battery degradation profile file found and BATTERY_DEGRADATION_FACTOR is not set. Assuming no degradation.")
            self.battery_degradation_profile = None

    def create_baseload_ppa(self, baseload_mw, profile_length):
        """
        Create a baseload PPA profile with constant value.
        
        Args:
            baseload_mw (float): Constant baseload value in MW
            profile_length (int): The length of the profile to generate.
            
        Returns:
            list: PPA profile with constant baseload value
        """
        if baseload_mw is None:
            raise ValueError("Baseload value must be provided when generating a PPA profile.")
        
        return [baseload_mw] * profile_length

    def check_complete_years(self, df, profile_name, verbose=False):
        """
        Checks if the data for each month in each year is complete (has the correct number of hours).
        Warns if any hours are missing and lists the missing hours.

        Args:
            df (pd.DataFrame): DataFrame with a 'datetime' column.
            profile_name (str): Name of the profile being checked (e.g., 'wind', 'solar').
            verbose (bool): Whether to print the missing hours.
        """
        if df is None or df.empty:
            return

        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month

        # Group by year and month and count the hours
        hourly_counts = df.groupby(['year', 'month']).size().reset_index(name='hours')

        for _, row in hourly_counts.iterrows():
            year, month, actual_hours = int(row['year']), int(row['month']), int(row['hours'])
            
            # Get the number of days in the month, accounting for leap years
            days_in_month = calendar.monthrange(year, month)[1]
            expected_hours = days_in_month * 24
            
            if actual_hours < expected_hours:
                missing_hours = expected_hours - actual_hours
                print(f"Warning: In '{profile_name}' profile, for {year}-{month:02d}, there are {missing_hours} missing hours. "
                      f"Expected {expected_hours}, but found {actual_hours}.")

                # Find missing hours
                if verbose:
                    start = pd.Timestamp(year=year, month=month, day=1, hour=0)
                    end = pd.Timestamp(year=year, month=month, day=days_in_month, hour=23)
                    all_hours = pd.date_range(start=start, end=end, freq='h')
                    present_hours = set(df[(df['year'] == year) & (df['month'] == month)]['datetime'])
                    missing_datetimes = sorted(list(set(all_hours) - present_hours))
                    if missing_datetimes:
                        print(f"  Missing hours for {year}-{month:02d}:")
                        for dt in missing_datetimes:
                            print(f"    - {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Clean up columns added
        df.drop(columns=['year', 'month'], inplace=True) 