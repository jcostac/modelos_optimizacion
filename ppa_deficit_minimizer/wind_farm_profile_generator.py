"""
The ``turbine_cluster_modelchain_example`` module shows how to calculate the
power output of wind farms and wind turbine clusters with the windpowerlib.
A cluster can be useful if you want to calculate the feed-in of a region for
which you want to use one single weather data point.

Functions that are used in the ``modelchain_example``, like the initialization
of wind turbines, are imported and used without further explanations.

SPDX-FileCopyrightText: 2019 oemof developer group <contact@oemof.org>
SPDX-License-Identifier: MIT
"""
import os
import pandas as pd
import requests
import logging
import numpy as np
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

from windpowerlib import WindFarm, WindTurbine, create_power_curve
from windpowerlib import WindTurbineCluster
from windpowerlib import TurbineClusterModelChain

# You can use the logging package to get logging messages from the windpowerlib
# Change the logging level if you want more or less messages
import logging

logging.getLogger().setLevel(logging.INFO)


def get_weather_data(filename="weather.csv", **kwargs) -> pd.DataFrame:
    r"""
    Imports weather data from a file.
    The data include wind speed at two different heights in m/s, air
    temperature in two different heights in K, surface roughness length in m
    and air pressure in Pa. The height in m for which the data applies is
    specified in the second row.
    In case no weather data file exists, an example weather data file is
    automatically downloaded and stored in the same directory as this example.
    Parameters
    ----------
    filename : str
        Filename of the weather data file. Default: 'weather.csv'.
    Other Parameters
    ----------------
    datapath : str, optional
        Path where the weather data file is stored.
        Default is the same directory this example is stored in.
    Returns
    -------
    :pandas:`pandas.DataFrame<frame>`
        DataFrame with time series for wind speed `wind_speed` in m/s,
        temperature `temperature` in K, roughness length `roughness_length`
        in m, and pressure `pressure` in Pa.
        The columns of the DataFrame are a MultiIndex where the first level
        contains the variable name as string (e.g. 'wind_speed') and the
        second level contains the height as integer at which it applies
        (e.g. 10, if it was measured at a height of 10 m). The index is a
        DateTimeIndex.
    """

    if "datapath" not in kwargs:
        kwargs["datapath"] = os.path.dirname(__file__)

    file = os.path.join(kwargs["datapath"], filename)

    # download example weather data file in case it does not yet exist
    if not os.path.isfile(file):
        logging.debug("Download weather data for example.")
        req = requests.get("https://osf.io/59bqn/download")
        with open(file, "wb") as fout:
            fout.write(req.content)

    # read csv file
    weather_df = pd.read_csv(
        file,
        index_col=0,
        header=[0, 1],
    )
    weather_df.index = pd.to_datetime(weather_df.index, utc=True)

    # change time zone
    weather_df.index = weather_df.index.tz_convert("Europe/Madrid")

    return weather_df


def create_wind_turbine(wind_power_data: dict = None, **kwargs) -> WindTurbine:
    r"""
    Initializes a :class:`~.wind_turbine.WindTurbine` object from a dictionary.
    Parameters
    ----------
    wind_power_data : dict
        Dictionary with wind power data.
        Keys: 'wind_speed', 'power'
        Values: list of wind speeds, list of power values
    Other Parameters
    ----------------
    nominal_power : float
        Nominal power of the wind turbine in Watt.
    hub_height : float
        Hub height of the wind turbine in meters.
    Returns
    -------
    :class:`~.wind_turbine.WindTurbine`
    """
    if wind_power_data is not None:
        # Validate wind_power_data keys
        if not isinstance(wind_power_data, dict):
            raise ValueError("wind_power_data must be a dictionary.")
        if "wind_speed" not in wind_power_data or "power" not in wind_power_data:
            raise ValueError("wind_power_data must contain 'wind_speed' and 'power' keys.")

        try:
            wind_speed = pd.Series(wind_power_data["wind_speed"])
            power = pd.Series(wind_power_data["power"])
        except Exception as e:
            raise ValueError(f"Error converting wind_power_data to pandas Series: {e}")

        nominal_power = kwargs.get("nominal_power")
        hub_height = kwargs.get("hub_height")

        if nominal_power is None:
            raise ValueError("nominal_power is required")
        if hub_height is None:
            raise ValueError("hub_height is required")

        power_curve = create_power_curve(wind_speed=wind_speed, power=power)
        turbine = WindTurbine(
            nominal_power=nominal_power,
            hub_height=hub_height,
            power_curve=power_curve,
        )
    else:
        
        turbine_type = kwargs.get("turbine_type")
        hub_height = kwargs.get("hub_height")

        if hub_height is None:
            raise ValueError("hub_height is required")
        if turbine_type is None:
            raise ValueError("turbine_type is required")

        turbine = WindTurbine(
            turbine_type=turbine_type,
            hub_height=hub_height,
        )

    return turbine


def create_wind_farm(farm_name: str, turbine_configs: list) -> WindFarm:
    """
    Creates a WindFarm object from a list of turbine configurations.
    Parameters
    ----------
    farm_name : str
        Name of the wind farm.
    turbine_configs : list of dicts
        List of dictionaries, where each dictionary specifies a turbine type
        and its count or total capacity in the farm.
    Returns
    -------
    windpowerlib.WindFarm
    """
    turbines = []
    n_turbines = []
    total_capacities = []

    for config in turbine_configs:
        # Allow for absence of wind_power_data in config
        wind_power_data = config.get("wind_power_data", None)
        # For generic turbines, require turbine_type
        turbine_type = config.get("turbine_type", None)
        nominal_power = config.get("nominal_power", None)
        hub_height = config.get("hub_height", None)

        if wind_power_data is not None:
            turbine = create_wind_turbine(
                wind_power_data=wind_power_data,
                nominal_power=nominal_power,
                hub_height=hub_height,
            )
        else:
            # If wind_power_data is not provided, require turbine_type
            if turbine_type is None:
                raise ValueError("If 'wind_power_data' is not provided, 'turbine_type' must be specified in the config.")
            turbine = create_wind_turbine(
                turbine_type=turbine_type,
                hub_height=hub_height,
            )

        number_of_turbines = config.get("number_of_turbines")
        total_capacity = config.get("total_capacity")
        if (number_of_turbines is not None and total_capacity is not None):
            raise ValueError("Specify either 'number_of_turbines' or 'total_capacity', not both, in each turbine config.")
        turbines.append(turbine)
        n_turbines.append(number_of_turbines if number_of_turbines is not None else np.nan)
        total_capacities.append(total_capacity if total_capacity is not None else np.nan)

    wind_turbine_fleet = pd.DataFrame(
        {
            "wind_turbine": turbines,
            "number_of_turbines": n_turbines,
            "total_capacity": total_capacities,
        }
    )

    return WindFarm(name=farm_name, wind_turbine_fleet=wind_turbine_fleet)


def calculate_power_output(weather, wind_farm):
    r"""
    Calculates power output of a wind farm.
    The :class:`~.turbine_cluster_modelchain.TurbineClusterModelChain` is a
    class that provides all necessary steps to calculate the power output of a
    wind farm or cluster. The power output in W is stored in the `wind_farm`
    object and the power output in MW is returned.
    Parameters
    ----------
    weather : :pandas:`pandas.DataFrame<frame>`
        Contains weather data time series.
    wind_farm : :class:`~.wind_farm.WindFarm`
        WindFarm object.
    Returns
    -------
    pandas.Series
        Time series of the wind farm's power output in MW.
    """
    # The TurbineClusterModelChain can be used for single farms as well.
    mc_wind_farm = TurbineClusterModelChain(wind_farm).run_model(weather)
    # write power output time series to WindFarm object
    wind_farm.power_output = mc_wind_farm.power_output

    # Convert power output from W to MW and return
    return wind_farm.power_output / 1e6


def plot_or_print(wind_farm, plot=True):
    r"""
    Plots or prints power output of a wind farm.
    Parameters
    ----------
    wind_farm : :class:`~.wind_farm.WindFarm`
        WindFarm object.
    plot : bool
        If True, plots the power output. Otherwise, prints it.
    """

    # plot or print power output
    if plot and plt:
        wind_farm.power_output.plot(legend=True, label=wind_farm.name)
        plt.xlabel("Time")
        plt.ylabel("Power in W")
        plt.show()
    else:
        print(wind_farm.power_output)


def run_wind_farm_profile_generator():
    r"""
    Runs the wind farm example with a configurable set of turbines.
    """
    # Define turbine power curve data
    # Vestas V163 - power in W
    v163_power_data = {
        "wind_speed": [
            3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10,
            10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16,
            16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22,
            22.5, 23, 23.5, 24,
        ],
        "power": [
            p * 1000 for p in [
                86.4, 189.0, 316.4, 472.8, 666.0, 902.0, 1182.8, 1513.4,
                1889.0, 2337.2, 2830.4, 3358.6, 3862.2, 4207.2, 4394.6,
                4462.4, 4492.6, 4498.8, 4500.0, 4500.0, 4500.0, 4500.0,
                4500.0, 4500.0, 4500.0, 4500.0, 4500.0, 4500.0, 4493.0,
                4440.0, 4303.0, 4115.0, 3920.0, 3714.0, 3494.0, 3268.0,
                3047.0, 2836.0, 2636.0, 2440.0, 2248.0, 2070.0, 1939.0,
            ]
        ],
    }

    # ** Configure your wind farm here **
    # You can define multiple turbine types with different counts or total capacities.
    farm_config = [
        {
            "wind_power_data": v163_power_data,
            "nominal_power": 4.5e6,  # W
            "hub_height": 113,  # m
            "number_of_turbines": 8, # can either be number of turbines or total capacity
        }
    ]

    # Get weather data
    # Make sure to place your 'weather.csv' in the 'inputs' directory
    data_path = os.path.join(os.path.dirname(__file__), "..", "inputs")
    weather = get_weather_data(filename="weather.csv", datapath=data_path)

    # Create wind farm
    my_wind_farm = create_wind_farm(farm_name="my_flexible_farm", turbine_configs=farm_config)

    # Calculate power output
    power_output_mw = calculate_power_output(weather, my_wind_farm)

    # Calculate total energy production in MWh
    total_output = power_output_mw.sum() 
    print(f"\nTotal energy production: {total_output:,.2f} MWh")

    # Calculate average power output
    average_output = total_output / 8760
    print(f"\nAverage power output: {average_output:,.2f} MW")

    # Plot results
    plot_or_print(my_wind_farm, plot=False)


if __name__ == "__main__":
    run_wind_farm_profile_generator()