"""
The ``modelchain_example`` module shows a simple usage of the windpowerlib by
using the :class:`~.modelchain.ModelChain` class. The modelchains are
implemented to ensure an easy start into the Windpowerlib. They work like
models that combine all functions provided in the library. Via parameters
desired functions of the windpowerlib can be selected. For parameters not being
specified default parameters are used.

There are mainly three steps. First you have to import your weather data, then
you need to specify your wind turbine, and in the last step call the
windpowerlib functions to calculate the feed-in time series.

Install the windpowerlib and optionally matplotlib to see the plots:

   pip install windpowerlib
   pip install matplotlib

Go down to the "run_example()" function to start the example.

SPDX-FileCopyrightText: 2019 oemof developer group <contact@oemof.org>
SPDX-License-Identifier: MIT
"""
import os
import pandas as pd
import requests
import logging
from windpowerlib import ModelChain, WindTurbine, create_power_curve

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None


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

def initialize_wind_turbine(wind_power_data:dict, **kwargs) -> WindTurbine:
    r"""
    Initializes three :class:`~.wind_turbine.WindTurbine` objects.

    This function shows three ways to initialize a WindTurbine object. You can
    either use turbine data from the OpenEnergy Database (oedb) turbine library
    that is provided along with the windpowerlib, as done for the
    'enercon_e126', or specify your own turbine by directly providing a power
    (coefficient) curve, as done below for 'my_turbine', or provide your own
    turbine data in csv files, as done for 'my_turbine2'.

    To get a list of all wind turbines for which power and/or power coefficient
    curves are provided execute `
    `windpowerlib.wind_turbine.get_turbine_types()``.

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
    Tuple (:class:`~.wind_turbine.WindTurbine`,
           :class:`~.wind_turbine.WindTurbine`,
           :class:`~.wind_turbine.WindTurbine`)

    """
    
    # ************************************************************************
    # **** Specification of wind turbine with your own data ******************
    # **** NOTE: power values and nominal power have to be in Watt

    if wind_power_data is None:
       raise ValueError("wind_power_data is required")

    #convet to series 
    wind_speed = pd.Series(wind_power_data["wind_speed"])
    power = pd.Series(wind_power_data["power"])



    turbine_specs = {
        "nominal_power": kwargs.get("nominal_power", None),  # in W
        "hub_height": kwargs.get("hub_height", None),  # in m
        "power_curve": create_power_curve(
            wind_speed=wind_speed, power=power
        )
    }

    if turbine_specs["nominal_power"] is None:
        raise ValueError("nominal_power is required")
    if turbine_specs["hub_height"] is None:
        raise ValueError("hub_height is required")

    turbine = WindTurbine(**turbine_specs)

    # ************************************************************************
    # **** Specification of wind turbine with data in own file ***************

    # Read your turbine data from your data file using functions like
    # pandas.read_csv().
    # >>> import pandas as pd
    # >>> my_data = pd.read_csv("path/to/my/data/file")
    # >>> my_power = my_data["my_power"]
    # >>> my_wind_speed = my_data["my_wind_speed"]

    return  turbine

def calculate_power_output(weather, turbine):
    r"""
    Calculates power output of wind turbines using the
    :class:`~.modelchain.ModelChain`.

    The :class:`~.modelchain.ModelChain` is a class that provides all necessary
    steps to calculate the power output of a wind turbine. You can either use
    the default methods for the calculation steps, as done for 'my_turbine',
    or choose different methods, as done for the 'e126'. Of course, you can
    also use the default methods while only changing one or two of them, as
    done for 'my_turbine2'.

    Parameters
    ----------
    weather : :pandas:`pandas.DataFrame<frame>`
        Contains weather data time series.
    turbine : :class:`~.wind_turbine.WindTurbine`
        WindTurbine object with self provided power curve.
    """

    # ************************************************************************
    # **** ModelChain with non-default specifications ************************
    modelchain_data = {
        "wind_speed_model": "logarithmic",  # 'logarithmic' (default),
        # 'hellman' or
        # 'interpolation_extrapolation'
        "density_model": "ideal_gas",  # 'barometric' (default), 'ideal_gas' or
        # 'interpolation_extrapolation'
        "temperature_model": "linear_gradient",  # 'linear_gradient' (def.) or
        # 'interpolation_extrapolation'
        "power_output_model": "power_curve",  # 'power_curve'
        # (default) or 'power_coefficient_curve'
        "density_correction": True,  # False (default) or True
        "obstacle_height": 0,  # default: 0
        "hellman_exp": None,
    }  # None (default) or None

    # ************************************************************************
    # **** ModelChain with non-default specifications ************************
    # initialize ModelChain with own specifications and use run_model method
    # to calculate power output
    custom_mc_turbine = ModelChain(turbine, **modelchain_data).run_model(weather)
    # write power output time series to WindTurbine object
    turbine.power_output = custom_mc_turbine.power_output

    # ************************************************************************
    # **** ModelChain with default parameter *********************************
    mc_turbine = ModelChain(turbine).run_model(weather)
    # write power output time series to WindTurbine object
    turbine.power_output = mc_turbine.power_output

    print(turbine.power_output)

    return 

def plot_or_print(turbine, plot=False):
    r"""
    Plots or prints power output and power (coefficient) curves.

    Parameters
    ----------
    my_turbine : :class:`~.wind_turbine.WindTurbine`
        WindTurbine object with self provided power curve.
    e126 : :class:`~.wind_turbine.WindTurbine`
        WindTurbine object with power curve from the OpenEnergy Database
        turbine library.
    my_turbine2 : :class:`~.wind_turbine.WindTurbine`
        WindTurbine object with power coefficient curve from example file.

    """

    # plot or print turbine power output
    if plt:
        turbine.power_output.plot(legend=True, label="Vestas V163")
        plt.xlabel("Time")
        plt.ylabel("Power in W")
        plt.show()
    else:
        print(turbine.power_output)

    # plot or print power curve
    if plot and plt:
        if turbine.power_curve is not False:
            turbine.power_curve.plot(
                x="wind_speed",
                y="value",
                style="*",
                title="Vestas V163 power curve",
            )
            plt.xlabel("Wind speed in m/s")
            plt.ylabel("Power coefficient $\mathrm{C}_\mathrm{P}$")
            plt.show()

    else:
        if turbine.power_curve is not False:
            print(turbine.power_curve)
            breakpoint()


def run_single_turbine():
    """
    Runs the basic example.

    """
    # You can use the logging package to get logging messages from the
    # windpowerlib. Change the logging level if you want more or less messages:
    # logging.DEBUG -> many messages
    # logging.INFO -> few messages
    logging.getLogger().setLevel(logging.DEBUG)

    data_path = "/Users/jjcosta/Desktop/optimize/modelos_optimizacion/inputs"


    wind_power_data = {
    "wind_speed": [3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24],
    "power":  [p * 1000 for p in [86.4, 189.0, 316.4, 472.8, 666.0, 902.0, 1182.8, 1513.4, 1889.0, 2337.2, 2830.4, 3358.6, 3862.2, 4207.2, 4394.6, 4462.4, 4492.6, 4498.8, 4500.0, 4500.0, 4500.0, 4500.0, 4500.0, 4500.0, 4500.0, 4500.0, 4500.0, 4500.0, 4493.0, 4440.0, 4303.0, 4115.0, 3920.0, 3714.0, 3494.0, 3268.0, 3047.0, 2836.0, 2636.0, 2440.0, 2248.0, 2070.0, 1939.0]]}

    weather = get_weather_data(datapath=data_path)
    v163 = initialize_wind_turbine(wind_power_data, nominal_power=4.5e6, hub_height=113)

    calculate_power_output(weather, v163)
    plot_or_print(v163)


if __name__ == "__main__":
    run_single_turbine()
