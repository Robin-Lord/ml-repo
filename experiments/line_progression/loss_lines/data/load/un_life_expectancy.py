"""
Module to help load the life expectancy data from the United Nations data dump

From https://population.un.org/wpp/Download/Standard/MostUsed/

"""

import pandas as pd

# For type setting
from typing import Tuple
import numpy as np


def fetch_life_expectancy_data(
    filepath="experiments/line_progression/loss_lines/data/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx",
) -> Tuple[np.array, np.array]:
    """
    Function to load life expectancy data

    args:
        filepath [str]: (default ="WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx")
                        the location from which to load the file

    returns:
        np.array: the years we have data for
        np.array: the life expectancy value for each year

    """
    df = pd.read_excel(filepath, sheet_name="Estimates", skiprows=16)
    df_global = df[df["Region, subregion, country or area *"] == "World"]

    # Filter for global data and forest area (acres/hectares)
    df_global = df_global[
        ["Year", "Life Expectancy at Birth, both sexes (years)"]
    ]  # Use year and forest area columns
    df_global = df_global.dropna().reset_index(
        drop=True
    )  # Remove rows with missing data
    df_global.columns = ["Year", "Life Expectancy"]
    X = df_global[["Year"]].values
    y = df_global[["Life Expectancy"]].values

    return X, y
