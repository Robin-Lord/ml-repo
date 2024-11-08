"""
Module to handle loading and basic initial restructure of Our World In Data CO2 tracking from


https://github.com/owid/co2-data

Related:
https://ourworldindata.org/explorers/co2?country=~OWID_WRL&Gas+or+Warming=CO%E2%82%82&Accounting=Territorial&Fuel+or+Land+Use+Change=All+fossil+emissions&Count=Per+capita

"""

import pandas as pd


def fetch_global_co2_emissions_data(
    filepath="src/experiments/line_progression/loss_lines/data/owid-co2-data.csv",
):
    df = pd.read_csv(filepath)

    # Filter for global data and CO₂ emissions
    df_global = df[df["country"] == "World"]
    df_global = df_global[["year", "co2"]]  # Use year and CO2 columns
    df_global = df_global.dropna().reset_index(
        drop=True
    )  # Remove rows with missing data
    df_global.columns = ["Year", "CO2 Emissions"]

    X = df_global[["Year"]].values
    y = df_global[["CO2 Emissions"]].values

    return X, y
