"""
Module to support downloading gini index data

"""

import requests  # type: ignore
import pandas as pd


def fetch_gini_index_data(country_code="USA"):
    """
    Download Gini index data for a specific country (e.g., United States) from the World Bank API

    args:
        country_code [str]: (default = "USA") the country code to use with the World Bank API

    """
    # First try to load from local
    try:
        return pd.read_csv("gini_us.csv")
    except Exception:
        # If not possible - download first
        url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/SI.POV.GINI?format=json&per_page=100"
        response = requests.get(url)
        data = response.json()

        # Convert data to DataFrame
        records = []
        for entry in data[1]:
            year = int(entry["date"])
            gini_index = entry["value"]
            if gini_index is not None:
                records.append((year, gini_index))
        df = pd.DataFrame(records, columns=["Year", "Gini Index"])
        df.to_csv("gini_us.csv")
        return df.sort_values(by="Year")
