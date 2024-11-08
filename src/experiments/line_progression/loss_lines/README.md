# Loss Lines Visualization

A visualization tool for machine learning model training progression, specifically designed to show how models evolve during the training process. This project creates aesthetic visualizations of model fitting across different datasets.

## Overview

This project visualizes the training progression of polynomial regression models across three different datasets:

1. **US Gini Index** (1963-2022)
   - Tracks wealth inequality over time
   - Uses polynomial regression (degree 2)
   - Visualizes in gold/white on black background

2. **Global CO2 Emissions**
   - Historical carbon dioxide emissions data
   - Uses polynomial regression (degree 3)
   - Visualizes in black on white background

3. **Global Life Expectancy**
   - UN data on worldwide life expectancy
   - Uses polynomial regression (degree 2)
   - Visualizes in red/white on black background

## Features

- Visualizes every step of the training process
- Creates three versions of each visualization:
  - Training progression only
  - Training progression with final prediction
  - Full visualization with axes and original data
- Customizable visualization parameters:
  - Background colors
  - Line colors
  - Transparency
  - Axis visibility
  - Data point visibility

## Data Sources

- Gini Index: US Census Bureau data
- CO2 Emissions: Our World in Data (OWID)
- Life Expectancy: United Nations World Population Prospects

## Usage

Run any of the following functions in the main script:


```
python
Choose which visualization to run
run_for_gini() # US wealth inequality
run_for_carbon() # Global CO2 emissions
run_for_life_expectancy() # Global life expectancy
```


## Configuration

Each visualization can be customized with:
- Polynomial degree
- Learning rate
- Number of iterations
- Visual styling (colors, transparency, etc.)

## Dependencies

- NumPy
- Pandas
- Matplotlib
- Requests (for data downloading)

## Project Structure

- `data/`
  - `load/`: Data loading modules for each dataset
  - Raw data files
- `outputs/`: Generated visualizations
- `loss-lines.py`: Main execution script
- `training_visualisation.py`: Visualization utilities

## Notes

- The visualizations are designed for both aesthetic and analytical purposes
- Training progression shows the model's evolution from initial state to final fit
- Multiple output formats are generated for different use cases (presentation, analysis, etc.)
