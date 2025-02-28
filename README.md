# Technical Test

Author: Sharon Lizzet Trejo Luna

This project focuses on Exploratory Data Analysis (EDA) and regression model development. It includes a Jupyter notebook for performing data analysis and model training.

# Python Version
`3.12.7`

## Project Structure
```
project-root/ 
├── data/ 
├── pyproject.toml
├── poetry.lock
├── test_nb.ipynb 
├── funcs.py
├── .gitignore
├── README.md
└── utils.py
```

- `pyproject.toml`: Poetry configuration file defining project dependencies and metadata
- `poetry.lock`: Lock file ensuring reproducible installations
- `data/`: Directory to store input data files for analysis and modeling
- `test_nb.ipynb`: Jupyter notebook containing the main code for performing EDA and model development
- `funcs.py`: Python script with helper functions used in `test_nb.ipynb` to optimize analysis and modeling processes

# Installation

This project uses Poetry for dependency management. First, ensure you have Poetry installed.


# Usage

- `test_nb.ipynb`: This notebook contains all the steps for data analysis and model development, including:
  - Data Loading and Cleaning
  - Exploratory Data Analysis (EDA)
  - Model Training and Evaluation

`funcs.py`: Contains helper functions used in `test_nb.ipynb` to make the analysis more modular and organized.

## Requirements

* Python >= 3.8, Python < 3.13.0
* Poetry (recommended) or pip
* Required Python packages are managed through Poetry and defined in `pyproject.toml`
