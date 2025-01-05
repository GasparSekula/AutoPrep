# Auto-Prep

**Auto-Prep** is an automated data preprocessing and analysis pipeline that generates comprehensive LaTeX reports. It handles common preprocessing tasks, creates insightful visualizations, and documents the entire process in a professional PDF report. It focuses on tabular data, supporting numerous explainable AI models. Emphasizing interpretability and ease of use, it includes subsections for each model, explaining their strengths, weaknesses, and providing usage examples.

## [Docs](https://pawlo77.github.io/AutoPrep/)

## Features

- **Automated data cleaning and preprocessing**
- **Intelligent feature type detection**
- **Advanced categorical encoding with rare category handling**
- **Comprehensive exploratory data analysis (EDA)**
- **Automated visualization generation**
- **Professional LaTeX report generation**
- **Modular and extensible design**
- **Support for numerous explainable AI models**
- **Explainability with model-specific examples**

## Report Contents

The generated report includes:

1. **Title page and table of contents**
2. **Overview**
   - Platform structure
   - Dataset structure
3. **Exploratory Data Analysis**
   - Distribution plots
   - Correlation matrix
   - Missing value analysis
4. **Model Performance**
   - Accuracy metrics
   - Model details

## Installation

### Using pip (Recommended)

1. Install Auto-Prep directly from PyPI:
    ```bash
    pip install auto-prep
    ```

2. Run the example usage:
    ```bash
    python example_usage.py
    ```

### Using Poetry

1. Ensure you have Poetry installed:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

2. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/auto-prep.git
    cd auto-prep
    ```

3. Install dependencies:
    ```bash
    poetry install
    ```

4. Activate the virtual environment:
    ```bash
    poetry shell
    ```

5. Run the example usage:
    ```bash
    python example_usage.py
    ```

## Important informations

- for changes in config to be loaded, config.update must be called before any other import from autoprep package - as example:

    ```python
    import logging
    from auto_prep.utils import config

    config.update(log_level=logging.DEBUG)

    import numpy as np

    from auto_prep.prep import AutoPrep
    from sklearn.datasets import fetch_openml

    # Load your dataset
    data = fetch_openml(name="titanic", version=1, as_frame=True, parser="auto").frame
    data["survived"] = data["survived"].astype(np.uint8)

    # Create and run pipeline
    pipeline = AutoPrep()
    pipeline.run(data, target_column="survived")
    ```

    For same reason AutoPrep is not exported to top-level package. It is known implementation fault.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Author

- **Paweł Pozorski** - [GitHub](https://github.com/Pawlo77)
- **Katarzyna Rogalska**
- **Julia Kruk**
- **Gaspar Sekula**

## Acknowledgments

- Inspired by the need for automated preprocessing and reporting in data science workflows
- Built with modern Python tools and best practices

## Notes for Developers

1. Poetry is used for dependency management and virtual environments. The following functions are implemented:
   - `poetry run format` - Format code
   - `poetry run lint` - Lint code
   - `poetry run check` - Check code
   - `poetry run test` - Run tests
   - `poetry run pre-commit run --all-files` - Run pre-commit hooks
