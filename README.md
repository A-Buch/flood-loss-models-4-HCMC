# Modelling Flood Losses to Microbusinesses in Ho Chi Minh City, Vietnam

## Master thesis project: Modelling Flood Impacts of Microbusinesses in Ho Chi Minh City, Vietnam

## Abstract 
Microbusinesses are important sources of livelihood for low- and middle-income households.
In Ho Chi Minh City (HCMC), Vietnam, many microbusinesses are set up in the ground floor of residential houses susceptible to urban floods.
Increasing flood risk in HCMC threatens the financial resources of microbusinesses by damaging business contents and causing business interruption. Since flood loss estimations are rarely conducted at object-level resolution and are often focused on households or large companies, the losses suffered by microbusinesses are often overlooked.
This study aims to derive the drivers of flood losses in microbusinesses by applying a Conditional Random Forest to survey data (content losses: n=317; business interruption losses: n=361) collected from microbusinesses in HCMC.
The variability of content losses and business interruption were adequately explained by the revenues of the businesses from monthly sales, age of the building where the business is established and water depth in the building during the flood event.
Based on the identified drivers, probabilistic loss models (non-parametric Bayesian Networks) were developed using a combination of data-driven and expert-based model formulation.
The models estimated the flood losses for HCMC’s microbusinesses with a mean absolute error of 3.8 % for content losses and 18.7 % for business interruption losses. The Bayesian Network model for business interruption performed with a similar predictive performance when it was regionally transferred and applied to comparable survey data from another Vietnamese city, Can Tho.
The flood loss models introduced in this study make it possible to derive flood risk metrics specific to microbusinesses to support adaptation decision making and risk transfer mechanisms.

```
@Article{
        egusphere-2024-2340,
        AUTHOR = {Buch, A. and Paprotny, D. and Shahi, K. R. and Kreibich, H. and Sairam, N.},
        TITLE = {Modelling Flood Losses to Microbusinesses in Ho Chi Minh City, Vietnam},
        JOURNAL = {EGUsphere},
        VOLUME = {2024},
        YEAR = {2024},
        PAGES = {1--24},
        URL = {https://egusphere.copernicus.org/preprints/2024/egusphere-2024-2340/},
        DOI = {10.5194/egusphere-2024-2340}
}
```

## Objectives of this study
- Identify the main drivers of relative flood losses to microbusinesses in delta cities of South-East Asia. In particular the drivers of economic losses to business content (e.g. machinery, products, furniture) and losses due to interrupted business operations during or shortly after the flood event
- Estimating both economic losses and assessing the prediction uncertainties using probabilisitc models applied on the drivers identified. The models are validated in regard to their regional transferability (Can Tho City) and the model performances are benchmarked against a ML-based reference model.



## Run

- set project root as working directory: `export PYTHONPATH="src:$PYTHONPATH"`
- install packages defined in `pyproject.toml` via poetry: `poetry install` & `poetry shell` 
- Initially install pre-commit hooks: `poetry run pre-commit install`
- Install `R` and `RStudio` with following packages and their dependencies. Needed to train and evaluate Conditional Random Forest used for feature selection:
        - `nestedcv`  - *Nested Cross-Validation*
        - `pdp`  -  *Partial Dependencies*
        - `party` - *Conditional Random Forest*
        - `caret` - *Note: package version needs to be higher than >= 6.0-90*


## File description
**Abbreviation for relative interruption losses used in the files:** *rbred* \
**Abbreviations for relative content losses used in the files:** *chance of rcloss, degree of rcloss*\
\
**Note**: \
The modelling of content losses was split into two prediction tasks - the modelling of the occurrence of content loss (*chance of rcloss*) and the  degree of experienced content loss (*degree of rcloss*). It is an ensemble approach of two probabilistic models (Probabilistic Logistic Regression for the chance of loss and Bayesian Network for the degree of loss)\\
Files for the survey data preparation and the explorative analysis were saved as `Jupyter Notebooks` in order to enhance the understanding and reproducibility for the single steps. The same applies for the calibration and validation of the flood loss models, ecept for the reference model.
Files for the identification of the flood loss drivers were kept as `python scripts`

### Preprocessing and explorative analysis of the survey datasets
- `./data-preparation/hcmc_floor_numbers.ipynb`  - *enrich records of HCMC dataset with information about floor numbers*
- `./data-preparation/hcmc_preprocessing_exploration.ipynb`  -  *preprocessing and explorative analysis for the HCMC dataset*
- `./data-preparation/cantho_preprocessing_exploration.ipynb` -  *preprocessing and explorative analysis for the HCMC dataset*

### Selection of flood loss drivers
- `./feature-selection/feature_selection_classification.py`   - *chance of rcloss*
- `./feature-selection/feature_selection_regression_degree.py`  - *degree of rcloss*
- `./feature-selection/feature_selection_regression_rbred.py`  -  *rbred*

### Flood loss models
- `./bayesian_network/bayesian_network_rcloss.ipynb`  - *incl. validation of transferability to Can THo*
- `./bayesian_network/bayesian_network_rbred.ipynb`  - *incl. validation of transferability to Can THo*
- `./reference_model/reference_model_rf.py`  - *Random Forest model used for benchmarking the probabilistic flood loss models*

