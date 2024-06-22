# flood-loss-models-4-HCMC

## Master thesis project: Modelling Flood Impacts of Microbusinesses in Ho Chi Minh City, Vietnam**

### Abstract
Very small companies, known as microbusinesses, are essential for the local economy of delta-cities such as Ho Chi Minh City (HCMC) in Vietnam. These businesses are important sources of income for low- and middle-income households, however the rising frequency and intensity of flood events in these delta-cities poses a continues threat to them, that directly threaten their financial resources.  The high level of flood-adapted behaviour among microbusinesses already reduces their experienced economic losses, still they are exposed to losses to business contents and interruption in the business processes due to the flood event. Since flood loss estimations are rarely conducted at object-level resolution and are often focused on households or large companies, the impact suffered by microbusinesses are overlooked and have not been quantified before. Using an empirical survey dataset with 324 records about content losses and 361 records of business interruption losses from microbusinesses in HCMC, we use a Conditional Random Forest (CRF) to understand flood loss processes. The revenues from monthly sales, building age and water depth in the building were found to drive the relative flood losses of microbusinesses. Based on the identified drivers, probabilistic loss models (Bayesian Networks) were developed consisting of a combination of data-driven and expert-based model formulation. The estimation of frequent occurred cases of only small economic losses or even an absence of loss remains challenging for the probabilistic models and could be partly solved by an ensemble modelling approach. The regional transferability of the probabilistic models is validated using empirical data from another delta-city, namely Can Tho City. The models estimated the flood losses for HCMC’s microbusinesses moderately well with a Mean Absolute Error (MAE) of 3.8% for relative content losses and MAE of 18.7% for interruption-related losses. However, only the probabilistic model for interruption losses was regionally transferable and led to prediction errors of comparable magnitudes as when applied to the HCMC samples. The results of this study show that the main drivers of flood losses to microbusinesses differ from those of larger companies. Therefore, separate modelling of flood losses to very small businesses may be beneficial in developing more comprehensive flood risk management.


## Objectives of this study
- Identification of the main drivers of relative flood losses to microbusinesses in South Vietnam. In particular the drivers of economic losses to business content (e.g. machinery, products, furniture) and losses due to interrupted business operations during or shortly after the flood event (rbred, e.g. disrupted supply chains, reduced sales and production)
- Providing an approach for estimating these two economic losses by probabilisitc models based on the identified drivers. The models are validated in regard to their regional transferability (Can Tho City) and the model performances are benchmarked against a ML-based reference model. Furthermore, the uncertainities in the probabilistic estimates are assessed.


### File description
*rbred: interruption losses* \\
*chance and degree of rcloss: content losses* \\
\\
**Preprocessing of Survey Datasets**\\
./data_preprocessing/HCMC_geolocations.ipynb \\
./data_preprocessing/data_cleaning.ipynb \\
./data_preprocessing/cantho_data_cleaning_and_HCMC_comparison.ipynb \\
\\
**Selection of Flood Loss Drivers**\\
./feature-selection/feature_selection_classification.py   - *chance of rcloss* \\
./feature-selection/feature_selection_regression_degree.py  - *degree of rcloss* \\
./feature-selection/feature_selection_regression_rbred.py  \\
\\
**Flood Loss Models** - *incl. validation of transferability to Can THo*\\
./bayesian_network/bayesian_network_rcloss_hcmc.ipynb \\
./bayesian_network/bayesian_network_rbred_hcmc.ipynb \\
./reference_model/reference_model_rf.*\\
