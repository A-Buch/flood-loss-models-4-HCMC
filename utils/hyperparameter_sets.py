def main():
    {
        "LogisticRegression_hyperparameters":{
            "model__penalty": ["elasticnet"],
            "model__tol": [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5],
            "model__C": [1,2,3,4, 5, 6, 7, 8, 9],
            "model__max_iter": [1,2,3,4],
            "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
            "model__solver": ["saga"],
            "model__class_weight": [{0:0.30, 1:0.70}],
            "model__random_state": [42]
        },
        "XGBRegressor_hyperparameters":{
            "model__n_estimators": [ 3, 5, 10, 30, 50, 100, 250],
            "model__max_depth": [ 3, 5, 10, 15],
            "model__booster": ["gbtree"],
            "model__colsample_bytree": [0.33, 0.66, 1.0],
            "model__learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
            "model__seed":[42]
        },
        "SGDClassifier_hyperparameters":{
            "model__alpha": [0.0, 0.3, 0.5, 1, 3, 5],
            "model__max_iter": [2, 3, 4, 6, 8],
            "model__l1_ratio": [0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
            "model__tol": [0.1, 0.3, 0.5, 0.7, 1.0, 5.0],
            "model__random_state": [42]
        },
        "ElasticNet_hyperparameters":{
            "model__alpha": [0.5, 0.75, 1, 1.5, 3, 5],
            "model__max_iter": [2, 3, 4, 6, 8, 10],
            "model__l1_ratio": [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
            "model__tol": [0.1, 0.3, 0.5, 0.7, 1.0, 5.0],
            "model__selection": ["cyclic", "random"],
            "model__random_state": [42]
        },
        "cforest_hyperparameters":{
            "mtry" : [2, 18, 2]
        }
    }

    # if __name__ == "__main__":
        # main()