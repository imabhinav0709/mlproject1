import numpy as np
import pandas as pd
import sys 
import os

from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")

            X_train,X_test,y_train,y_test = (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )

            models= {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbours Regressor': KNeighborsRegressor(),
                'Adaboost Regressor': AdaBoostRegressor(),
                'XGBoost': XGBRegressor(),
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,
                                          X_test=X_test,y_test=y_test,models=models)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[ 
                list(model_report.values()).index(best_model_score)
            ]
            #best_model_name = max(model_report, key=model_report.get)
            #best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("NO best model is found",sys)
            
            logging.info(f"Best model: --{best_model_name}-- and score: --{best_model_score}--")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2score = r2_score(y_test,predicted)

            return (
                best_model_name,
                r2score,
            )

        except Exception as e:
            raise CustomException(e,sys)