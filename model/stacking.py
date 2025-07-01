from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
import miceforest as mf
import xgboost as xgb
import os
import pandas as pd
import numpy as np
import itertools
import time
from tinydb import TinyDB, Query
import warnings
import pickle
from preparation.clean import ColumnSet

# its not bidirectional but good enough 
def compare_dicts(dict1, dict2):
    for key, value in dict1.items():
        if key in dict2:
            if isinstance(value, tuple) or isinstance(value, list):
                if list(value) != list(dict2[key]):
                    return False
            elif value != dict2[key]:
                return False
        else:
            return False
    return True


class RegressionModel:
    def __init__(self, path="db.json", columns: ColumnSet = ColumnSet.EXTENDED, diagnosis: bool =True):
        self.db = TinyDB(f"{os.environ["OUT_DIR"]}{columns}_{diagnosis}_{path}")
        self.columns = columns
        self.diagnosis = diagnosis
        self.estimators = [
        ('lr', ElasticNet()),
        ('xgb', xgb.XGBRegressor(
            random_state=42,
            enable_categorical=True
            )),
        ('mlp', MLPRegressor(
            random_state=19,
            max_iter=10000
            )),
        ('svc', LinearSVR(
            random_state=198))
        ]

        self.param_grid = {
            'lr__l1_ratio': [0.0, 0.5, 1.0],
            'svc__C': [0.01, 1.0],
            'mlp__learning_rate_init': [1e-4, 1e-2],
            'mlp__hidden_layer_sizes': [(100, 100), (500, 500, 500)],
            # 'xgb__n_estimators': [20, 1000],
            'xgb__max_depth':[3, 6, 9]
        }

        self.filename = f"{os.environ["OUT_DIR"]}model.pickle"
        self.training = f"{os.environ["OUT_DIR"]}{columns}_{diagnosis}_train_trunc.parquet"
        self.validation = f"{os.environ["OUT_DIR"]}{columns}_{diagnosis}_validation_trunc.parquet"
        # get dataset and truncate outlier icu los
        try:
            self.x_train = pd.read_parquet(self.training, engine="fastparquet")
            _ = self.x_train.pop("index")
            self.y_train = self.x_train.pop('icu_los_hrs') / 24
            self.x_test = pd.read_parquet(self.validation, engine="fastparquet")
            _ = self.x_test.pop("index")
            self.y_test = self.x_test.pop('icu_los_hrs') / 24
        except Exception as e:
            print(e)
            print("Unable to read or prepare data. Please run data preparation routine first.")
        print("Regression Model Initialised. Train or load model.")

    def grid_search(self, overwrite=False):

        idx = self.x_train.sample(n=10000, random_state=2983).index
        x_set = self.x_train.loc[idx].reset_index(drop=True)
        y_set = self.y_train.loc[idx].reset_index(drop=True)

        for idx, est in enumerate(self.estimators):
            # Generate all combinations of parameters
            keys, values = zip(*self.param_grid.items())
            param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

            print(f"Round {idx + 1} with {len(param_combinations)} combinations")

            for params in param_combinations:
                final = est
                estimators = self.estimators

                kf = KFold(n_splits=5)

                param_grid = { }

                for key, val in params.items():
                    split = key.split("__")
                    if split[0] == final[0]:
                        param_grid[f'classifier__final_estimator__{split[1]}'] = val
                    param_grid[f'classifier__{split[0]}__{split[1]}'] = val

                Result = Query()
                query = self.db.search(
                        Result.params.test(
                    lambda params: compare_dicts(params.get("params", {}), param_grid) and params.get("final", "") == final[0]
                            )
                        )
                if len(query) > 0:
                    continue

                assert len(query) < 2, "there should only be 1 or 0 entries for each combination"

                mae, maet = [], []
                rmse, rmset = [], []
                rsquared, rsquaredt = [], []
                
                for train_index, test_index in kf.split(x_set):
                    x_train = x_set.loc[train_index, :].reset_index(drop=True)
                    x_test = x_set.loc[test_index, :].reset_index(drop=True)
                    y_train = y_set.loc[train_index].reset_index(drop=True)
                    y_test = y_set.loc[test_index].reset_index(drop=True)

                    stacking = StackingRegressor(
                    estimators=estimators, final_estimator=final[1],
                    n_jobs=12, verbose=1
                    )

                    # preprocess data
                    kernel = mf.ImputationKernel(
                            x_train,
                            mean_match_strategy={feat: "fast" for feat, cat in self.x_train.dtypes.items() if cat == "category"},
                            num_datasets=1, 
                            random_state=102)

                    preprocessor = Pipeline(
                        steps=[("imputer", kernel)]
                    )

                    clf = Pipeline(
                        steps=[("preprocessor", preprocessor), ("classifier", stacking)]
                    )
            

                    print(param_grid)
                    clf.set_params(**param_grid)

                    print("start training")
                    start = time.time()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clf.fit(x_train, y_train, preprocessor__imputer__iterations=2)
                    end = time.time()
                    print(f"finish training in {end - start:.4f} seconds")

                    rsquared.append(clf.score(x_test, y_test))
                    rsquaredt.append(clf.score(x_train, y_train))

                    predictions = clf.predict(x_test)
                    predictionst = clf.predict(x_train)

                    mae.append(np.mean(np.absolute(predictions - y_test)))
                    maet.append(np.mean(np.absolute(predictionst - y_train)))

                    rmse.append(np.mean(np.power(predictions - y_test, 2))**0.5)
                    rmset.append(np.mean(np.power(predictionst - y_train, 2))**0.5)

                row = {
                    "mae_test": np.mean(mae),
                    "mae_train": np.mean(maet),
                    "rmse_test": np.mean(rmse),
                    "rmse_train": np.mean(rmset),
                    "r2_test": np.mean(rsquared),
                    "r2_train": np.mean(rsquaredt),
                    "params": {"params":param_grid, "final": final[0]},
                    }
                print(row)
                self.db.insert(row)

        data = self.db.all()
        best = min(data, key = lambda x: x["rmse_test"])
        return best["rmse_test"], best["params"]

    def train_optimal_model(self, sample=False):
        # assert grid search has been done
        assert len(self.db) > 0, "grid search db empty"
        # import grid search results
        # get optimal model params from grid search
        best = min(self.db.all(), key = lambda x: x["rmse_test"])
        params = best["params"]["params"]
        final_name = best["params"]["final"]

        # create model and train
        stacking = StackingRegressor(
            estimators=self.estimators, final_estimator=[est for est in self.estimators if est[0] == final_name][0][1],
            n_jobs=12, verbose=1
        )

        kernel = mf.ImputationKernel(
                self.x_train,
                mean_match_strategy={feat: "fast" for feat, cat in self.x_train.dtypes.items() if cat == "category"},
                num_datasets=1, 
                random_state=102)

        preprocessor = Pipeline(
            steps=[("imputer", kernel)]
        )

        clf = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", stacking)]
        )
        clf.set_params(**params)

        if sample:
            sampled_idx = self.x_train.sample(n=len(self.x_train), replace=True).index
            clf.fit(self.x_train.loc[sampled_idx], self.y_train.loc[sampled_idx],
                preprocessor__imputer__iterations=2)
        else:
            clf.fit(self.x_train, self.y_train,
                preprocessor__imputer__iterations=2)

        # set fitted model to self.model
        self.model = clf

    def validate_optimal_model(self, seed=None):
        if self.model:
            if seed:
                idx = self.x_test.sample(n=len(self.x_test), replace=True, random_state=seed).index
                preds = self.model.predict(self.x_test.loc[idx].reset_index(drop=True))
                score = self.model.score(self.x_test.loc[idx].reset_index(drop=True), self.y_test.loc[idx].reset_index(drop=True))
                mae = np.mean(np.absolute(preds - self.y_test.loc[idx].reset_index(drop=True)))
                rmse = np.mean(np.power(preds - self.y_test.loc[idx].reset_index(drop=True), 2))**0.5
                return {
                    "score": score,
                    "mae": mae,
                    "rmse": rmse
                }
            else:
                preds = self.model.predict(self.x_test)
                score = self.model.score(self.x_test, self.y_test)
                mae = np.mean(np.absolute(preds - self.y_test))
                rmse = np.mean(np.power(preds - self.y_test, 2))**0.5
                return {
                    "score": score,
                    "mae": mae,
                    "rmse": rmse
                }
        else:
            print("Please train model prior to validating")

    def save_optimal_model(self):
        # save optimal model to OUT_DIR
        filename = f"{os.environ["OUT_DIR"]}{self.columns}_{self.diagnosis}_regression.pickle"
        if self.model:
            pickle.dump(self.model, open(filename, 'wb'))
        else:
            print("Please train model prior to saving")

    def load_optimal_model(self, path=None):
        # load the optimal model from OUT_DIR to self.model
        if path:
            filename = path
        else:
            filename = f"{os.environ["OUT_DIR"]}{self.columns}_{self.diagnosis}_regression.pickle"
        try:
            print('loading model')
            self.model = pickle.load(open(filename, 'rb'))
            print('model loaded')
        except:
            print('Could not load model, do you need to train first?')


