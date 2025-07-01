from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
import miceforest as mf
import xgboost as xgb
import pandas as pd
import numpy as np
import argparse
import pickle
import os

parser = argparse.ArgumentParser(
                    prog='Run or Train ICU Model',
                    description='',
                    epilog='')

parser.add_argument('-t', '--train',
                    action='store_true')  # on/off flag
args = parser.parse_args()

filename = f"{os.environ["OUT_DIR"]}model.pickle"
training = f"{os.environ["OUT_DIR"]}train_trunc.parquet"
validation = f"{os.environ["OUT_DIR"]}validation_trunc.parquet"

if not args.train:
    try:
        print('loading model')
        clf = pickle.load(open(filename, 'rb'))
        print('model loaded')
    except:
        print('Could not load model, do you need to train first?')
        quit()

# get dataset and truncate outlier icu los
x_train = pd.read_parquet(training, engine="fastparquet")
_ = x_train.pop("index")
y_train = x_train.pop('icu_los_hrs') / 24

if args.train:
    estimators = [
    ('lr', LinearRegression()),
    ('xgb', xgb.XGBRegressor(
        random_state=42,
        enable_categorical=True
        ))
    ]

    stacking = StackingRegressor(
    estimators=estimators, final_estimator=RandomForestRegressor(),
    n_jobs=12, verbose=1
    )

    # preprocess data
    kernel = mf.ImputationKernel(
            x_train,
            mean_match_strategy={feat: "fast" for feat, cat in x_train.dtypes.items() if cat == "category"},
            num_datasets=1, 
            random_state=102)

    preprocessor = Pipeline(
        steps=[("imputer", kernel)]
    )

    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", stacking)]
    )

    print('fitting model to training data')
    clf.fit(
            x_train, 
            y_train, 
            preprocessor__imputer__iterations=2
            )
    
    print('saving model')

    pickle.dump(clf, open(filename, 'wb'))



x_test = pd.read_parquet(validation, engine="fastparquet")
y_test = x_test.pop('icu_los_hrs') / 24
_ = x_test.pop("index")

rsquared = clf.score(x_test, y_test)
predictions = clf.predict(x_test)
rsquaredt = clf.score(x_train, y_train)
predictionst = clf.predict(x_train)

mae = np.mean(np.absolute(predictions - y_test))
rmse = np.mean(np.power(predictions - y_test, 2))**0.5
maet = np.mean(np.absolute(predictionst - y_train))
rmset = np.mean(np.power(predictionst - y_train, 2))**0.5

print(rsquared, mae, rmse)
print(rsquaredt, maet, rmset)


