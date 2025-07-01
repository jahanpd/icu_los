import os
from typing import Mapping
import numpy as np
import pandas as pd
import miceforest as mf
from .diagnosis import EmbedDiagnosis
from enum import Enum

def get_mapping(mapping_str: str) -> dict[float | str, float]:
    maplist = mapping_str.split(";")
    mapdict: dict[float | str, float] = {}
    for m in maplist:
        vals = m.split("=")
        try:
            mapdict[float(vals[0])] = float(vals[1])
        except:
            mapdict[vals[0]] = float(vals[1])
    return mapdict

class ColumnSet(str, Enum):
    MUTUAL = "mutual"
    EXTENDED = "extended"

def prepare_anzics_data(
        source: str, 
        out_dir: str, 
        data_dict_csv: str, 
        diagnosis: bool = True,
        columns: ColumnSet = ColumnSet.EXTENDED
        ):
    """ This function takes the anzics data and prepares it for
    LOS analysis. It takes the location of the data, and saves the
    prepared training set and validation set in a directory.
    Function takes 4 args:
    source: location of the anzics data
    out_dir: directory to save the train and test sets
    data_dict: location of the data dictionary csv 
    diag_dict_csv: the location of the csv that maps AP3 diagnostic codes to their broad category (only used if using categorical diagnoses).
    """

    # import data and data_dict
    anzics = pd.read_csv(source)
    data_dict = pd.read_csv(data_dict_csv)

    # clean data

    # if feature and base_feature = length 1, and type numerical, then generate feature
    # if feature and base_feature = length 1, and type binary or cat, then map feature
    # otherwise it is a special case
    new_df = {}
    if columns == ColumnSet.EXTENDED:
        dd_subset_mapped = data_dict.dropna(subset=["features", "base_features", "mapping"])
    elif columns == ColumnSet.MUTUAL:
        dd_subset_mapped = data_dict.dropna(subset=["mutual_brazil_aus", "features", "base_features", "mapping"])

    for index, row in dd_subset_mapped.iterrows():
        mapdict = get_mapping(row["mapping"])
        new_df[row["features"]] = [
            None if x not in mapdict else mapdict[x] for x in anzics[row["base_features"]]
                    ]
        
    if columns == ColumnSet.EXTENDED:
        dd_subset_numerical = data_dict.dropna(subset=["features", "base_features"]).loc[data_dict["type"] == "numerical" ]
    elif columns == ColumnSet.MUTUAL:
        dd_subset_numerical = data_dict.dropna(subset=["mutual_brazil_aus", "features", "base_features"]).loc[data_dict["type"] == "numerical" ]

    for index, row in dd_subset_numerical.iterrows():
        if ";" not in row["base_features"]:
            new_df[row["features"]] = anzics[row["base_features"]]

    dd_subset_numerical = data_dict.dropna(subset=["features", "base_features"]).loc[data_dict["type"] == "datetime" ]
    for index, row in dd_subset_numerical.iterrows():
        new_df[row["features"]] = anzics[row["base_features"]]
 

    # special cases
    # BMI
    new_df["bmi"] = anzics["WEIGHT"] / ((anzics["HEIGHT"] / 100) ** 2)

    # ELECTIVE
    # the following method retains nans instead of coercing to 0
    map = get_mapping("1=1;2=0")
    elect = [None if x not in map else map[x] for x in anzics["ELECT"]]
    elect_surg = [None if x not in map else map[x] for x in anzics["ELECT_SURG"]]
    new_df["admission_type"] = [ 0 if (x is None and y == 0) or (y is None and x == 0) else x or y 
                                 for x, y in zip(elect, elect_surg)]

    # READMISSION
    map = lambda x: 1 if x > 1 else 0
    episode = [None if not x > 0 else map(x) for x in anzics["AdmEpisode"]]
    map = get_mapping("1=1;2=0")
    readmission = [None if x not in map else map[x] for x in anzics["READMITTED"]]
    new_df["readmission"] = [ 0 if (x is None and y == 0) or (y is None and x == 0) else x or y 
                                 for x, y in zip(episode, readmission)]
    # # MECH_VENT
    # map = get_mapping("1=1;2=0")
    # new_df["mech_vent"] = [None if x not in map else map[x] for x in anzics["INV_DAYONE"]]

    # LIVER
    map = get_mapping("1=1;2=0")
    cirrhosis = [None if x not in map else map[x] for x in anzics["CIRRHOS"]]
    chronic_liver = [None if x not in map else map[x] for x in anzics["CHR_LIV"]]
    new_df["apache_chronic_liver"] = [ 0 if (x is None and y == 0) or (y is None and x == 0) else x or y 
                                 for x, y in zip(cirrhosis, chronic_liver)]

    if diagnosis:
        # DIAGNOSIS
        embed = EmbedDiagnosis()
        latent_dims = 8
        codes = [c if pd.isnull(s) else s for c,s in zip(anzics["AP3DIAG"], anzics["AP3_SUBCODE"])]
        nulls = [pd.isnull(c) and pd.isnull(s) for c, s in zip(anzics["AP3DIAG"], anzics["AP3_SUBCODE"])]
        for dim in range(latent_dims):
            new_df[f"diag_latent_{dim}"] = [None if n else embed.return_small_embedding(c)[dim] for c, n in zip(codes, nulls)]

    # mapping_df = pd.read_csv(diag_dict_csv)
    # mapd = {}
    # for idx, row in mapping_df.iterrows():
    #     mapd[int(row.ap3diag)] = row.majordiag
    # codes = [c if pd.isnull(s) else int(s) for c,s in zip(anzics["AP3DIAG"], anzics["AP3_SUBCODE"])]
    # nulls = [pd.isnull(c) and pd.isnull(s) for c, s in zip(anzics["AP3DIAG"], anzics["AP3_SUBCODE"])]
    # codedf = pd.DataFrame({
    #   "ap3": [None if n else mapd[c] for c, n in zip(codes, nulls)]
    #   })
    # coded_dummies = pd.get_dummies(codedf, columns=["ap3"], drop_first=True)
    # for cat in list(coded_dummies):
    #     new_df[cat] = coded_dummies[cat]

    ndf = pd.DataFrame(new_df)
    ndf.dropna(subset=["icu_los_hrs"], inplace=True)

    # create data report and save to out_dir
    report = {
        "missing": [],
        "median": [],
        "iqr_low": [],
        "iqr_high": [],
        "fraction": [],
        "min": [],
        "max": [],
        "type": []
    }
    index = []

    if columns == ColumnSet.EXTENDED:
        subset = ["features"]
    elif columns == ColumnSet.MUTUAL:
        subset = ["features", "mutual_brazil_aus"]
    for i, row in data_dict.dropna(subset=subset).iterrows():
        feat = ndf[row["features"]]
        missing = feat.isnull().sum() / len(feat)
        if row["type"] == "numerical":
            # report median and IQR
            index.append(row["features"])
            report["missing"].append(missing)
            report["median"].append(feat.median())
            report["iqr_low"].append(feat.quantile(0.25))
            report["iqr_high"].append(feat.quantile(0.75))
            report["fraction"].append(None)
            report["min"].append(feat.min())
            report["max"].append(feat.max())
            report["type"].append("numerical")

        elif row["type"] == "binary":
            # report percentage of 1
            index.append(row["features"])
            report["missing"].append(missing)
            report["median"].append(None)
            report["iqr_low"].append(None)
            report["iqr_high"].append(None)
            counts = feat.value_counts(normalize=True)
            string = ",".join(["{} {:.2f}".format(idx, val) for idx, val in counts.items()])
            report["fraction"].append(string)
            report["min"].append(None)
            report["max"].append(None)
            report["type"].append("binary")

        elif row["type"] == "categorical":
            # report percentage of each group
            index.append(row["features"])
            report["missing"].append(missing)
            report["median"].append(None)
            report["iqr_low"].append(None)
            report["iqr_high"].append(None)
            counts = feat.value_counts(normalize=True)
            string = ",".join(["{} {:.2f}".format(idx, val) for idx, val in counts.items()])
            report["fraction"].append(string)
            report["min"].append(None)
            report["max"].append(None)
            report["type"].append("categorical")

        elif row["type"] == "datetime":
            # report percentage of each group
            index.append(row["features"])
            report["missing"].append(missing)
            report["median"].append(None)
            report["iqr_low"].append(None)
            report["iqr_high"].append(None)
            report["fraction"].append(None)
            report["min"].append(feat.min())
            report["max"].append(feat.max())
            report["type"].append("categorical")


    report = pd.DataFrame(report, index=index)
    report.to_csv(f"{out_dir}/{columns}_{diagnosis}_report.csv")

    # set cat features for the next section
    dd_subset = data_dict.dropna(subset=subset)
    
    variable_parameters = {}
    for feature, dtype in zip(dd_subset.features, dd_subset.type):
        if dtype == "categorical" or dtype == "binary":
            ndf[feature] = ndf[feature].astype("category")
            variable_parameters[feature] = {
                'num_class': ndf[feature].nunique(),
            }

    # split train and test set
    ndf.sort_values(by="icu_adm_datetime", inplace=True)
    val_size = int(len(ndf) * 0.1)
    
    train = ndf.iloc[:-val_size].reset_index()
    test = ndf.iloc[-val_size:].reset_index()

    assert (len(train) + len(test)) == len(ndf)

    print("saving pretruncated")
    train.to_parquet(f"{os.environ["OUT_DIR"]}/{columns}_{diagnosis}_train.parquet", engine="fastparquet")
    test.to_parquet(f"{os.environ["OUT_DIR"]}/{columns}_{diagnosis}_validation.parquet", engine="fastparquet")

    # perform imputation and truncate ICU LOS
    _ = train.pop("icu_adm_datetime")
    _ = test.pop("icu_adm_datetime")

    icu_los = train.icu_los_hrs

    Q1 = icu_los.quantile(0.25)
    Q3 = icu_los.quantile(0.75)
    outlier = Q3 + (3 * (Q3 - Q1))
    print(outlier)

    train.loc[train.icu_los_hrs > outlier, "icu_los_hrs"] = np.nan
    test.loc[test.icu_los_hrs > outlier, "icu_los_hrs"] = np.nan
    print(f"% > outlier {train.icu_los_hrs.isnull().sum() / len(train)}")
    
    print(train.dtypes)

    kernel_train = mf.ImputationKernel(
      train,
      variable_schema=["icu_los_hrs"],
      num_datasets=1,
      random_state=69
    )
    kernel_test = mf.ImputationKernel(
      test,
      variable_schema=["icu_los_hrs"],
      num_datasets=1,
      random_state=69
    )

    print("training imputation model train")
    kernel_train.mice(1)
    print("training imputation model test")
    kernel_test.mice(1)

    impute_train = kernel_train.impute_new_data(train)
    impute_train = impute_train.complete_data()
    impute_test = kernel_test.impute_new_data(test)
    impute_test = impute_test.complete_data()

    train.icu_los_hrs = impute_train.icu_los_hrs
    test.icu_los_hrs = impute_test.icu_los_hrs

    print("saving truncated data")
    train.to_parquet(f"{out_dir}/{columns}_{diagnosis}_train_trunc.parquet", engine="fastparquet")
    test.to_parquet(f"{out_dir}/{columns}_{diagnosis}_validation_trunc.parquet", engine="fastparquet")


# prepare_anzics_data(
#         source=os.environ["IN_DATA"], 
#         out_dir=os.environ["OUT_DIR"],
#         data_dict_csv=os.environ["DATA_DICT"],
#         diag_dict_csv=os.environ["DIAG_MAP"]
#         )
