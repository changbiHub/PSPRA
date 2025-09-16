import pandas as pd
import numpy as np
import sklearn.metrics
from preprocessor import *
import os
import argparse
import joblib

parser = argparse.ArgumentParser(description="Run experiment with specified model.")
parser.add_argument('--experiment', type=str, default="p1",
                    help='Experiment identifier (e.g., "p1", "p2", "p3", "p3m")')
parser.add_argument('--model_name', type=str, default="TCN",
                    help='Model name to use (e.g., "TCN", "RNN")')
parser.add_argument('--result_path', type=str, default="../results",
                    help='Path to save results')
parser.add_argument('--model_save_path', type=str, default="../models",
                    help='Path to save trained models')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite existing results')
args = parser.parse_args()
experiment = args.experiment
model_name = args.model_name
script_path = os.path.dirname(os.path.abspath(__file__))
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

# %%
# Handle multivariate experiment (p3m)
if experiment == "p3m":
    th_val = 80
    toxin_names = ['C-1', 'C-2', 'GTX-1', 'GTX-2', 'GTX-3', 'GTX-4','GTX-5', 'NEOSTX', 'STX', 'dcGTX-2', 'dcGTX-3', 'dcSTX']
    multivariate_data_path = os.path.join(script_path, os.pardir, "data", f"PSP_BC_multivariate.csv")
    df = pd.read_csv(multivariate_data_path)[["site","date","compound","value"]]
    df_pivot = pd.pivot_table(df,index=["site","date"],columns="compound",values="value").reset_index().dropna()
    df_pivot = phrase(df_pivot)
    df_pivot = df_pivot[df_pivot.year>2012]
    data_toxins = df_pivot.loc[:,df_pivot.columns!="PSP-Total"]
    data_total = df_pivot[["site","date","PSP-Total","year"]]
    data_total["PSP-Total-Cal"] = data_toxins[toxin_names].sum(axis=1)*100
    data_toxins["PSP_total"] = data_total["PSP-Total-Cal"]
    data_toxins["risk_flag"] = 0
    data_toxins.loc[data_toxins.PSP_total>th_val,"risk_flag"] = 1
    
    data_toxins_norm = data_toxins.copy()
    train_idx = data_toxins[data_toxins.year.isin([2015,2016,2017])].index
    test_idx = data_toxins[data_toxins.year.isin([2018,2019,2020])].index
    
    data_toxins_norm_train = data_toxins_norm.loc[train_idx]
    data_toxins_norm_test = data_toxins_norm.loc[test_idx]
    
    history_ls_train, target_ls_train, TI_train = preprocessor(data_toxins_norm_train, mode='multivariate')
    history_ls_test, target_ls_test, TI_test = preprocessor(data_toxins_norm_test, mode='multivariate')
    
    X_train = TI_train
    y_train = target_ls_train.risk_flag.values.astype(int)
    X_test = TI_test
    y_test = target_ls_test.risk_flag.values.astype(int)
    
else:
    # Handle univariate experiments (p1, p2, p3)
    univariate_data_path = os.path.join(script_path, os.pardir, "data", f"PSP_BC_univariate.csv")
    df = pd.read_csv(univariate_data_path)
    timeSeries_df = phrase(df)

    all_year = None
    train_year = None
    test_year = None
    risk_th = None

    if experiment=="p1":
        all_year = set([2013,2014,2015,2016,2017,2018,2019,2020])
        test_year = set([2017,2018,2019,2020])
        train_year = all_year-test_year
        risk_th = 80
        
    if experiment=="p2":
        all_year = set(np.unique(timeSeries_df.year))
        train_year = set(range(2001,2011))
        test_year = set(range(2013,2021))
        timeSeries_df["value_valeur"] = timeSeries_df.value_valeur.clip(40,None)
        risk_th = 80

    if experiment=="p3":
        all_year = set([2015,2016,2017,2018,2019,2020])
        test_year = set([2018,2019,2020])
        train_year = all_year-test_year
        risk_th = 80

    timeSeries_df = timeSeries_df[timeSeries_df.year.isin(all_year)]
    timeSeries_df["risk_flag"] = 0
    timeSeries_df.loc[timeSeries_df["value"]>risk_th,"risk_flag"] = 1

    data_train = timeSeries_df[timeSeries_df.year.isin(train_year)]
    data_test = timeSeries_df[timeSeries_df.year.isin(test_year)]

    y_smoothed_train, target_train, history_train = preprocessor(data_train, mode='univariate')
    y_smoothed_test, target_test, history_test = preprocessor(data_test, mode='univariate')

    X_train = y_smoothed_train
    y_train = target_train.risk_flag.values.astype(int)
    X_test = y_smoothed_test
    y_test = target_test.risk_flag.values.astype(int)

# %%
y_pred_proba = None
y_pred = None
y_pred_proba_train = None
y_pred_train = None

if model_name == "stacking_ensemble":
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from nnModel import NNmodel, ReshapeTransformer
    from sklearn.pipeline import Pipeline
    
    # Prepare data - flatten for traditional ML models
    if experiment == "p3m":
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        num_features = X_train.shape[2]
        time_steps = X_train.shape[1]
    else:
        X_train_flat = X_train
        X_test_flat = X_test
        num_features = 1
        time_steps = X_train.shape[1]
    
    # Define all base estimators including neural networks in pipelines
    base_estimators = [
        ('gbc', GradientBoostingClassifier()),
        ('rf', RandomForestClassifier()),
        ('lr', LogisticRegression(max_iter=1000)),
        ('dt', DecisionTreeClassifier()),
        ('catboost', CatBoostClassifier(verbose=0)),
        ('ada', AdaBoostClassifier()),
        ('lda', LinearDiscriminantAnalysis()),
        ('et', ExtraTreesClassifier()),
        ('lgb', LGBMClassifier(verbose=-1)),
        ('qda', QuadraticDiscriminantAnalysis()),
        ('nb', GaussianNB()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)),
        ('knn', KNeighborsClassifier()),
        ('tcn', Pipeline([
            ('reshape', ReshapeTransformer(time_steps=time_steps, num_features=num_features)),
            ('model', NNmodel(time_steps=time_steps, num_features=num_features, model_name="TCN", 
                             best_model_filepath=f"temp_tcn_{timestamp}.keras"))
        ])),
        ('rnn', Pipeline([
            ('reshape', ReshapeTransformer(time_steps=time_steps, num_features=num_features)),
            ('model', NNmodel(time_steps=time_steps, num_features=num_features, model_name="RNN", 
                             best_model_filepath=f"temp_rnn_{timestamp}.keras"))
        ]))
    ]
    
    # Create stacking classifier
    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, penalty='elasticnet', solver='saga', l1_ratio=0.5),
        cv=5
    )
    
    print("Training stacking ensemble with all models...")
    # Train the stacking ensemble - use flattened data
    model.fit(X_train_flat, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test_flat)[:, 1]
    y_pred = model.predict(X_test_flat)
    y_pred_proba_train = model.predict_proba(X_train_flat)[:, 1]
    y_pred_train = model.predict(X_train_flat)
    
    # Save model
    if args.model_save_path is not None:
        if not args.overwrite:
            save_path = os.path.join(args.model_save_path, experiment, model_name, f"model_{timestamp}.joblib")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(model, save_path)
            print(f"Stacking ensemble saved to {save_path}")
        else:
            save_path = os.path.join(args.model_save_path, experiment, model_name, f"model.joblib")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(model, save_path)
            print(f"Stacking ensemble saved to {save_path}")

elif model_name in ["RNN","TCN"]:
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    from nnModel import NNmodel

    if args.model_save_path is not None:
        if not args.overwrite:
            os.makedirs(os.path.join(args.model_save_path, experiment, model_name), exist_ok=True)
            model_save_filepath = os.path.join(args.model_save_path, experiment, model_name, f"best_model_{timestamp}.keras")
        else:
            os.makedirs(args.model_save_path, exist_ok=True)
            model_save_filepath = os.path.join(args.model_save_path, experiment, model_name, f"best_model_{experiment}_{model_name}.keras")
    else:
        model_save_filepath = f"best_model_{experiment}_{model_name}.keras"

    if experiment == "p3m":
        model = NNmodel(time_steps=X_train.shape[1], num_features=X_train.shape[2], model_name=model_name, best_model_filepath=model_save_filepath)
        model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_train = model.predict(X_train)
    else:
        model = NNmodel(time_steps=X_train.shape[1], model_name=model_name, best_model_filepath=model_save_filepath)
        model.fit(X_train.reshape(-1,X_train.shape[1],1), y_train, epochs=200, batch_size=32, validation_split=0.1)
        y_pred_proba = model.predict_proba(X_test.reshape(-1,X_test.shape[1],1))[:, 1]
        y_pred = model.predict(X_test.reshape(-1,X_test.shape[1],1))
        y_pred_proba_train = model.predict_proba(X_train.reshape(-1,X_train.shape[1],1))[:, 1]
        y_pred_train = model.predict(X_train.reshape(-1,X_train.shape[1],1))

elif model_name in ["GBC","RF","LR","DT","LR","catboost","ADA","LDA","ET","LGB","QDA","NB","XGB","KNN"]:
    
    # For multivariate experiment, flatten the input
    if experiment == "p3m":
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train_flat = X_train
        X_test_flat = X_test

    if model_name in ["GBC"]:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()
    elif model_name in ["RF"]:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    elif model_name in ["LR"]:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000)
    elif model_name in ["DT"]:
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
    elif model_name in ["catboost"]:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(verbose=0)
    elif model_name in ["ADA"]:
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier()
    elif model_name in ["LDA"]:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis()
    elif model_name in ["ET"]:
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier()
    elif model_name in ["LGB"]:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier()
    elif model_name in ["QDA"]:
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        model = QuadraticDiscriminantAnalysis()
    elif model_name in ["NB"]:
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif model_name in ["XGB"]:
        from xgboost import XGBClassifier
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif model_name in ["KNN"]:
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    
    model.fit(X_train_flat, y_train)
    y_pred_proba = model.predict_proba(X_test_flat)[:, 1]
    y_pred = model.predict(X_test_flat)
    y_pred_proba_train = model.predict_proba(X_train_flat)[:, 1]
    y_pred_train = model.predict(X_train_flat)
    
    if args.model_save_path is not None:
        if not args.overwrite:
            save_path = os.path.join(args.model_save_path, experiment, model_name,f"model_{timestamp}.joblib")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(model, save_path)
            print(f"Model saved to {save_path}")
        else:
            save_path = os.path.join(args.model_save_path, experiment, model_name,f"model.joblib")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(model, save_path)
            print(f"Model saved to {save_path}")
else:
    raise ValueError(f"Model {model_name} not recognized.")

auc = sklearn.metrics.roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc}")
# Save results
if not args.overwrite:
    save_path = os.path.join(args.result_path, experiment, model_name,f"results_{timestamp}.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path,
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, y_pred_proba=y_pred_proba, y_pred=y_pred,y_pred_proba_train=y_pred_proba_train,y_pred_train=y_pred_train)
    print(f"Results saved to {save_path}")
else:
    save_path = os.path.join(args.result_path, experiment, model_name,f"results.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path,
            X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test, y_pred_proba=y_pred_proba, y_pred =y_pred,y_pred_proba_train=y_pred_proba_train,y_pred_train=y_pred_train)
    print(f"Results overwritten at {save_path}")
