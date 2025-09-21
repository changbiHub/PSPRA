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
parser.add_argument('--data_path', type=str, default=None,
                    help='Path to the dataset (if None, defaults will be used)')
parser.add_argument('--result_path', type=str, default=None,
                    help='Path to save results')
parser.add_argument('--model_save_path', type=str, default=None,
                    help='Path to save trained models')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite existing results')
parser.add_argument('--max_gap', type=int, default=21,
                    help='Maximum gap allowed in data preprocessing')
parser.add_argument('--deep_shap', action='store_true',
                    help='Use whole training and test sets for SHAP computation (slower but more comprehensive)')
args = parser.parse_args()
experiment = args.experiment
model_name = args.model_name
script_path = os.path.dirname(os.path.abspath(__file__))

# Set default paths relative to script directory if not provided
if args.result_path is None:
    args.result_path = os.path.join(script_path, "..", "results")
if args.model_save_path is None:
    args.model_save_path = os.path.join(script_path, "..", "models")
if args.data_path is None:
    args.data_path = os.path.join(script_path, "..", "data")

timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
identifier = np.random.randint(10000)
# %%
# Handle multivariate experiment (p3m)
if experiment == "p3m":
    th_val = 80
    toxin_names = ['C-1', 'C-2', 'GTX-1', 'GTX-2', 'GTX-3', 'GTX-4','GTX-5', 'NEOSTX', 'STX', 'dcGTX-2', 'dcGTX-3', 'dcSTX']
    multivariate_data_path = os.path.join(args.data_path, f"data_multivariate.csv")
    df = pd.read_csv(multivariate_data_path)[["site","date","compound","value"]]
    df_pivot = pd.pivot_table(df,index=["site","date"],columns="compound",values="value").reset_index().dropna()
    df_pivot = phrase(df_pivot)
    df_pivot = df_pivot[df_pivot.year>2012]
    data_toxins = df_pivot.loc[:,df_pivot.columns!="PSP-Total"]
    data_total = df_pivot[["site","date","year"]]
    data_total["PSP-Total-Cal"] = data_toxins[toxin_names].sum(axis=1)*100
    data_toxins["PSP_total"] = data_total["PSP-Total-Cal"]
    data_toxins["risk_flag"] = 0
    data_toxins.loc[data_toxins.PSP_total>th_val,"risk_flag"] = 1
    
    data_toxins_norm = data_toxins.copy()
    train_idx = data_toxins[data_toxins.year.isin([2015,2016,2017])].index
    test_idx = data_toxins[data_toxins.year.isin([2018,2019,2020])].index
    
    data_toxins_norm_train = data_toxins_norm.loc[train_idx]
    data_toxins_norm_test = data_toxins_norm.loc[test_idx]
    
    history_ls_train, target_ls_train, TI_train = preprocessor(data_toxins_norm_train, max_gap=args.max_gap, mode='multivariate')
    history_ls_test, target_ls_test, TI_test = preprocessor(data_toxins_norm_test, max_gap=args.max_gap, mode='multivariate')
    
    X_train = TI_train
    y_train = target_ls_train.risk_flag.values.astype(int)
    X_test = TI_test
    y_test = target_ls_test.risk_flag.values.astype(int)
    
else:
    # Handle univariate experiments (p1, p2, p3)
    univariate_data_path = os.path.join(args.data_path, f"data_univariate.csv")
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
        timeSeries_df["value"] = timeSeries_df.value.clip(40,None)
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

    y_smoothed_train, target_train, history_train = preprocessor(data_train, max_gap=args.max_gap, mode='univariate')
    y_smoothed_test, target_test, history_test = preprocessor(data_test, max_gap=args.max_gap, mode='univariate')

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
        ('catboost', CatBoostClassifier(verbose=0)),
        ('ada', AdaBoostClassifier()),
        ('lda', LinearDiscriminantAnalysis()),
        ('et', ExtraTreesClassifier()),
        ('lgb', LGBMClassifier(verbose=-1)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)),
        ('tcn', Pipeline([
            ('reshape', ReshapeTransformer(time_steps=time_steps, num_features=num_features)),
            ('model', NNmodel(time_steps=time_steps, num_features=num_features, model_name="TCN", 
                             best_model_filepath=f"temp_model/temp_tcn_{timestamp}_{identifier}.keras"))
        ])),
        ('rnn', Pipeline([
            ('reshape', ReshapeTransformer(time_steps=time_steps, num_features=num_features)),
            ('model', NNmodel(time_steps=time_steps, num_features=num_features, model_name="RNN", 
                             best_model_filepath=f"temp_model/temp_rnn_{timestamp}_{identifier}.keras"))
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
    
    # Add end-to-end SHAP analysis after model training
    import shap
    
    print("Computing end-to-end SHAP values for stacking ensemble...")
    
    # Extract feature order for metadata
    if experiment == "p3m":
        # Get the feature names from df_pivot (excluding non-toxin columns)
        feature_columns = [col for col in df_pivot.columns if col not in ['site', 'date', 'year', 'PSP_total', 'risk_flag']]
        feature_names = feature_columns  # These are the toxin names in order
    else:
        feature_names = ['PSP_value']  # For univariate experiments
    
    try:
        # Create a prediction function for SHAP
        def model_predict_proba(X):
            return model.predict_proba(X)[:, 1]  # Return probability of positive class
        
        # Use different data sizes based on deep_shap argument
        if args.deep_shap:
            print("Using deep SHAP mode - full datasets (this may take a while...)")
            background = X_train_flat
            explain_data = X_test_flat
            background_size = X_train_flat.shape[0]
            explain_size = X_test_flat.shape[0]
        else:
            print("Using fast SHAP mode - data subsets")
            # Use a subset of training data as background for faster computation
            background_size = min(100, X_train_flat.shape[0])
            background = X_train_flat[:background_size]
            
            # Use a subset of test data for explanation
            explain_size = min(50, X_test_flat.shape[0])
            explain_data = X_test_flat[:explain_size]
        
        print(f"Background size: {background_size}, Explanation size: {explain_size}")
        
        # Create SHAP explainer treating the whole model as black box
        explainer = shap.Explainer(model_predict_proba, background)
        shap_values = explainer(explain_data)
        
        print(f"SHAP values computed for {explain_size} test samples")
        
        # Restore original shape for p3m experiment
        if experiment == "p3m":
            # Reshape SHAP values back to original 3D shape
            shap_values_reshaped = shap_values.values.reshape(-1, time_steps, num_features)
            explain_data_reshaped = explain_data.reshape(-1, time_steps, num_features)
        else:
            shap_values_reshaped = shap_values.values
            explain_data_reshaped = explain_data
        
        # Save SHAP results with feature metadata
        shap_filename = f"shap_deep_{timestamp}_{identifier}.npz" if args.deep_shap else f"shap_fast_{timestamp}_{identifier}.npz"
        shap_save_path = os.path.join(args.result_path, experiment, model_name, shap_filename)
        os.makedirs(os.path.dirname(shap_save_path), exist_ok=True)
        
        np.savez(shap_save_path,
                shap_values=shap_values.values,  # Keep flattened for model compatibility
                shap_values_reshaped=shap_values_reshaped,  # Original shape
                base_values=shap_values.base_values,
                data=explain_data,  # Keep flattened
                data_reshaped=explain_data_reshaped,  # Original shape
                expected_value=explainer.expected_value,
                time_steps=time_steps,
                num_features=num_features,
                feature_names=feature_names,  # Feature order from df_pivot
                experiment=experiment,  # Experiment type for reference
                deep_shap=args.deep_shap,  # Flag indicating computation mode
                background_size=background_size,
                explain_size=explain_size)
        
        print(f"End-to-end SHAP values saved to {shap_save_path}")
        print(f"Feature order: {feature_names}")
        print(f"Deep SHAP mode: {args.deep_shap}")
        
    except Exception as e:
        print(f"Could not compute end-to-end SHAP values: {e}")
        
        # Fallback: Try with KernelExplainer for more robust black-box explanation
        try:
            print("Trying KernelExplainer as fallback...")
            
            # Use smaller datasets for fallback regardless of deep_shap setting
            fallback_background = X_train_flat[:20] if not args.deep_shap else X_train_flat[:100]
            fallback_explain = explain_data[:10] if not args.deep_shap else explain_data[:50]
            
            explainer = shap.KernelExplainer(model_predict_proba, fallback_background)
            shap_values = explainer.shap_values(fallback_explain)
            
            # Restore original shape for fallback too
            if experiment == "p3m":
                shap_values_reshaped = shap_values.reshape(-1, time_steps, num_features)
                explain_data_reshaped = fallback_explain.reshape(-1, time_steps, num_features)
            else:
                shap_values_reshaped = shap_values
                explain_data_reshaped = fallback_explain
            
            # Save fallback results with feature metadata
            fallback_filename = f"shap_kernel_deep_{timestamp}_{identifier}.npz" if args.deep_shap else f"shap_kernel_fast_{timestamp}_{identifier}.npz"
            fallback_save_path = os.path.join(args.result_path, experiment, model_name, fallback_filename)
            np.savez(fallback_save_path,
                    shap_values=shap_values,  # Keep flattened
                    shap_values_reshaped=shap_values_reshaped,  # Original shape
                    data=fallback_explain,  # Keep flattened
                    data_reshaped=explain_data_reshaped,  # Original shape
                    expected_value=explainer.expected_value,
                    time_steps=time_steps,
                    num_features=num_features,
                    feature_names=feature_names,  # Feature order from df_pivot
                    experiment=experiment,  # Experiment type for reference
                    deep_shap=args.deep_shap,  # Flag indicating computation mode
                    background_size=len(fallback_background),
                    explain_size=len(fallback_explain))
            
            print(f"Fallback SHAP values saved to {fallback_save_path}")
            print(f"Feature order: {feature_names}")
            print(f"Deep SHAP mode: {args.deep_shap}")
            
        except Exception as e2:
            print(f"Fallback SHAP also failed: {e2}")

    # Save model
    if args.model_save_path is not None:
        if not args.overwrite:
            save_path = os.path.join(args.model_save_path, experiment, model_name, f"model_{timestamp}_{identifier}.joblib")
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
            model_save_filepath = os.path.join(args.model_save_path, experiment, model_name, f"best_model_{timestamp}_{identifier}.keras")
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
            save_path = os.path.join(args.model_save_path, experiment, model_name,f"model_{timestamp}_{identifier}.joblib")
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
    save_path = os.path.join(args.result_path, experiment, model_name,f"results_{timestamp}_{identifier}.npz")
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
