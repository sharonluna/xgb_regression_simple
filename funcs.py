# Extra functions
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error


# scoring ############################################################################################################
def score_rmse_model(model: xgb.core.Booster, dmat: xgb.core.DMatrix) -> float:
    """
    Function to score RMSE in a trained xgb model.

    Args:
        model (xgb.core.Booster): A trained xgb model.
        dmat (xgb.core.DMatrix): An  xgb data matrix.

    Returns:
        float: RMSE score
    """
    y_true = dmat.get_label() 
    y_pred = model.predict(dmat) 
    return root_mean_squared_error(y_true, y_pred)

def score_r2_model(model: xgb.core.Booster, dmat: xgb.core.DMatrix) -> float:
    """Function to score r2 in a trained xgb model

    Args:
        model (xgb.core.Booster): A trained xgb model
        dmat (xgb.core.DMatrix): An xgb data matrix

    Returns:
        float: r2 score
    """
    y_true = dmat.get_label() 
    y_pred = model.predict(dmat) 
    return r2_score(y_true, y_pred)

def score_mae_model(model: xgb.core.Booster, dmat: xgb.core.DMatrix) -> float:
    """
    CFunction to score MAE in a trained xgb model

    Args:
        model (xgb.core.Booster): A trained xgb model
        dmat (xgb.core.DMatrix): An xgb data matrix

    Returns:
        float: MAE score
    """
    y_true = dmat.get_label() 
    y_pred = model.predict(dmat) 
    return mean_absolute_error(y_true, y_pred)


# Feature importance ###########################################################################################
def enhanced_shap_analysis(model: xgb.core.Booster, X: xgb.core.DMatrix, shap_values: np.ndarray) -> dict:
    """
    Function to enhance shap analysis with additional insights

    Args:
        model (xgb.core.Booster): A trianed xgb model
        X (xgb.core.DMatrix): A xgb data matrix
        shap_values (np.ndarray): shape values

    Returns:
        dict: FEature importance stats, feature interaction values and high impact ranges
    """

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': np.abs(shap_values).mean(0),
        'std_shap': np.std(shap_values, axis=0),
        'max_impact': np.max(np.abs(shap_values), axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    interaction_values = shap.TreeExplainer(model).shap_interaction_values(X)
    
    high_impact_ranges = {}
    for feature in X.columns:
        feature_idx = list(X.columns).index(feature)
        values = X[feature].values
        impacts = shap_values[:, feature_idx]
        
        threshold = np.percentile(np.abs(impacts), 90)
        high_impact_mask = np.abs(impacts) > threshold
        high_impact_ranges[feature] = {
            'value_range': (np.min(values[high_impact_mask]), 
                          np.max(values[high_impact_mask])),
            'impact_range': (np.min(impacts[high_impact_mask]), 
                           np.max(impacts[high_impact_mask]))
        }
    
    return {
        'importance_stats': feature_importance,
        'interaction_values': interaction_values,
        'high_impact_ranges': high_impact_ranges
    }