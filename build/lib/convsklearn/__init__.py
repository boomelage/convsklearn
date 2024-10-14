from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,\
    MinMaxScaler, RobustScaler, Normalizer, PowerTransformer, \
        SplineTransformer, PolynomialFeatures, KernelCenterer, \
            QuantileTransformer, OrdinalEncoder,OneHotEncoder

print("a proprietary package of convenience wrappers for sklearn")       
"""
[
 'spot_price', 'strike_price', 'w', 'heston_price', 
 'risk_free_rate', 'dividend_rate', 'moneyness'
 'kappa', 'theta', 'rho', 'eta', 'v0', 'days_to_maturity',
 'expiration_date', 'calculation_date', 'moneyness_tag',
 ]
"""

target_name = 'observed_price'

numerical_features = [
    'spot_price', 'strike_price', 'days_to_maturity', 
    'risk_free_rate',
    'dividend_rate',
    'kappa', 'theta', 'rho', 'eta', 'v0',
    
    'barrier',
    
    ]

categorical_features = [
    
    'barrier_type_name',
    
    # 'outin',
    
    # 'updown',
    
    'w'
    
    ]

feature_set = numerical_features + categorical_features

transformers = [
    # ("QuantileTransformer",QuantileTransformer(),numerical_features),
    ("StandardScaler",StandardScaler(),numerical_features),
    # ("MinMaxScaler",MinMaxScaler(),numerical_features),
    # ("MaxAbsScaler",MaxAbsScaler(),numerical_features),
    # ("PowerTransformer",PowerTransformer(),numerical_features),
    # ("Normalizer",Normalizer(),numerical_features),
    
    # ("OrdinalEncoder", OrdinalEncoder(),categorical_features),
    ("OneHotEncoder", OneHotEncoder(
        sparse_output=False),categorical_features)
    
    ]

target_transformer_pipeline = Pipeline([
        ("StandardScaler", StandardScaler()),
        # ("RobustScaler", RobustScaler()),
        ])


from .convsklearn import convsklearn

barrier_trainer = convsklearn(
    target_name, numerical_features, categorical_features, 
    transformers, target_transformer_pipeline
    )

