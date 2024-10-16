"""

a proprietary package of convenience wrappers for sklearn


"""

    
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

from .convsklearn import convsklearn

barrier_trainer = convsklearn(
    target_name, numerical_features, categorical_features, 
    )
