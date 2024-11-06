"""

a proprietary package of convenience wrappers for sklearn


"""
from .convsklearn import convsklearn

convsklearn = convsklearn

barrier_features = {
    'numerical_features':[
        'spot_price', 'strike_price', 'days_to_maturity',
        'rebate', 'barrier',
        'risk_free_rate','dividend_rate', 
        'theta', 'kappa', 'rho', 'eta', 'v0'
    ],
    'categorical_features':['w','barrier_type_name'],
    'target_name':'observed_price'
}


asian_features = {
    'numerical_features':[
        'spot_price','strike_price','days_to_maturity'
        'fixing_frequency','n_fixings','past_fixings',
        'dividend_rate','risk_free_rate',
        'theta', 'kappa', 'rho', 'eta', 'v0'
    ],
    'categorical_features':['w','averaging_type'],
    'target_name':'observed_price'
}