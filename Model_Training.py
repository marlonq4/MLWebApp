from pycaret.datasets import get_data
from pycaret.regression import *
data = get_data('insurance')


s = setup(data, target='charges', session_id=123)

lr = create_model('lr')

s2 = setup(data, target='charges', session_id=123,
           normalize=True,
           polynomial_features=True, trigonometry_features=True, feature_interaction=True,
           bin_numeric_features=['age', 'bmi'])

lr = create_model('lr')

save_model(lr, model_name='C:/Users/marqu/MLWebApp/deployment_28042020')
deployment_28042020 = load_model('deployment_28042020')
deployment_28042020
