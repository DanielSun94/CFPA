from model.causal_trajectory_prediction import unit_test as model_unit_test
from default_config import args

test_script = 'train'
if test_script == 'model':
    model_unit_test(args)
else:
    raise ValueError('')
