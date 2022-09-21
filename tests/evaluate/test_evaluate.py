import os
import pytest

from src.ansys.twin.evaluate.evaluate import TwinModel
from src.ansys.twin.evaluate.evaluate import TwinModelError
from tests.test_utilities import compare_dictionary


class TestEvaluate:

    def test_instantiation_with_valid_model_filepath(self):
        model_filepath = os.path.join('data', 'CoupleClutches_22R2_other.twin')
        TwinModel(model_filepath=model_filepath)

    def test_instantiation_with_invalid_model_filepath(self):
        with pytest.raises(TwinModelError) as e:
            TwinModel(model_filepath=None)
        assert 'Please provide valid filepath' in str(e)
        with pytest.raises(TwinModelError) as e:
            TwinModel(model_filepath='')
        assert 'Please provide existing filepath' in str(e)

    def test_parameters_property(self):
        model_filepath = os.path.join('data', 'CoupleClutches_22R2_other.twin')
        twin = TwinModel(model_filepath=model_filepath)
        # Test parameters have starting values JUST AFTER INSTANTIATION
        parameters_ref = {'CoupledClutches1_Inert1_J': 1.0,
                          'CoupledClutches1_Inert2_J': 1.0,
                          'CoupledClutches1_Inert3_J': 1.0,
                          'CoupledClutches1_Inert4_J': 1.0}
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Test parameters have been well updated AFTER FIRST EVALUATION INITIALIZATION
        new_parameters = {'CoupledClutches1_Inert1_J': 3.0,
                          'CoupledClutches1_Inert2_J': 2.0}
        twin.initialize_evaluation(parameters=new_parameters)
        parameters_ref = {'CoupledClutches1_Inert1_J': 3.0,
                          'CoupledClutches1_Inert2_J': 2.0,
                          'CoupledClutches1_Inert3_J': 1.0,
                          'CoupledClutches1_Inert4_J': 1.0}
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Test parameters keep same values AFTER STEP BY STEP EVALUATION
        twin.evaluate_step_by_step(step_size=0.001)
        parameters_ref = {'CoupledClutches1_Inert1_J': 3.0,
                          'CoupledClutches1_Inert2_J': 2.0,
                          'CoupledClutches1_Inert3_J': 1.0,
                          'CoupledClutches1_Inert4_J': 1.0}
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Test parameters have been updated to starting values AFTER NEW INITIALIZATION
        twin.initialize_evaluation()
        parameters_ref = {'CoupledClutches1_Inert1_J': 1.0,
                          'CoupledClutches1_Inert2_J': 1.0,
                          'CoupledClutches1_Inert3_J': 1.0,
                          'CoupledClutches1_Inert4_J': 1.0}
        assert compare_dictionary(twin.parameters, parameters_ref)

    def test_inputs_property_with_step_by_step_eval(self):
        model_filepath = os.path.join('data', 'CoupleClutches_22R2_other.twin')
        twin = TwinModel(model_filepath=model_filepath)
        # Test inputs have starting values JUST AFTER INSTANTIATION
        inputs_ref = {'Clutch1_in': 0.0,
                      'Clutch2_in': 0.0,
                      'Clutch3_in': 0.0,
                      'Torque_in': 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        # Test inputs have been well updated AFTER FIRST EVALUATION INITIALIZATION
        new_inputs = {'Clutch1_in': 1.0,
                      'Clutch2_in': 1.0,
                      'Clutch3_in': 1.0}
        twin.initialize_evaluation(inputs=new_inputs)
        inputs_ref = {'Clutch1_in': 1.0,
                      'Clutch2_in': 1.0,
                      'Clutch3_in': 1.0,
                      'Torque_in': 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        # Test inputs have been well updated AFTER STEP BY STEP EVALUATION
        new_inputs = {'Clutch1_in': 2.0,
                      'Clutch2_in': 2.0}
        twin.evaluate_step_by_step(step_size=0.001, inputs=new_inputs)
        inputs_ref = {'Clutch1_in': 2.0,
                      'Clutch2_in': 2.0,
                      'Clutch3_in': 1.0,
                      'Torque_in': 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        new_inputs = {'Clutch1_in': 3.0}
        twin.evaluate_step_by_step(step_size=0.001, inputs=new_inputs)
        inputs_ref = {'Clutch1_in': 3.0,
                      'Clutch2_in': 2.0,
                      'Clutch3_in': 1.0,
                      'Torque_in': 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        # Test inputs have been set to starting values AFTER NEW INITIALIZATION
        twin.initialize_evaluation()
        inputs_ref = {'Clutch1_in': 0.0,
                      'Clutch2_in': 0.0,
                      'Clutch3_in': 0.0,
                      'Torque_in': 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        # Test inputs have been well updated after step by step evaluation
        new_inputs = {'Clutch1_in': 2.0,
                      'Clutch2_in': 2.0}
        twin.evaluate_step_by_step(step_size=0.001, inputs=new_inputs)
        inputs_ref = {'Clutch1_in': 2.0,
                      'Clutch2_in': 2.0,
                      'Clutch3_in': 0.0,
                      'Torque_in': 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)

    def test_outputs_property_with_step_by_step_eval(self):
        model_filepath = os.path.join('data', 'CoupleClutches_22R2_other.twin')
        twin = TwinModel(model_filepath=model_filepath)
        # Test outputs have None values JUST AFTER INSTANTIATION
        outputs_ref = {'Clutch1_torque': None, 'Clutch2_torque': None, 'Clutch3_torque': None}
        assert compare_dictionary(twin.outputs, outputs_ref)
        # Test outputs have good values AFTER FIRST EVALUATION INITIALIZATION
        twin.initialize_evaluation()
        outputs_ref = {'Clutch1_torque': 0.0, 'Clutch2_torque': 0.0, 'Clutch3_torque': 0.0}
        assert compare_dictionary(twin.outputs, outputs_ref)
        # Test outputs have good values AFTER STEP BY STEP EVALUATION
        new_inputs = {'Clutch1_in': 1.0,
                      'Clutch2_in': 1.0}
        twin.evaluate_step_by_step(step_size=0.001, inputs=new_inputs)
        outputs_ref = {'Clutch1_torque': -10.0, 'Clutch2_torque': -5.0, 'Clutch3_torque': 0.0}
        assert compare_dictionary(twin.outputs, outputs_ref)
        # Test outputs have good values AFTER NEW EVALUATION INITIALIZATION
        twin.initialize_evaluation()
        outputs_ref = {'Clutch1_torque': 0.0, 'Clutch2_torque': 0.0, 'Clutch3_torque': 0.0}
        assert compare_dictionary(twin.outputs, outputs_ref)
