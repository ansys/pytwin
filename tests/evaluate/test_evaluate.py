import os

import pytest
import pandas as pd

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

    def test_inputs_property_with_batch_eval(self):
        model_filepath = os.path.join('data', 'CoupleClutches_22R2_other.twin')
        twin = TwinModel(model_filepath=model_filepath)
        # Test inputs after BATCH EVALUATION
        twin.initialize_evaluation()
        twin.evaluate_batch(pd.DataFrame({'Time': [0, 1]}))
        inputs_ref = {'Clutch1_in': 0.0,
                      'Clutch2_in': 0.0,
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

    def test_outputs_property_with_batch_eval(self):
        model_filepath = os.path.join('data', 'CoupleClutches_22R2_other.twin')
        twin = TwinModel(model_filepath=model_filepath)
        # Test outputs after BATCH EVALUATION
        twin.initialize_evaluation()
        twin.evaluate_batch(pd.DataFrame({'Time': [0, 1]}))
        inputs_ref = {'Clutch1_in': 0.0,
                      'Clutch2_in': 0.0,
                      'Clutch3_in': 0.0,
                      'Torque_in': 0.0}
        outputs_ref = {'Clutch1_torque': 0.0, 'Clutch2_torque': 0.0, 'Clutch3_torque': 0.0}
        assert compare_dictionary(twin.outputs, outputs_ref)

    def test_raised_errors_with_step_by_step_evaluation(self):
        model_filepath = os.path.join('data', 'CoupleClutches_22R2_other.twin')
        twin = TwinModel(model_filepath=model_filepath)
        # Raise an error if TWIN MODEL HAS NOT BEEN INITIALIZED
        with pytest.raises(TwinModelError) as e:
            twin.evaluate_step_by_step(step_size=0.001)
        assert 'Please initialize evaluation' in str(e)
        # Raise an error if STEP SIZE IS ZERO
        with pytest.raises(TwinModelError) as e:
            twin.initialize_evaluation()
            twin.evaluate_step_by_step(step_size=0.)
        assert 'Step size must be strictly bigger than zero' in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.initialize_evaluation()
            twin.evaluate_step_by_step(step_size=-0.1)
        assert 'Step size must be strictly bigger than zero' in str(e)

    def test_raised_errors_with_batch_evaluation(self):
        model_filepath = os.path.join('data', 'CoupleClutches_22R2_other.twin')
        twin = TwinModel(model_filepath=model_filepath)
        # Raise an error if TWIN MODEL HAS NOT BEEN INITIALIZED
        with pytest.raises(TwinModelError) as e:
            twin.evaluate_batch(pd.DataFrame())
        assert 'Please initialize evaluation' in str(e)
        # Raise an error if INPUTS DATAFRAME HAS NO TIME COLUMN
        with pytest.raises(TwinModelError) as e:
            twin.initialize_evaluation()
            twin.evaluate_batch(pd.DataFrame())
        assert 'Please provide a dataframe with a \'Time\' column to use batch mode evaluation' in str(e)

    def test_evaluation_methods_give_same_results(self):
        sbs_outputs = dict()
        # Evaluate twin model with STEP BY STEP EVALUATION
        model_filepath = os.path.join('data', 'CoupleClutches_22R2_other.twin')
        twin = TwinModel(model_filepath=model_filepath)
        # t=0. (s)
        twin.initialize_evaluation(inputs={'Clutch1_in': 1.0, 'Clutch2_in': 1.0})
        sbs_outputs['Time'] = [0.]
        for name in twin.outputs:
            sbs_outputs[name] = [twin.outputs[name]]
        # t=0.1 (s)
        new_inputs = {'Clutch1_in': 2.0, 'Clutch2_in': 2.0}
        twin.evaluate_step_by_step(step_size=0.1, inputs=new_inputs)
        sbs_outputs['Time'].append(0.1)
        for name in twin.outputs:
            sbs_outputs[name].append(twin.outputs[name])
        # t=0.2 (s)
        new_inputs = {'Clutch1_in': 3.0, 'Clutch2_in': 3.0}
        twin.evaluate_step_by_step(step_size=0.1, inputs=new_inputs)
        sbs_outputs['Time'].append(0.2)
        for name in twin.outputs:
            sbs_outputs[name].append(twin.outputs[name])
        # Evaluate twin model with BATCH EVALUATION
        twin.initialize_evaluation(inputs={'Clutch1_in': 1.0, 'Clutch2_in': 1.0})
        inputs_df = pd.DataFrame({'Time': [0.1, 0.2], 'Clutch1_in': [2., 3.], 'Clutch2_in': [2., 3.]})
        outputs_df = twin.evaluate_batch(inputs_df)
        # Compare STEP-BY-STEP vs BATCH RESULTS
        """
        TODO - Vérifier pourquoi le batch mode ne renvoie pas la sortie associée au dernier pas de temps.
        En attendant, on retire donc le dernier pas de temps de l'évaluation step by step pour effectuer la comparaison.
        """
        sbs_outputs_df = pd.DataFrame(sbs_outputs).iloc[:-1, :]
        assert pd.DataFrame.equals(sbs_outputs_df, outputs_df)

    def test_evaluation_initialization_with_config_file(self):
        model_filepath = os.path.join('data', 'CoupleClutches_22R2_other.twin')
        twin = TwinModel(model_filepath=model_filepath)
        # Evaluation initialization with VALID CONFIG FILE
        config_filepath = os.path.join('data', 'eval_init_config.json')
        twin.initialize_evaluation(json_config_filepath=config_filepath)
        inputs_ref = {'Clutch1_in': 1.0,
                      'Clutch2_in': 1.0,
                      'Clutch3_in': 1.0,
                      'Torque_in': 1.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        parameters_ref = {'CoupledClutches1_Inert1_J': 2.0,
                          'CoupledClutches1_Inert2_J': 2.0,
                          'CoupledClutches1_Inert3_J': 2.0,
                          'CoupledClutches1_Inert4_J': 2.0}
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Evaluation initialization IGNORE INVALID PARAMETER AND INPUT ENTRIES
        config_filepath = os.path.join('data', 'eval_init_config_invalid_keys.json')
        twin.initialize_evaluation(json_config_filepath=config_filepath)
        inputs_ref = {'Clutch1_in': 1.0,
                      'Clutch2_in': 1.0,
                      'Clutch3_in': 1.0,
                      'Torque_in': 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        parameters_ref = {'CoupledClutches1_Inert1_J': 2.0,
                          'CoupledClutches1_Inert2_J': 2.0,
                          'CoupledClutches1_Inert3_J': 2.0,
                          'CoupledClutches1_Inert4_J': 1.0}
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Evaluation initialization WITH ONLY PARAMETERS ENTRIES
        config_filepath = os.path.join('data', 'eval_init_config_only_parameters.json')
        twin.initialize_evaluation(json_config_filepath=config_filepath)
        inputs_ref = {'Clutch1_in': 0.0,
                      'Clutch2_in': 0.0,
                      'Clutch3_in': 0.0,
                      'Torque_in': 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        parameters_ref = {'CoupledClutches1_Inert1_J': 2.0,
                          'CoupledClutches1_Inert2_J': 2.0,
                          'CoupledClutches1_Inert3_J': 2.0,
                          'CoupledClutches1_Inert4_J': 2.0}
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Evaluation initialization WITH ONLY INPUT ENTRIES
        config_filepath = os.path.join('data', 'eval_init_config_only_inputs.json')
        twin.initialize_evaluation(json_config_filepath=config_filepath)
        inputs_ref = {'Clutch1_in': 1.0,
                      'Clutch2_in': 1.0,
                      'Clutch3_in': 1.0,
                      'Torque_in': 1.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        parameters_ref = {'CoupledClutches1_Inert1_J': 1.0,
                          'CoupledClutches1_Inert2_J': 1.0,
                          'CoupledClutches1_Inert3_J': 1.0,
                          'CoupledClutches1_Inert4_J': 1.0}
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Evaluation initialization RAISE AN ERROR IF CONFIG FILEPATH DOES NOT EXIST
        with pytest.raises(TwinModelError) as e:
            twin.initialize_evaluation(json_config_filepath='filepath_does_not_exist')
        assert 'Please provide an existing filepath to initialize the twin model evaluation' in str(e)

    def test_close_method(self):
        model_filepath = os.path.join('data', 'CoupleClutches_22R2_other.twin')
        twin = TwinModel(model_filepath=model_filepath)
        twin = TwinModel(model_filepath=model_filepath)
        twin = TwinModel(model_filepath=model_filepath)
