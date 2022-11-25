import os
import time

import pandas as pd
import pytest

from pytwin import TwinModel, TwinModelError, examples
from pytwin.settings import (
    get_pytwin_log_file,
    get_pytwin_logger,
    get_pytwin_working_dir,
    modify_pytwin_working_dir,
    reinit_settings_for_unit_tests,
)
from tests.utilities import compare_dictionary

COUPLE_CLUTCHES_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "CoupleClutches_22R2_other.twin")
DYNAROM_HX_23R1 = os.path.join(os.path.dirname(__file__), "data", "HX_scalarDRB_23R1_other.twin")
RC_HEAT_CIRCUIT_23R1 = os.path.join(os.path.dirname(__file__), "data", "RC_heat_circuit_23R1.twin")

UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")


def reinit_settings():
    import shutil

    from pytwin.settings import reinit_settings_for_unit_tests

    reinit_settings_for_unit_tests()
    if os.path.exists(UNIT_TEST_WD):
        shutil.rmtree(UNIT_TEST_WD)
    return UNIT_TEST_WD


class TestTwinModel:
    def test_instantiation_with_valid_model_filepath(self):
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        TwinModel(model_filepath=model_filepath)

    def test_instantiation_with_invalid_model_filepath(self):
        with pytest.raises(TwinModelError) as e:
            TwinModel(model_filepath=None)
        assert "Please provide valid filepath" in str(e)
        with pytest.raises(TwinModelError) as e:
            TwinModel(model_filepath="")
        assert "Please provide existing filepath" in str(e)

    def test_parameters_property(self):
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # Test parameters have starting values JUST AFTER INSTANTIATION
        parameters_ref = {
            "CoupledClutches1_Inert1_J": 1.0,
            "CoupledClutches1_Inert2_J": 1.0,
            "CoupledClutches1_Inert3_J": 1.0,
            "CoupledClutches1_Inert4_J": 1.0,
        }
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Test parameters have been well updated AFTER FIRST EVALUATION INITIALIZATION
        new_parameters = {"CoupledClutches1_Inert1_J": 3.0, "CoupledClutches1_Inert2_J": 2.0}
        twin.initialize_evaluation(parameters=new_parameters)
        parameters_ref = {
            "CoupledClutches1_Inert1_J": 3.0,
            "CoupledClutches1_Inert2_J": 2.0,
            "CoupledClutches1_Inert3_J": 1.0,
            "CoupledClutches1_Inert4_J": 1.0,
        }
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Test parameters keep same values AFTER STEP BY STEP EVALUATION
        twin.evaluate_step_by_step(step_size=0.001)
        parameters_ref = {
            "CoupledClutches1_Inert1_J": 3.0,
            "CoupledClutches1_Inert2_J": 2.0,
            "CoupledClutches1_Inert3_J": 1.0,
            "CoupledClutches1_Inert4_J": 1.0,
        }
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Test parameters have been updated to starting values AFTER NEW INITIALIZATION
        twin.initialize_evaluation()
        parameters_ref = {
            "CoupledClutches1_Inert1_J": 1.0,
            "CoupledClutches1_Inert2_J": 1.0,
            "CoupledClutches1_Inert3_J": 1.0,
            "CoupledClutches1_Inert4_J": 1.0,
        }
        assert compare_dictionary(twin.parameters, parameters_ref)

    def test_inputs_property_with_step_by_step_eval(self):
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # Test inputs have starting values JUST AFTER INSTANTIATION
        inputs_ref = {"Clutch1_in": 0.0, "Clutch2_in": 0.0, "Clutch3_in": 0.0, "Torque_in": 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        # Test inputs have been well updated AFTER FIRST EVALUATION INITIALIZATION
        new_inputs = {"Clutch1_in": 1.0, "Clutch2_in": 1.0, "Clutch3_in": 1.0}
        twin.initialize_evaluation(inputs=new_inputs)
        inputs_ref = {"Clutch1_in": 1.0, "Clutch2_in": 1.0, "Clutch3_in": 1.0, "Torque_in": 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        # Test inputs have been well updated AFTER STEP BY STEP EVALUATION
        new_inputs = {"Clutch1_in": 2.0, "Clutch2_in": 2.0}
        twin.evaluate_step_by_step(step_size=0.001, inputs=new_inputs)
        inputs_ref = {"Clutch1_in": 2.0, "Clutch2_in": 2.0, "Clutch3_in": 1.0, "Torque_in": 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        new_inputs = {"Clutch1_in": 3.0}
        twin.evaluate_step_by_step(step_size=0.001, inputs=new_inputs)
        inputs_ref = {"Clutch1_in": 3.0, "Clutch2_in": 2.0, "Clutch3_in": 1.0, "Torque_in": 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        # Test inputs have been set to starting values AFTER NEW INITIALIZATION
        twin.initialize_evaluation()
        inputs_ref = {"Clutch1_in": 0.0, "Clutch2_in": 0.0, "Clutch3_in": 0.0, "Torque_in": 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        # Test inputs have been well updated after step by step evaluation
        new_inputs = {"Clutch1_in": 2.0, "Clutch2_in": 2.0}
        twin.evaluate_step_by_step(step_size=0.001, inputs=new_inputs)
        inputs_ref = {"Clutch1_in": 2.0, "Clutch2_in": 2.0, "Clutch3_in": 0.0, "Torque_in": 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)

    def test_inputs_and_parameters_initialization(self):
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # TEST DEFAULT VALUES BEFORE FIRST INITIALIZATION
        inputs_default = {"Clutch1_in": 0.0, "Clutch2_in": 0.0, "Clutch3_in": 0.0, "Torque_in": 0.0}
        parameters_default = {
            "CoupledClutches1_Inert1_J": 1.0,
            "CoupledClutches1_Inert2_J": 1.0,
            "CoupledClutches1_Inert3_J": 1.0,
            "CoupledClutches1_Inert4_J": 1.0,
        }
        assert compare_dictionary(twin.inputs, inputs_default)
        assert compare_dictionary(twin.parameters, parameters_default)
        # TEST INITIALIZATION UPDATES VALUES
        inputs = {"Clutch1_in": 1.0, "Clutch2_in": 1.0, "Clutch3_in": 1.0, "Torque_in": 1.0}
        parameters = {
            "CoupledClutches1_Inert1_J": 2.0,
            "CoupledClutches1_Inert2_J": 2.0,
            "CoupledClutches1_Inert3_J": 2.0,
            "CoupledClutches1_Inert4_J": 2.0,
        }
        twin.initialize_evaluation(parameters=parameters, inputs=inputs)
        assert compare_dictionary(twin.inputs, inputs)
        assert compare_dictionary(twin.parameters, parameters)
        # TEST NEW INITIALIZATION OVERRIDES PREVIOUS VALUES IF GIVEN.
        # OTHERWISE, RESET VALUES TO DEFAULT.
        new_inputs = {"Clutch1_in": 2.0, "Clutch2_in": 2.0}
        new_parameters = {"CoupledClutches1_Inert1_J": 3.0, "CoupledClutches1_Inert2_J": 3.0}
        twin.initialize_evaluation(parameters=new_parameters, inputs=new_inputs)
        new_inputs_ref = {"Clutch1_in": 2.0, "Clutch2_in": 2.0, "Clutch3_in": 0.0, "Torque_in": 0.0}
        new_parameters_ref = {
            "CoupledClutches1_Inert1_J": 3.0,
            "CoupledClutches1_Inert2_J": 3.0,
            "CoupledClutches1_Inert3_J": 1.0,
            "CoupledClutches1_Inert4_J": 1.0,
        }
        assert compare_dictionary(twin.inputs, new_inputs_ref)
        assert compare_dictionary(twin.parameters, new_parameters_ref)
        # TEST NEW INITIALIZATION RESET VALUES TO DEFAULT IF NOT GIVEN (ALL NONE)
        twin.initialize_evaluation()
        assert compare_dictionary(twin.inputs, inputs_default)
        assert compare_dictionary(twin.parameters, parameters_default)
        # TEST NEW INITIALIZATION RESET VALUES TO DEFAULT IF NOT GIVEN (PARAMETER ONLY, INPUT=NONE --> DEFAULT)
        twin.initialize_evaluation(parameters=new_parameters, inputs=new_inputs)
        assert compare_dictionary(twin.inputs, new_inputs_ref)
        assert compare_dictionary(twin.parameters, new_parameters_ref)
        twin.initialize_evaluation(parameters=new_parameters)
        assert compare_dictionary(twin.inputs, inputs_default)
        assert compare_dictionary(twin.parameters, new_parameters_ref)
        # TEST NEW INITIALIZATION RESET VALUES TO DEFAULT IF NOT GIVEN (INPUTS ONLY, PARAMETER=NONE --> DEFAULT)
        twin.initialize_evaluation(parameters=new_parameters, inputs=new_inputs)
        assert compare_dictionary(twin.inputs, new_inputs_ref)
        assert compare_dictionary(twin.parameters, new_parameters_ref)
        twin.initialize_evaluation(inputs=new_inputs)
        assert compare_dictionary(twin.inputs, new_inputs_ref)
        assert compare_dictionary(twin.parameters, parameters_default)

    def test_inputs_property_with_batch_eval(self):
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # Test inputs after BATCH EVALUATION
        twin.initialize_evaluation()
        twin.evaluate_batch(pd.DataFrame({"Time": [0, 1]}))
        inputs_ref = {"Clutch1_in": 0.0, "Clutch2_in": 0.0, "Clutch3_in": 0.0, "Torque_in": 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)

    def test_outputs_property_with_step_by_step_eval(self):
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # Test outputs have None values JUST AFTER INSTANTIATION
        outputs_ref = {"Clutch1_torque": None, "Clutch2_torque": None, "Clutch3_torque": None}
        assert compare_dictionary(twin.outputs, outputs_ref)
        # Test outputs have good values AFTER FIRST EVALUATION INITIALIZATION
        twin.initialize_evaluation()
        outputs_ref = {"Clutch1_torque": 0.0, "Clutch2_torque": 0.0, "Clutch3_torque": 0.0}
        assert compare_dictionary(twin.outputs, outputs_ref)
        # Test outputs have good values AFTER STEP BY STEP EVALUATION
        new_inputs = {"Clutch1_in": 1.0, "Clutch2_in": 1.0}
        twin.evaluate_step_by_step(step_size=0.001, inputs=new_inputs)
        outputs_ref = {"Clutch1_torque": -10.0, "Clutch2_torque": -5.0, "Clutch3_torque": 0.0}
        assert compare_dictionary(twin.outputs, outputs_ref)
        # Test outputs have good values AFTER NEW EVALUATION INITIALIZATION
        twin.initialize_evaluation()
        outputs_ref = {"Clutch1_torque": 0.0, "Clutch2_torque": 0.0, "Clutch3_torque": 0.0}
        assert compare_dictionary(twin.outputs, outputs_ref)

    def test_outputs_property_with_batch_eval(self):
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # Test outputs after BATCH EVALUATION
        twin.initialize_evaluation()
        twin.evaluate_batch(pd.DataFrame({"Time": [0, 1]}))
        inputs_ref = {"Clutch1_in": 0.0, "Clutch2_in": 0.0, "Clutch3_in": 0.0, "Torque_in": 0.0}
        outputs_ref = {"Clutch1_torque": 0.0, "Clutch2_torque": 0.0, "Clutch3_torque": 0.0}
        assert compare_dictionary(twin.outputs, outputs_ref)

    def test_raised_errors_with_step_by_step_evaluation(self):
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # Raise an error if TWIN MODEL HAS NOT BEEN INITIALIZED
        with pytest.raises(TwinModelError) as e:
            twin.evaluate_step_by_step(step_size=0.001)
        assert "Please initialize evaluation" in str(e)
        # Raise an error if STEP SIZE IS ZERO
        with pytest.raises(TwinModelError) as e:
            twin.initialize_evaluation()
            twin.evaluate_step_by_step(step_size=0.0)
        assert "Step size must be strictly bigger than zero" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.initialize_evaluation()
            twin.evaluate_step_by_step(step_size=-0.1)
        assert "Step size must be strictly bigger than zero" in str(e)

    def test_raised_errors_with_batch_evaluation(self):
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # Raise an error if TWIN MODEL HAS NOT BEEN INITIALIZED
        with pytest.raises(TwinModelError) as e:
            twin.evaluate_batch(pd.DataFrame())
        assert "Please initialize evaluation" in str(e)
        # Raise an error if INPUTS DATAFRAME HAS NO TIME COLUMN
        with pytest.raises(TwinModelError) as e:
            twin.initialize_evaluation()
            twin.evaluate_batch(pd.DataFrame())
        assert "Please provide a dataframe with a 'Time' column to use batch mode evaluation" in str(e)
        # Raise an error if INPUTS DATAFRAME HAS NO TIME INSTANT ZERO
        with pytest.raises(TwinModelError) as e:
            twin.initialize_evaluation()
            twin.evaluate_batch(pd.DataFrame({"Time": [0.1]}))
        assert "Please provide inputs at time instant t=0.s" in str(e)
        # Raise an error if INPUTS DATAFRAME HAS NO TIME INSTANT ZERO
        with pytest.raises(TwinModelError) as e:
            twin.initialize_evaluation()
            twin.evaluate_batch(pd.DataFrame({"Time": [1e-50]}))
        assert "Please provide inputs at time instant t=0.s" in str(e)

    def test_evaluation_methods_give_same_results(self):
        inputs_df = pd.DataFrame(
            {"Time": [0.0, 0.1, 0.2, 0.3], "Clutch1_in": [0.0, 1.0, 2.0, 3.0], "Clutch2_in": [0.0, 1.0, 2.0, 3.0]}
        )
        sbs_outputs = {"Time": [], "Clutch1_torque": [], "Clutch2_torque": [], "Clutch3_torque": []}
        # Evaluate twin model with STEP BY STEP EVALUATION
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # t=0. (s)
        t_idx = 0
        twin.initialize_evaluation(
            inputs={"Clutch1_in": inputs_df["Clutch1_in"][t_idx], "Clutch2_in": inputs_df["Clutch2_in"][t_idx]}
        )
        sbs_outputs["Time"].append(twin.evaluation_time)
        for name in twin.outputs:
            sbs_outputs[name].append(twin.outputs[name])
        for t_idx in range(1, inputs_df.shape[0]):
            # Evaluate state at instant t + step_size with inputs from instant t
            step_size = inputs_df["Time"][t_idx] - inputs_df["Time"][t_idx - 1]
            new_inputs = {
                "Clutch1_in": inputs_df["Clutch1_in"][t_idx - 1],
                "Clutch2_in": inputs_df["Clutch2_in"][t_idx - 1],
            }
            twin.evaluate_step_by_step(step_size=step_size, inputs=new_inputs)
            sbs_outputs["Time"].append(twin.evaluation_time)
            for name in twin.outputs:
                sbs_outputs[name].append(twin.outputs[name])
        # Evaluate twin model with BATCH EVALUATION
        twin.initialize_evaluation(
            inputs={"Clutch1_in": inputs_df["Clutch1_in"][0], "Clutch2_in": inputs_df["Clutch2_in"][0]}
        )
        outputs_df = twin.evaluate_batch(inputs_df)
        # Compare STEP-BY-STEP vs BATCH RESULTS
        sbs_outputs_df = pd.DataFrame(sbs_outputs)
        assert pd.DataFrame.equals(sbs_outputs_df, outputs_df)

    def test_evaluation_initialization_with_config_file(self):
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # Evaluation initialization with VALID CONFIG FILE
        config_filepath = os.path.join(os.path.dirname(__file__), "data", "eval_init_config.json")
        twin.initialize_evaluation(json_config_filepath=config_filepath)
        inputs_ref = {"Clutch1_in": 1.0, "Clutch2_in": 1.0, "Clutch3_in": 1.0, "Torque_in": 1.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        parameters_ref = {
            "CoupledClutches1_Inert1_J": 2.0,
            "CoupledClutches1_Inert2_J": 2.0,
            "CoupledClutches1_Inert3_J": 2.0,
            "CoupledClutches1_Inert4_J": 2.0,
        }
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Evaluation initialization IGNORE INVALID PARAMETER AND INPUT ENTRIES
        config_filepath = os.path.join(os.path.dirname(__file__), "data", "eval_init_config_invalid_keys.json")
        twin.initialize_evaluation(json_config_filepath=config_filepath)
        inputs_ref = {"Clutch1_in": 1.0, "Clutch2_in": 1.0, "Clutch3_in": 1.0, "Torque_in": 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        parameters_ref = {
            "CoupledClutches1_Inert1_J": 2.0,
            "CoupledClutches1_Inert2_J": 2.0,
            "CoupledClutches1_Inert3_J": 2.0,
            "CoupledClutches1_Inert4_J": 1.0,
        }
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Evaluation initialization WITH ONLY PARAMETERS ENTRIES
        config_filepath = os.path.join(os.path.dirname(__file__), "data", "eval_init_config_only_parameters.json")
        twin.initialize_evaluation(json_config_filepath=config_filepath)
        inputs_ref = {"Clutch1_in": 0.0, "Clutch2_in": 0.0, "Clutch3_in": 0.0, "Torque_in": 0.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        parameters_ref = {
            "CoupledClutches1_Inert1_J": 2.0,
            "CoupledClutches1_Inert2_J": 2.0,
            "CoupledClutches1_Inert3_J": 2.0,
            "CoupledClutches1_Inert4_J": 2.0,
        }
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Evaluation initialization WITH ONLY INPUT ENTRIES
        config_filepath = os.path.join(os.path.dirname(__file__), "data", "eval_init_config_only_inputs.json")
        twin.initialize_evaluation(json_config_filepath=config_filepath)
        inputs_ref = {"Clutch1_in": 1.0, "Clutch2_in": 1.0, "Clutch3_in": 1.0, "Torque_in": 1.0}
        assert compare_dictionary(twin.inputs, inputs_ref)
        parameters_ref = {
            "CoupledClutches1_Inert1_J": 1.0,
            "CoupledClutches1_Inert2_J": 1.0,
            "CoupledClutches1_Inert3_J": 1.0,
            "CoupledClutches1_Inert4_J": 1.0,
        }
        assert compare_dictionary(twin.parameters, parameters_ref)
        # Evaluation initialization RAISE AN ERROR IF CONFIG FILEPATH DOES NOT EXIST
        with pytest.raises(TwinModelError) as e:
            twin.initialize_evaluation(json_config_filepath="filepath_does_not_exist")
        assert "Please provide an existing filepath to initialize the twin model evaluation" in str(e)

    def test_close_method(self):
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        twin = TwinModel(model_filepath=model_filepath)
        twin = TwinModel(model_filepath=model_filepath)

    def test_each_twin_model_has_a_subfolder_in_wd(self):
        # Init unit test
        reinit_settings_for_unit_tests()
        logger = get_pytwin_logger()
        # Verify a subfolder is created each time a new twin model is instantiated
        m_count = 10
        for m in range(m_count):
            model = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)
            time.sleep(0.5)
        wd = get_pytwin_working_dir()
        temp = os.listdir(wd)
        assert len(os.listdir(wd)) == m_count + 2

    def test_model_dir_migration_after_modifying_wd_dir(self):
        # Init unit test
        wd = reinit_settings()
        assert not os.path.exists(wd)
        model = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)
        assert os.path.split(model.model_dir)[0] == get_pytwin_working_dir()
        # Run test
        modify_pytwin_working_dir(new_path=wd)
        assert os.path.split(model.model_dir)[0] == wd
        assert len(os.listdir(wd)) == 1 + 1  # model + pytwin log
        model2 = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)
        assert os.path.split(model2.model_dir)[0] == wd
        assert len(os.listdir(wd)) == 2 + 1 + 1  # 2 models + pytwin log + .temp

    def test_model_warns_at_initialization(self):
        # Init unit test
        wd = reinit_settings()
        model = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)
        log_file = get_pytwin_log_file()
        # Warns if given parameters have wrong names
        wrong_params = {}
        for p in model.parameters:
            wrong_params[f"{p}%"] = 0.0
        model.initialize_evaluation(parameters=wrong_params)
        with open(log_file, "r") as f:
            lines = f.readlines()
        msg = "has not been found in model parameters!"
        assert "".join(lines).count(msg) == 4
        # Warns if given inputs have wrong names
        wrong_inputs = {}
        for i in model.inputs:
            wrong_inputs[f"{i}%"] = 0.0
        model.initialize_evaluation(inputs=wrong_inputs)
        with open(log_file, "r") as f:
            lines = f.readlines()
        msg = "has not been found in model inputs!"
        assert "".join(lines).count(msg) == 4

    def test_model_warns_at_evaluation_step_by_step(self):
        # Init unit test
        wd = reinit_settings()
        model = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)
        log_file = get_pytwin_log_file()
        model.initialize_evaluation()
        # Warns if given inputs have wrong names
        wrong_inputs = {}
        for i in model.inputs:
            wrong_inputs[f"{i}%"] = 0.0
        model.evaluate_step_by_step(step_size=0.1, inputs=wrong_inputs)
        with open(log_file, "r") as f:
            lines = f.readlines()
        msg = "has not been found in model inputs!"
        assert "".join(lines).count(msg) == 4

    def test_model_warns_at_evaluation_batch(self):
        # Init unit test
        wd = reinit_settings()
        model = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)
        log_file = get_pytwin_log_file()
        model.initialize_evaluation()
        # Warns if given inputs have wrong names
        wrong_inputs_df = pd.DataFrame({"Time": [0.0, 0.1], "Clutch1_in%": [0.0, 1.0], "Clutch2_in%": [0.0, 1.0]})
        model.evaluate_batch(inputs_df=wrong_inputs_df)
        with open(log_file, "r") as f:
            lines = f.readlines()
        msg = "has not been found in model inputs!"
        assert "".join(lines).count(msg) == 2

    def test_save_and_load_state_multiple_times(self):
        # Init unit test
        wd = reinit_settings()
        # Save state test
        model1 = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)
        model2 = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)

        model1.initialize_evaluation()

        model1.evaluate_step_by_step(step_size=0.01, inputs={"Clutch1_in": 1.0})
        model1.save_state()
        model2.load_state(model1.id, model1.evaluation_time)
        assert compare_dictionary(model1.outputs, model2.outputs)

        model1.evaluate_step_by_step(step_size=0.01, inputs={"Clutch1_in": 2.0})
        model1.save_state()
        model2.load_state(model1.id, model1.evaluation_time)
        assert compare_dictionary(model1.outputs, model2.outputs)

        model1.evaluate_step_by_step(step_size=0.01, inputs={"Clutch1_in": 3.0})
        model1.save_state()
        model2.load_state(model1.id, model1.evaluation_time)
        assert compare_dictionary(model1.outputs, model2.outputs)

        model1.evaluate_step_by_step(step_size=0.01, inputs={"Clutch1_in": 4.0})
        model1.save_state()
        model2.load_state(model1.id, model1.evaluation_time)
        assert compare_dictionary(model1.outputs, model2.outputs)

    def test_save_and_load_state_with_coupled_clutches(self):
        # Init unit test
        wd = reinit_settings()
        # Save state test
        model1 = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)
        model1.initialize_evaluation()
        model1.evaluate_step_by_step(step_size=0.01, inputs={"Clutch1_in": 1.0})
        model1.evaluate_step_by_step(step_size=0.01, inputs={"Clutch1_in": 2.0})
        model1.evaluate_step_by_step(step_size=0.01, inputs={"Clutch1_in": 3.0})
        model1.evaluate_step_by_step(step_size=0.01, inputs={"Clutch1_in": 4.0})
        model1.evaluate_step_by_step(step_size=0.01, inputs={"Clutch1_in": 5.0})
        model1.save_state()
        out1 = model1.outputs
        # Load state test
        model2 = TwinModel(model_filepath=COUPLE_CLUTCHES_FILEPATH)
        model2.load_state(model1.id, model1.evaluation_time)
        out2 = model2.outputs
        assert compare_dictionary(out1, out2)
        # Progress step by step evaluations give same results
        model1.evaluate_step_by_step(step_size=0.05)
        model2.evaluate_step_by_step(step_size=0.05)
        out1 = model1.outputs
        out2 = model2.outputs
        assert compare_dictionary(out1, out2)

    def test_save_and_load_state_with_dynarom(self):
        # Init unit test
        wd = reinit_settings()
        # Save state test
        model1 = TwinModel(model_filepath=DYNAROM_HX_23R1)
        model1.initialize_evaluation()
        model1.evaluate_step_by_step(step_size=0.1, inputs={"HeatFlow": 1})
        model1.evaluate_step_by_step(step_size=0.5, inputs={"HeatFlow": 10})
        model1.evaluate_step_by_step(step_size=1, inputs={"HeatFlow": 100})
        model1.evaluate_step_by_step(step_size=5, inputs={"HeatFlow": 1000})
        model1.save_state()
        out1 = model1.outputs
        # Load state test
        model2 = TwinModel(model_filepath=DYNAROM_HX_23R1)
        model2.load_state(model1.id, model1.evaluation_time)
        out2 = model2.outputs
        assert compare_dictionary(out1, out2)
        # Progress step by step evaluations give same results
        model1.evaluate_step_by_step(step_size=5)
        model2.evaluate_step_by_step(step_size=5)
        out1 = model1.outputs
        out2 = model2.outputs
        assert not compare_dictionary(out1, out2)  # TODO - Fix BU732106

    def test_save_and_load_state_with_rc_heat_circuit(self):
        # Init unit test
        wd = reinit_settings()
        # Save state test
        model1 = TwinModel(model_filepath=RC_HEAT_CIRCUIT_23R1)
        model1.initialize_evaluation(parameters={"SimModel2_C": 10.0})
        model1.evaluate_step_by_step(step_size=1, inputs={"heat_in": 100})
        model1.evaluate_step_by_step(step_size=1)
        model1.evaluate_step_by_step(step_size=1)
        model1.save_state()
        # Load state test
        model2 = TwinModel(model_filepath=RC_HEAT_CIRCUIT_23R1)
        model2.load_state(model1.id, model1.evaluation_time)
        assert compare_dictionary(model1.outputs, model2.outputs)
        assert compare_dictionary(model1.inputs, model2.inputs)
        assert compare_dictionary(model1.parameters, model2.parameters)
        assert model1.evaluation_time == model2.evaluation_time
        # Progress step by step evaluations give same results
        model1.evaluate_step_by_step(step_size=10)
        model2.evaluate_step_by_step(step_size=10)
        assert compare_dictionary(model1.outputs, model2.outputs)

    def test_raised_errors_with_tbrom_resource_directory(self):
        wd = reinit_settings()
        model_filepath = COUPLE_CLUTCHES_FILEPATH
        twin = TwinModel(model_filepath=model_filepath)
        # Raise an error if TWIN MODEL IS NOT INITIALIZED
        with pytest.raises(TwinModelError) as e:
            twin._tbrom_resource_directory(rom_name="test")
        assert "Please initialize evaluation" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_geometry_filepath(rom_name="test")
        assert "Please initialize evaluation" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_snapshot_filepath(rom_name="test")
        assert "Please initialize evaluation" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_available_view_names(rom_name="test")
        assert "Please initialize evaluation" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_image_filepath(rom_name="test", view_name="test")
        assert "Please initialize evaluation" in str(e)
        twin.initialize_evaluation()
        # Raise an error if TWIN MODEL DOES NOT INCLUDE ANY TBROM
        with pytest.raises(TwinModelError) as e:
            twin._tbrom_resource_directory(rom_name="test")
        assert "Twin model does not include any TBROM!" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_geometry_filepath(rom_name="test")
        assert "Twin model does not include any TBROM!" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_snapshot_filepath(rom_name="test")
        assert "Twin model does not include any TBROM!" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_available_view_names(rom_name="test")
        assert "Twin model does not include any TBROM!" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_image_filepath(rom_name="test", view_name="test")
        assert "Twin model does not include any TBROM!" in str(e)
        model_filepath = examples.download_file("ThermalTBROM_23R1_other.twin", "twin_files")
        twin = TwinModel(model_filepath=model_filepath)
        twin.initialize_evaluation()
        # Raise an error if TWIN MODEL DOES NOT INCLUDE ANY TBROM NAMED 'test'
        with pytest.raises(TwinModelError) as e:
            twin._tbrom_resource_directory(rom_name="test")
        assert "Twin model does not include any TBROM named" in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_geometry_filepath(rom_name="test")
        assert "Please call the geometry file getter with a valid TBROM name." in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_snapshot_filepath(rom_name="test")
        assert "Please call the snapshot file getter with a valid TBROM name." in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_available_view_names(rom_name="test")
        assert "Please call this method with a valid TBROM name." in str(e)
        with pytest.raises(TwinModelError) as e:
            twin.get_image_filepath(rom_name="test", view_name="test")
        assert "Please call this method with a valid TBROM name." in str(e)
        # Raise an error if IMAGE VIEW DOES NOT EXIST
        with pytest.raises(TwinModelError) as e:
            twin.get_image_filepath(rom_name=twin.tbrom_names[0], view_name="test")
        assert "Please call this method with a valid view name." in str(e)
        # Raise an error if GEOMETRY POINT FILE HAS BEEN DELETED
        with pytest.raises(TwinModelError) as e:
            filepath = twin.get_geometry_filepath(rom_name=twin.tbrom_names[0])
            os.remove(filepath)
            twin.get_geometry_filepath(rom_name=twin.tbrom_names[0])
        assert "Could not find the geometry file for given available rom_name" in str(e)
        # Raise a warning if SNAPSHOT FILE AT GIVEN EVALUATION TIME DOES NOT EXIST
        twin.get_snapshot_filepath(rom_name=twin.tbrom_names[0], evaluation_time=1.234567)
        log_file = get_pytwin_log_file()
        with open(log_file, "r") as log:
            log_str = log.readlines()
        assert "Could not find the snapshot file for given available rom_name" in "".join(log_str)
        # Raise a warning if IMAGE FILE AT GIVEN EVALUATION TIME DOES NOT EXIST
        twin.get_image_filepath(
            rom_name=twin.tbrom_names[0],
            view_name=twin.get_available_view_names(twin.tbrom_names[0])[0],
            evaluation_time=1.234567,
        )
        log_file = get_pytwin_log_file()
        with open(log_file, "r") as log:
            log_str = log.readlines()
        assert "Could not find the image file for given available rom_name" in "".join(log_str)

    @pytest.mark.skip("TODO - FIX ISSUE TO DELETE LOG FILE WHEN TBROM IS USED")
    def test_clean_unit_test(self):
        reinit_settings()
