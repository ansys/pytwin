import pytest
from pytwin import TwinRuntime, TwinRuntimeError
import pytwin.examples.downloads as downloads
import pytwin.twin_runtime.twin_runtime_error as twin_runtime_error


class TestTwinRuntime:
    def test_evaluate_twin_status(self):
        model_fp = downloads.download_file("CoupledClutches_23R1_other.twin", "twin_files", force_download=True)
        twin_runtime = TwinRuntime(model_fp, load_model=True)
        twin_runtime.twin_instantiate()
        # Test TwinRuntime warning
        TwinRuntime.evaluate_twin_status(1, twin_runtime, "unit_test_method")
        # Test TwinRuntime error
        with pytest.raises(TwinRuntimeError) as e:
            TwinRuntime.evaluate_twin_status(2, twin_runtime, "unit_test_method")
        assert "error" in str(e.value)
        # Test TwinRuntime fatal error
        with pytest.raises(TwinRuntimeError) as e:
            TwinRuntime.evaluate_twin_status(3, twin_runtime, "unit_test_method")
        assert "fatal error" in str(e.value)

    def test_evaluate_twin_prop_status(self):
        model_fp = downloads.download_file("CoupledClutches_23R1_other.twin", "twin_files", force_download=True)
        twin_runtime = TwinRuntime(model_fp, load_model=True)
        twin_runtime.twin_instantiate()
        # Property error
        with pytest.raises(twin_runtime_error.PropertyError) as e:
            TwinRuntime.evaluate_twin_prop_status(4, twin_runtime, "unit_test_method", 0)
        assert "error" in str(e.value)
        # Property invalid error
        with pytest.raises(twin_runtime_error.PropertyInvalidError) as e:
            TwinRuntime.evaluate_twin_prop_status(3, twin_runtime, "unit_test_method", 0)
        assert "error" in str(e.value)
        # Property not applicable error
        with pytest.raises(twin_runtime_error.PropertyNotApplicableError) as e:
            TwinRuntime.evaluate_twin_prop_status(2, twin_runtime, "unit_test_method", 0)
        assert "error" in str(e.value)
        # Property not defined error
        with pytest.raises(twin_runtime_error.PropertyNotDefinedError) as e:
            TwinRuntime.evaluate_twin_prop_status(1, twin_runtime, "unit_test_method", 0)
        assert "error" in str(e.value)
