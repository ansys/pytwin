import pytwin.twin_runtime.twin_runtime_error as twin_runtime_error


class TestTwinRuntimeError:
    def test_twin_runtime_error(self):
        err = twin_runtime_error.TwinRuntimeError("msg1")
        assert "msg1" in err.message
        err.add_message("msg2")
        assert "msg2" in err.message
