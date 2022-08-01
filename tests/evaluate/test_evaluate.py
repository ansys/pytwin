import pathlib
import pytest

from src.ansys.twin.evaluate.evaluate import TwinModel, TwinState

DICT_1 = {
    "twinFileName": "CoupleClutches_22R2_other.twin"
}


class TestEvaluate:
    @pytest.mark.parametrize('filename', ['CoupleClutches_22R2_other.twin'])
    def test_instantiation(self, filename):
        cwd_address = str(pathlib.Path(__file__).parent.absolute())
        path = str(pathlib.Path(cwd_address, 'data', filename))
        twin = TwinModel(path)
        twin.instantiate_twin_model()
        assert twin.state == TwinState.TWIN_INSTANTIATED.value
