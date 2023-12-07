import os
import shutil
import time
import tracemalloc

from pytwin import TwinModel
from pytwin.settings import get_pytwin_working_dir

TBROM_MODEL_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "ThermalTBROM_FieldInput_23R1.twin")
UNIT_TEST_WD = os.path.join(os.path.dirname(__file__), "unit_test_wd")


def reinit_settings():
    from pytwin.settings import reinit_settings_for_unit_tests

    reinit_settings_for_unit_tests()
    if os.path.exists(UNIT_TEST_WD):
        try:
            shutil.rmtree(UNIT_TEST_WD)
        except Exception as e:
            pass
    return UNIT_TEST_WD


class TestTwinModelFinalize:
    def test_twin_model_finalizer_free_memory(self):
        # Init unit test
        reinit_settings()
        # TwinModel memory is freed at the end of a loop and its model directory is deleted
        tracemalloc.start()
        snapshot = tracemalloc.take_snapshot()
        allocated_mem_size = ""
        model_dir = ""
        for i in range(3):
            twin_model = TwinModel(model_filepath=TBROM_MODEL_FILEPATH)
            model_dir_old = model_dir
            model_dir = twin_model.model_dir
            snapshot2 = tracemalloc.take_snapshot()
            top_stats = snapshot2.compare_to(snapshot, "lineno")
            allocated_mem_size_old = allocated_mem_size
            allocated_mem_size = f"{top_stats[0]}".split("size=")[1].split(",")[0].split("+")[1].split(" ")[0]
            time.sleep(0.25)
            if i > 0:
                # Current twin_model directory exists
                assert os.path.exists(model_dir)
                # Previous twin_model directory as been deleted
                assert not os.path.exists(model_dir_old)
                # Previous twin_model memory as been freed (allow for +/- 0.5% difference of memory
                assert (
                    1.005 * int(allocated_mem_size_old) > int(allocated_mem_size) > 0.995 * int(allocated_mem_size_old)
                )

    def test_clean_unit_test(self):
        reinit_settings()
        temp_wd = get_pytwin_working_dir()
        parent_dir = os.path.dirname(temp_wd)
        try:
            for dir_name in os.listdir(parent_dir):
                if dir_name not in temp_wd:
                    shutil.rmtree(os.path.join(parent_dir, dir_name))
        except Exception as e:
            pass


"""
import os
import tracemalloc
from pytwin import TwinModel
tracemalloc.start()
snapshot = tracemalloc.take_snapshot()
twin_file = os.path.join(os.path.dirname(__file__), "data", "ThermalTBROM_FieldInput_23R1.twin")
for i in range(1, 10):
    #with TwinModel(model_filepath=twin_file) as twin_model:
    #    snapshot2 = tracemalloc.take_snapshot()
    #    top_stats = snapshot2.compare_to(snapshot, 'lineno')
    #    stat_tbrom_read_basis = top_stats[0]
    #    print(stat_tbrom_read_basis)
    #    print("end of the loop")
    twin_model = TwinModel(model_filepath=twin_file)
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot, 'lineno')
    stat_tbrom_read_basis = top_stats[0]
    print(stat_tbrom_read_basis)
    print("end of the loop")
    twin_model.close()
"""
