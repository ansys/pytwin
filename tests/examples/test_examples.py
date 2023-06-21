from pytwin import TwinModel, TwinRuntime
import pytwin.examples.downloads as dld


class TestExamples:
    def test_twin_model_from_examples_23r1(self):
        assert dld.delete_downloads()
        twin_model_names = [
            "CoupledClutches_23R1_other.twin",
            "ElectricRange_23R1_other.twin",
            "HX_scalarDRB_23R1_other.twin",
            # "HeatExchangerRS_23R1_other.twin",
            "ThermalTBROM_23R1_other.twin",
        ]
        # Test model download, instantiate and initialize
        for name in twin_model_names:
            model_fp = dld.download_file(name, "twin_files", force_download=True)
            model = TwinModel(model_fp)
            if model.parameters:
                model._twin_runtime.twin_set_param_by_index(0, 0)
                model._twin_runtime.twin_set_str_param_by_name(
                    list(model.parameters.keys())[0], list(model.parameters.keys())[0]
                )
            model.initialize_evaluation()
            assert model.evaluation_is_initialized
            assert TwinRuntime.twin_is_cross_platform(model_fp)
            model_v, twin_v = TwinRuntime.get_twin_version(model_fp)
            assert model_v
            assert twin_v == "23.1.1"
            model._twin_runtime.print_model_info()

            if model.parameters:
                model._twin_runtime.print_var_info(list(model.parameters.keys()), len(model.parameters.keys()))

            if model.outputs:
                model._twin_runtime.twin_get_output_by_name(list(model.outputs.keys())[0])

            if model.inputs:
                model._twin_runtime.twin_set_input_by_index(0, 0)
                model._twin_runtime.twin_set_inputs([0] * len(list(model.inputs.keys())))

            if model.tbrom_names:
                rom_name = model.tbrom_names[0]
                views = model.get_available_view_names(rom_name)
                model._twin_runtime.twin_get_default_rom_image_directory(rom_name)
                model._twin_runtime.twin_enable_rom_model_images(rom_name, views)
                model._twin_runtime.twin_get_rom_images_files(rom_name, views)
                model._twin_runtime.twin_disable_rom_model_images(rom_name, views)
                model._twin_runtime.twin_enable_3d_rom_model_data(rom_name)
                model._twin_runtime.twin_get_rom_mode_coef_files(rom_name)
                model._twin_runtime.twin_get_rom_snapshot_files(rom_name)
                model._twin_runtime.twin_disable_3d_rom_model_data(rom_name)
