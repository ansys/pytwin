from enum import Enum


class TwinRuntimeError(Exception):
    def __init__(self, message, twin_runtime=None, twin_status=None):
        self.message = message
        self.twin_status = twin_status
        if twin_runtime is not None:
            self.dll_message = twin_runtime.twin_get_status_string()
            if twin_status is None:
                self.twin_status = twin_runtime.twin_status

    def add_message(self, new_message):
        self.message += '\n'+new_message


class PropertyStatusFlag(Enum):
    TWIN_VARPROP_OK = 0
    TWIN_VARPROP_NOTDEFINED = 1
    TWIN_VARPROP_NOTAPPLICABLE = 2
    TWIN_VARPROP_INVALIDVAR = 3
    TWIN_VARPROP_ERROR = 4


class PropertyNotDefinedError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime.twin_status


class PropertyNotApplicableError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime.twin_status


class PropertyInvalidError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime.twin_status


class PropertyError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime.twin_status



