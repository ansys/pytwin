from enum import Enum


class TwinRuntimeError(Exception):
    def __init__(self, message, twin_runtime=None, twin_status=None):
        super().__init__(message, twin_runtime, twin_status)
        self.message = message
        self.twin_status = twin_status
        if twin_runtime is not None:
            self.dll_message = twin_runtime.twin_get_status_string()
            if twin_status is None:
                self.twin_status = twin_runtime._twin_status

    def __str__(self):
        parts = [self.message]
        if hasattr(self, "dll_message"):
            parts.append(f"DLL message: {self.dll_message}")
        if self.twin_status is not None:
            parts.append(f"Twin status: {self.twin_status}")
        return "\n".join(parts)

    def add_message(self, new_message):
        self.message += "\n" + new_message


class PropertyStatusFlag(Enum):
    TWIN_VARPROP_OK = 0
    TWIN_VARPROP_NOTDEFINED = 1
    TWIN_VARPROP_NOTAPPLICABLE = 2
    TWIN_VARPROP_INVALIDVAR = 3
    TWIN_VARPROP_ERROR = 4


class PropertyNotDefinedError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        super().__init__(message, twin_runtime, property_status_flag)
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime._twin_status

    def __str__(self):
        return (
            f"{self.message}"
            f"\nProperty status: {self.property_status_flag.name}"
            f"\nDLL message: {self.dll_message}"
            f"\nTwin status: {self.twin_status.value}"
        )


class PropertyNotApplicableError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        super().__init__(message, twin_runtime, property_status_flag)
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime._twin_status

    def __str__(self):
        return (
            f"{self.message}"
            f"\nProperty status: {self.property_status_flag.name}"
            f"\nDLL message: {self.dll_message}"
            f"\nTwin status: {self.twin_status.value}"
        )


class PropertyInvalidError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        super().__init__(message, twin_runtime, property_status_flag)
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime._twin_status

    def __str__(self):
        return (
            f"{self.message}"
            f"\nProperty status: {self.property_status_flag.name}"
            f"\nDLL message: {self.dll_message}"
            f"\nTwin status: {self.twin_status.value}"
        )


class PropertyError(Exception):
    def __init__(self, message, twin_runtime, property_status_flag):
        super().__init__(message, twin_runtime, property_status_flag)
        self.property_status_flag = PropertyStatusFlag(property_status_flag)
        self.message = message
        self.dll_message = twin_runtime.twin_get_status_string()
        self.twin_status = twin_runtime._twin_status

    def __str__(self):
        return (
            f"{self.message}"
            f"\nProperty status: {self.property_status_flag.name}"
            f"\nDLL message: {self.dll_message}"
            f"\nTwin status: {self.twin_status.value}"
        )
