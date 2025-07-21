import time
import warnings
from abc import ABCMeta

_INIT_THRESHOLD = 1


class TimedInitMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        cls._has_custom_setup = False

        for base in bases:
            if hasattr(base, "setup"):
                base_setup = base.setup

                if "setup" in namespace and namespace["setup"] is not base_setup:
                    cls._has_custom_setup = True
                    break
        else:
            if "setup" in namespace:
                cls._has_custom_setup = True

        return cls

    def __call__(cls, *args, **kwargs):
        start_time = time.perf_counter()
        instance = super().__call__(*args, **kwargs)
        elapsed = time.perf_counter() - start_time

        if elapsed >= _INIT_THRESHOLD and not cls._has_custom_setup:
            warnings.warn(
                f"""
                LitAPI.setup method helps in loading the model only on the required processes.
                {cls.__name__}.__init__ took {elapsed:.2f} seconds to execute, so it looks like that
                you perform model loading or some heavy processing in __init__. Please move heavy one-time
                loading code into {cls.__name__}.setup method.
                """,
                RuntimeWarning,
                stacklevel=2,
            )

        return instance
