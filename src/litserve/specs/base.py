from typing import Tuple, List, Callable


class LitSpec:
    def setup(self, obj):
        raise NotImplementedError()

    @property
    def endpoints(
        self,
    ) -> List[
        Tuple[
            str,
            Callable,
        ]
    ]:
        raise NotImplementedError()
