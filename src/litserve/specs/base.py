class LitSpec:
    _endpoints = []

    def setup(self, obj):
        raise NotImplementedError()

    def _add_endpoint(self, path, endpoint, methods):
        self._endpoints.append((path, endpoint, methods))

    @property
    def endpoints(self):
        return self._endpoints.copy()
