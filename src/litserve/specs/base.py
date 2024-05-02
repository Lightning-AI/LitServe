class LitSpec:
    _endpoints = []

    def setup(self, obj):
        raise NotImplementedError()

    def add_endpoint(self, path, endpoint, methods):
        """Register an endpoint in the spec."""
        self._endpoints.append((path, endpoint, methods))

    @property
    def endpoints(self):
        return self._endpoints.copy()
