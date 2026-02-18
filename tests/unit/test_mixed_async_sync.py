# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import litserve as ls
from litserve import LitServer


class TestMixedAsyncSyncAPI(ls.LitAPI):
    """API with mixed async and sync methods."""

    def setup(self, device):
        self.model = lambda x: x * 2

    def predict(self, x):
        # Sync method
        return self.model(x)

    async def encode_response(self, out):
        # Async method
        return {"output": out}


def test_mixed_async_sync_without_flag():
    """Test that mixed async/sync works without enable_async flag."""
    api = TestMixedAsyncSyncAPI()
    server = LitServer(api)

    # Verify auto-detection worked
    assert api.enable_async is True  # Auto-detected from encode_response
    assert api._async_method_types["predict"] is False
    assert api._async_method_types["encode_response"] is True


def test_all_sync_without_flag():
    """Test that all-sync API works without enable_async flag."""

    class AllSyncAPI(ls.LitAPI):
        def setup(self, device):
            self.model = lambda x: x * 2

        def predict(self, x):
            return self.model(x)

        def encode_response(self, out):
            return {"output": out}

    api = AllSyncAPI()
    assert api.enable_async is False  # No async methods detected


def test_all_async_without_flag():
    """Test that all-async API works without enable_async flag."""

    class AllAsyncAPI(ls.LitAPI):
        def setup(self, device):
            self.model = lambda x: x * 2

        async def predict(self, x):
            return self.model(x)

        async def encode_response(self, out):
            return {"output": out}

    api = AllAsyncAPI()
    assert api.enable_async is True  # Auto-detected
    assert api._async_method_types["predict"] is True
    assert api._async_method_types["encode_response"] is True


def test_explicit_enable_async_true():
    """Test that explicit enable_async=True still works."""
    api = TestMixedAsyncSyncAPI(enable_async=True)
    assert api.enable_async is True


def test_explicit_enable_async_false():
    """Test that explicit enable_async=False overrides auto-detection."""
    api = TestMixedAsyncSyncAPI(enable_async=False)
    assert api.enable_async is False
