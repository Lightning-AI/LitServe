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
import os
import secrets
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def _keys_match(provided: Optional[str], expected: Optional[str]) -> bool:
    """Compare two keys in constant time.

    `secrets.compare_digest` avoids leaking the expected key through response timing.

    """
    if not provided or not expected:
        return False
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def no_auth():
    """Dependency for endpoints that don't require authentication."""


def api_key_auth(x_api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
    """Require clients to send `LIT_SERVER_API_KEY` in the `X-API-Key` header."""
    # Read on every call so the variable can be set after `import litserve`.
    if not _keys_match(x_api_key, os.environ.get("LIT_SERVER_API_KEY")):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Make sure the 'X-API-Key' header matches your server's API key.",
        )
