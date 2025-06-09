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
"""This module helps create MCP servers for LitServe endpoints."""

import logging
import weakref
from typing import TYPE_CHECKING, Optional

import httpx

from litserve.utils import is_package_installed

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mcp.server.fastmcp.tools.base import Tool

    from litserve.api import LitAPI


class LitMCPSpec:
    """LitMCPSpec is a spec that can be used to create MCP tools for LitServe endpoints.

    It doesn't affect LitAPI.

    """

    def __init__(
        self,
        description: Optional[str] = None,
        input_schema: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        """
        Args:
            name: The name of the MCP tool.
            description: The description of the MCP tool.
            input_schema: The input schema of the MCP tool.
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def _connect(self, lit_api: "LitAPI"):
        # avoid tight coupling between LitAPI and LitMCPSpec
        self.lit_api = weakref.proxy(lit_api)

    async def client_fn(self, request: dict) -> dict:
        # get port from LitServer
        endpoint = f"http://localhost:8000{self.lit_api.api_path}"
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=request)
            return response.json()

    def as_tool(self) -> "Tool":
        if not is_package_installed("mcp"):
            raise RuntimeError("MCP is not installed. Please install it with `uv pip install mcp[cli]`")

        from mcp.server.fastmcp.tools.base import Tool

        name = self.name or self.lit_api.api_path
        decode_tool = Tool.from_function(self.lit_api.decode_request)
        description = self.description or self.lit_api.__doc__
        input_schema = self.input_schema or decode_tool.parameters
        logger.info(f"Creating MCP tool for {name} with description {description}")
        logger.info(f"Input schema: {input_schema}")

        # create a tool for the lit_api
        return Tool(
            fn=self.client_fn,
            name=name,
            description=self.description,
            parameters=decode_tool.parameters,
            fn_metadata=decode_tool.fn_metadata,
            input_schema=input_schema,
            annotations=decode_tool.annotations,
            is_async=True,
        )
