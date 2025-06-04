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

from typing import List
from fastapi import FastAPI
import httpx
from starlette.routing import Mount, BaseRoute, Host
from mcp import Tool
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette


class LitMCPServer:
    def __init__(self, app: FastAPI):
        self._fastapi_app = app

    def _get_endpoints(self) -> List[BaseRoute]:
        return self._fastapi_app.routes

    def endpoint_to_tools(self) -> List[Tool]:
        tools = []
        for endpoint in self._get_endpoints():
            tool_name = endpoint.path.replace("/", "_")
            tool = Tool(
                name=tool_name,
                description="generates predictions",
                inputSchema={"type": "object", "properties": {"input": {"type": "string"}}},
            )
            tools.append(tool)
        return tools

    def connect_mcp_server(self):
        mcp_server = FastMCP("LitServeMCP")

        # Access type-safe lifespan context in tools
        @mcp_server.tool()
        def query_db() -> str:
            return "This is an awesome MCP server example!"

        @mcp_server.tool()
        def calculate_bmi(weight_kg: float, height_m: float) -> float:
            """Calculate BMI given weight in kg and height in meters"""
            return weight_kg / (height_m**2)

        @mcp_server.tool()
        async def fetch_weather(city: str) -> str:
            """Fetch current weather for a city"""
            async with httpx.AsyncClient() as client:
                response = await client.get(f"https://api.weather.com/{city}")
                return response.text

        # app = Starlette(
        #     routes=[
        #         Mount('/', app=mcp_server.sse_app()),
        #     ]
        # )
        self._fastapi_app.mount("/mcp", mcp_server.sse_app())
