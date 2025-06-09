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

import contextlib
import inspect
import logging
import weakref
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Dict, List, Optional, get_origin

import httpx
import mcp.types as types
from fastapi import FastAPI
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool as ToolType
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from litserve.utils import is_package_installed

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from litserve.api import LitAPI


def extract_input_schema(func) -> Dict[str, Any]:
    """Extract JSON schema for function input parameters from a Python function. Supports regular type annotations,
    Pydantic Fields, and Pydantic BaseModel classes.

    Args:
        func: Python function to analyze

    Returns:
        Dict containing JSON schema for the function's input parameters

    """
    signature = inspect.signature(func)
    properties = {}
    required = []
    defs = {}

    for param_name, param in signature.parameters.items():
        # Skip *args and **kwargs
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        # Get type annotation
        param_type = param.annotation

        # Check if parameter is a Pydantic BaseModel
        if param_type != inspect.Parameter.empty and isinstance(param_type, type) and issubclass(param_type, BaseModel):
            # Generate schema for the BaseModel
            model_schema = param_type.model_json_schema()
            model_name = param_type.__name__

            # Extract definitions if they exist
            if "$defs" in model_schema:
                defs.update(model_schema["$defs"])

            # Remove $defs from the model schema and add it to our defs
            model_schema_clean = {k: v for k, v in model_schema.items() if k != "$defs"}
            defs[model_name] = model_schema_clean

            # Reference the model in properties
            properties[param_name] = {"$ref": f"#/$defs/{model_name}"}

            # BaseModel parameters are always required unless they have a default
            if param.default == param.empty:
                required.append(param_name)

            continue

        # Check if parameter has a Pydantic Field as default
        field_info = None
        has_default = param.default != param.empty

        # Check if it's a Pydantic Field
        if (
            has_default
            and hasattr(param.default, "__class__")
            and (param.default.__class__.__name__ == "FieldInfo" or str(type(param.default)).find("pydantic") != -1)
        ):
            field_info = param.default

        # Convert Python type to JSON schema type
        schema_type = _python_type_to_json_schema(param_type)

        # Create property entry
        property_schema = {"title": _param_name_to_title(param_name), "type": schema_type}

        # Add Field metadata if available
        if field_info is not None:
            # Add description if present
            if hasattr(field_info, "description") and field_info.description:
                property_schema["description"] = field_info.description

            # Add default value if present and not Undefined
            if (
                hasattr(field_info, "default")
                and field_info.default is not ...
                and not str(field_info.default).startswith("<pydantic")
            ):
                property_schema["default"] = field_info.default
            else:
                required.append(param_name)

            # Add constraints
            if hasattr(field_info, "ge") and field_info.ge is not None:
                property_schema["minimum"] = field_info.ge
            if hasattr(field_info, "le") and field_info.le is not None:
                property_schema["maximum"] = field_info.le
            if hasattr(field_info, "gt") and field_info.gt is not None:
                property_schema["exclusiveMinimum"] = field_info.gt
            if hasattr(field_info, "lt") and field_info.lt is not None:
                property_schema["exclusiveMaximum"] = field_info.lt
            if hasattr(field_info, "min_length") and field_info.min_length is not None:
                property_schema["minLength"] = field_info.min_length
            if hasattr(field_info, "max_length") and field_info.max_length is not None:
                property_schema["maxLength"] = field_info.max_length
        else:
            # Regular parameter without Field
            if not has_default:
                required.append(param_name)

        properties[param_name] = property_schema

    # Build the final schema
    schema = {"properties": properties, "required": required, "title": f"{func.__name__}Arguments", "type": "object"}

    # Add $defs if we have any BaseModel definitions
    if defs:
        schema["$defs"] = defs

    return schema


def _python_type_to_json_schema(python_type) -> str:
    """Convert Python type annotation to JSON schema type string."""
    if python_type == inspect.Parameter.empty:
        return "string"  # Default to string if no type annotation

    # Handle basic types
    type_mapping = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}

    # Check if it's a basic type
    if python_type in type_mapping:
        return type_mapping[python_type]

    # Handle typing module types (List, Dict, Optional, etc.)
    origin = get_origin(python_type)
    if origin is not None:
        if origin in type_mapping:
            return type_mapping[origin]
        if origin is list:
            return "array"
        if origin is dict:
            return "object"

    # Default to string for unknown types
    return "string"


def _param_name_to_title(param_name: str) -> str:
    """Convert parameter name to a readable title."""
    # Split on underscores and capitalize each word
    words = param_name.split("_")
    return " ".join(word.capitalize() for word in words)


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

    async def client_fn(self, request):
        # TODO: directly send the request to the endpoint handler
        endpoint = f"http://localhost:8000{self.lit_api.api_path}"
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=request)
            result = response.json()
            print(result)
            return result

    def as_tool(self) -> ToolType:
        if not is_package_installed("mcp"):
            raise RuntimeError("MCP is not installed. Please install it with `uv pip install mcp[cli]`")

        name = self.name or self.lit_api.api_path
        description = self.description or self.lit_api.__doc__

        if not name or len(name) == 0:
            raise ValueError("Name is required for MCP tool")
        if not description or len(description) == 0:
            raise ValueError("Description is required for MCP tool")

        logger.info(f"Creating MCP tool for `{name}` with description `{description}`")

        input_schema = extract_input_schema(self.lit_api.decode_request)
        name = name.replace("/", "_")
        return ToolType(
            name=name,
            description=description,
            inputSchema=input_schema,
        )


class _LitMCPServer:
    def __init__(self):
        self.mcp_app = Server("mcp-streamable-http-stateless")
        self.tools = []
        self.sse = SseServerTransport("/messages")

    @contextlib.asynccontextmanager
    async def lifespan(self, app: Starlette) -> AsyncIterator[None]:
        """Context manager for session manager."""
        try:
            yield
        finally:
            logger.info("MCP server shutting down...")

    def add_tool(self, tool: types.Tool):
        self.tools.append(tool)

    def list_tools(self) -> List[types.Tool]:
        return self.tools

    async def handle_streamable_http(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Add logging to see incoming requests
        if scope["type"] == "http":
            headers = dict(scope.get("headers", []))
            accept_header = headers.get(b"accept", b"").decode()
            print(f"Received request with Accept header: {accept_header}")

        await self.session_manager.handle_request(scope, receive, send)

    async def handle_sse(self, scope: Scope, receive: Receive, send: Send) -> None:
        async with self.sse.connect_sse(scope, receive, send) as streams:
            await self.mcp_app.run(streams[0], streams[1], self.mcp_app.create_initialization_options())

    async def handle_messages(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.sse.handle_post_message(scope, receive, send)

    def launch_with_fastapi(self, app: FastAPI):
        self.mcp_app.list_tools()(self.list_tools)

        # Create an ASGI application using the transport
        starlette_app = Starlette(
            debug=True,
            routes=[
                Mount("/sse", app=self.handle_sse),
                Mount("/messages", app=self.handle_messages),
            ],
            lifespan=self.lifespan,
        )

        app.mount("/mcp", starlette_app)
