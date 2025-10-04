import inspect
import sys
from typing import Optional
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.applications import Starlette

if sys.version_info < (3, 10):
    pytest.skip("Skipping test_mcp.py on Python < 3.10", allow_module_level=True)

import litserve as ls
from litserve.mcp import (
    MCP,
    _LitMCPServerConnector,
    _param_name_to_title,
    _python_type_to_json_schema,
    extract_input_schema,
)


@patch("litserve.mcp.is_package_installed", return_value=False)
def test_mcp_not_installed(mock_is_package_installed):
    with pytest.raises(
        RuntimeError,
        match="mcp package is required for MCP support. To install, run `pip install fastmcp` in the terminal.",
    ):
        MCP()


def test_python_type_to_json_schema():
    assert _python_type_to_json_schema(inspect.Parameter.empty) == "string"
    assert _python_type_to_json_schema(int) == "integer"
    assert _python_type_to_json_schema(float) == "number"
    assert _python_type_to_json_schema(str) == "string"
    assert _python_type_to_json_schema(bool) == "boolean"
    assert _python_type_to_json_schema(list) == "array"
    assert _python_type_to_json_schema(dict) == "object"
    assert _python_type_to_json_schema(Optional[int]) == {"nullable": True, "type": "integer"}
    assert _python_type_to_json_schema(Optional[str]) == {"nullable": True, "type": "string"}
    assert _python_type_to_json_schema(Optional[bool]) == {"nullable": True, "type": "boolean"}
    assert _python_type_to_json_schema(Optional[list]) == {"nullable": True, "type": "array"}
    assert _python_type_to_json_schema(Optional[dict]) == {"nullable": True, "type": "object"}


def test_param_name_to_title():
    assert _param_name_to_title("name") == "Name"
    assert _param_name_to_title("age") == "Age"
    assert _param_name_to_title("is_active") == "Is Active"


class MCPTestModel(BaseModel):
    name: str
    age: int


def test_mcp_schema():
    schema = extract_input_schema(MCPTestModel)
    assert schema == {
        "properties": {"mcptestmodel": {"$ref": "#/$defs/MCPTestModel"}},
        "required": ["mcptestmodel"],
        "title": "MCPTestModelArguments",
        "type": "object",
        "$defs": {
            "MCPTestModel": {
                "properties": {"name": {"title": "Name", "type": "string"}, "age": {"title": "Age", "type": "integer"}},
                "required": ["name", "age"],
                "title": "MCPTestModel",
                "type": "object",
            }
        },
    }, "Must adhere with MCP inputSchema format."


class NestedMCPTestModel(BaseModel):
    address: str
    nested: MCPTestModel


def test_mcp_schema_nested():
    schema = extract_input_schema(NestedMCPTestModel)
    (
        schema
        == {
            "properties": {
                "address": {"title": "Address", "type": "string"},
                "nested": {"$ref": "#/$defs/MCPTestModel"},
            },
            "required": ["address", "nested"],
            "title": "NestedMCPTestModelArguments",
            "type": "object",
            "$defs": {
                "MCPTestModel": {
                    "properties": {
                        "age": {"title": "Age", "type": "integer"},
                        "name": {"title": "Name", "type": "string"},
                    },
                    "required": ["name", "age"],
                    "title": "MCPTestModel",
                    "type": "object",
                }
            },
        },
        "Must adhere with MCP inputSchema format.",
    )


class MCPTestModelWithFields(BaseModel):
    name: str = Field(default="John")
    age: int = Field(ge=0, le=100)


def test_mcp_schema_with_fields():
    schema = extract_input_schema(MCPTestModelWithFields)
    assert schema == {
        "properties": {"mcptestmodelwithfields": {"$ref": "#/$defs/MCPTestModelWithFields"}},
        "required": ["mcptestmodelwithfields"],
        "title": "MCPTestModelWithFieldsArguments",
        "type": "object",
        "$defs": {
            "MCPTestModelWithFields": {
                "properties": {
                    "name": {"default": "John", "title": "Name", "type": "string"},
                    "age": {"maximum": 100, "minimum": 0, "title": "Age", "type": "integer"},
                },
                "required": ["age"],
                "title": "MCPTestModelWithFields",
                "type": "object",
            }
        },
    }


def mcp_test_function(name: str, age: int = 20):
    return None


def test_mcp_schema_with_default_values():
    schema = extract_input_schema(mcp_test_function)
    assert schema == {
        "properties": {
            "age": {"title": "Age", "type": "integer"},
            "name": {"title": "Name", "type": "string"},
        },
        "required": ["name"],
        "title": "mcp_test_functionArguments",
        "type": "object",
    }, "Must adhere with MCP inputSchema format."


def test_python_type_to_json_schema_complex():
    # Test generic types
    assert _python_type_to_json_schema(list[str]) == "array"
    assert _python_type_to_json_schema(dict[str, int]) == "object"

    # Test nested optional types
    assert _python_type_to_json_schema(Optional[list[str]]) == {"type": "array", "nullable": True}


class ModelWithConstraints(BaseModel):
    name: str = Field(min_length=3, max_length=50, description="User's full name")
    age: int = Field(
        gt=0,  # exclusive minimum
        lt=150,  # exclusive maximum
        description="User's age in years",
    )


def test_field_constraints():
    schema = extract_input_schema(ModelWithConstraints)
    print(schema)
    assert schema == {
        "properties": {"modelwithconstraints": {"$ref": "#/$defs/ModelWithConstraints"}},
        "required": ["modelwithconstraints"],
        "title": "ModelWithConstraintsArguments",
        "type": "object",
        "$defs": {
            "ModelWithConstraints": {
                "properties": {
                    "name": {
                        "description": "User's full name",
                        "maxLength": 50,
                        "minLength": 3,
                        "title": "Name",
                        "type": "string",
                    },
                    "age": {
                        "description": "User's age in years",
                        "exclusiveMaximum": 150,
                        "exclusiveMinimum": 0,
                        "title": "Age",
                        "type": "integer",
                    },
                },
                "required": ["name", "age"],
                "title": "ModelWithConstraints",
                "type": "object",
            }
        },
    }


def func_with_special_params(*args, **kwargs):
    pass


def test_args_kwargs_handling():
    schema = extract_input_schema(func_with_special_params)
    assert schema == {"properties": {}, "required": [], "title": "func_with_special_paramsArguments", "type": "object"}


class MCPLitAPI(ls.test_examples.SimpleLitAPI):
    def decode_request(self, request: MCPTestModel) -> int:
        return request.age


def test_mcp_cls():
    mc = MCP(description="A simple API", input_schema={"name": "string"})
    assert mc.description == "A simple API"
    assert mc.input_schema == {"name": "string"}
    assert mc.name is None

    with pytest.raises(RuntimeError, match="MCP is not connected to a LitAPI."):
        mc.as_tool()


def test_mcp_cls_with_lit_api():
    mcp = MCP(description="A simple API", input_schema={"name": "string"})
    api = MCPLitAPI(mcp=mcp)
    tool = mcp.as_tool()
    assert api.mcp is mcp
    assert tool.name == "predict"
    assert tool.endpoint == "/predict"
    assert tool.description == "A simple API"
    assert tool.inputSchema == {"name": "string"}


def test_mcp_cls_with_lit_api_no_input_schema():
    mcp = MCP(description="A simple API")
    api = MCPLitAPI(mcp=mcp)
    tool = mcp.as_tool()
    assert api.mcp is mcp
    assert tool.name == "predict"
    assert tool.endpoint == "/predict"
    assert tool.description == "A simple API"
    assert tool.inputSchema == {
        "properties": {"request": {"$ref": "#/$defs/MCPTestModel"}},
        "required": ["request"],
        "title": "decode_requestArguments",
        "type": "object",
        "$defs": {
            "MCPTestModel": {
                "properties": {"name": {"title": "Name", "type": "string"}, "age": {"title": "Age", "type": "integer"}},
                "required": ["name", "age"],
                "title": "MCPTestModel",
                "type": "object",
            }
        },
    }


def test_mcp_litserve_connector():
    connector = _LitMCPServerConnector()
    mcp = MCP(description="A simple API", input_schema={"name": "string"})
    api = MCPLitAPI(mcp=mcp)
    tool = mcp.as_tool()
    connector.add_tool(tool)
    assert api.mcp is mcp
    assert connector.list_tools() == [tool]
    assert connector.tool_endpoint_connections == {"predict": "/predict"}

    app = FastAPI()
    connector.connect_mcp_server([tool], app)
    mcp_mount = list(filter(lambda route: route.name == "mcp", app.routes))[0]
    assert isinstance(mcp_mount.app, Starlette)


@pytest.mark.asyncio
async def test_mcp_call_handler_with_request_param():
    """Test that _call_handler properly handles the 'request' parameter for endpoint_handler."""
    from litserve.mcp import _call_handler

    # Mock the _convert_to_content function since we don't have the actual MCP package
    import litserve.mcp as mcp_module
    original_convert = getattr(mcp_module, "_convert_to_content", None)
    mcp_module._convert_to_content = lambda x: x

    try:
        # Simulate endpoint_handler signature that expects a 'request' parameter
        async def endpoint_handler(request: MCPTestModel):
            return {"result": f"name={request.name}, age={request.age}"}

        # Test with arguments wrapped in 'request' key (the fix)
        arguments = {"name": "John", "age": 30}
        result = await _call_handler(endpoint_handler, request=arguments)

        assert result == {"result": "name=John, age=30"}
    finally:
        # Restore original function if it existed
        if original_convert:
            mcp_module._convert_to_content = original_convert
