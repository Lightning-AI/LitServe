import inspect
import sys
from typing import Dict, List, Optional

import pytest
from pydantic import BaseModel, Field

if sys.version_info < (3, 10):
    pytest.skip("Skipping test_mcp.py on Python < 3.10", allow_module_level=True)


from litserve.mcp import _param_name_to_title, _python_type_to_json_schema, extract_input_schema


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
    assert _python_type_to_json_schema(List[str]) == "array"
    assert _python_type_to_json_schema(Dict[str, int]) == "object"

    # Test nested optional types
    assert _python_type_to_json_schema(Optional[List[str]]) == {"type": "array", "nullable": True}


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
