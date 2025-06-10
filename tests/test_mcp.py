import inspect
from typing import Optional

from pydantic import BaseModel, Field

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
        "properties": {
            "age": {"title": "Age", "type": "integer"},
            "name": {"title": "Name", "type": "string"},
        },
        "required": ["name", "age"],
        "title": "MCPTestModelArguments",
        "type": "object",
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
        "properties": {"name": {"title": "Name", "type": "string"}, "age": {"title": "Age", "type": "string"}},
        "required": ["age"],
        "title": "MCPTestModelWithFieldsArguments",
        "type": "object",
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


if __name__ == "__main__":
    test_mcp_schema_with_fields()
