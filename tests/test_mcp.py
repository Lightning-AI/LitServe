from pydantic import BaseModel

from litserve.mcp import extract_input_schema


class MCPTestModel(BaseModel):
    name: str
    age: int


def test_mcp_model():
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


def mcp_test_function(name: str, age: int = 20):
    return None


def test_mcp_model_with_default_values():
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


class NestedMCPTestModel(BaseModel):
    address: str
    nested: MCPTestModel


def test_mcp_model_nested():
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
