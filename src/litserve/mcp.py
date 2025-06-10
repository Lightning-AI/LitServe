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

import inspect
import logging
from typing import TYPE_CHECKING, Any, Dict, Union, get_args, get_origin

from pydantic import BaseModel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


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

    # If func is a Pydantic model class, use its schema directly
    if isinstance(func, type) and issubclass(func, BaseModel):
        model_schema = func.model_json_schema()
        model_name = func.__name__

        # Extract definitions if they exist
        if "$defs" in model_schema:
            defs.update(model_schema["$defs"])

        # Remove $defs from the model schema and add it to our defs
        model_schema_clean = {k: v for k, v in model_schema.items() if k != "$defs"}
        defs[model_name] = model_schema_clean

        # Reference the model in properties
        properties[model_name.lower()] = {"$ref": f"#/$defs/{model_name}"}
        required.append(model_name.lower())

        schema = {
            "properties": properties,
            "required": required,
            "title": f"{func.__name__}Arguments",
            "type": "object",
        }
        if defs:
            schema["$defs"] = defs
        return schema

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
        property_schema = {"title": _param_name_to_title(param_name)}
        if isinstance(schema_type, str):
            property_schema["type"] = schema_type
        else:
            property_schema.update(schema_type)

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


def _python_type_to_json_schema(python_type) -> Union[str, dict]:
    """Convert Python type annotation to JSON schema type string or dict."""
    if python_type == inspect.Parameter.empty:
        return "string"  # Default to string if no type annotation

    # Handle basic types
    type_mapping = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # Check if it's a basic type
    if python_type in type_mapping:
        return type_mapping[python_type]

    # Handle typing module types (List, Dict, Optional, etc.)
    origin = get_origin(python_type)
    if origin is not None:
        # Handle Optional types (Union[T, None])
        if origin is Union:
            args = get_args(python_type)
            if len(args) == 2 and type(None) in args:
                # Get the non-None type
                actual_type = next(arg for arg in args if arg is not type(None))
                base_type = _python_type_to_json_schema(actual_type)
                if isinstance(base_type, str):
                    return {"type": base_type, "nullable": True}
                base_type["nullable"] = True
                return base_type

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
