"""
MCP to Gemini Function Calling Bridge
===================================

This module bridges MCP tools with Google Gemini's function calling system.
It converts MCP tool schemas to Gemini-compatible function definitions and
handles the execution flow between Gemini and MCP servers.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# Google Gemini imports
from google.genai import types

# MCP imports (with fallback handling)
try:
    from mcp.types import Tool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Tool = None  # No fallback Tool class; handle absence where needed

logger = logging.getLogger(__name__)

@dataclass
class ToolExecutionResult:
    """Result of executing an MCP tool through Gemini"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None

class MCPGeminiBridge:
    """
    Bridge between MCP tools and Gemini's function calling system.

    This class:
    1. Converts MCP tool schemas to Gemini function definitions
    2. Handles function calls from Gemini to MCP servers
    3. Manages the execution context and error handling
    """

    def __init__(self, mcp_client_manager, aura_internal_tools=None):
        self.mcp_client_manager = mcp_client_manager
        self.aura_internal_tools = aura_internal_tools
        self._gemini_functions: List[types.Tool] = []
        self._tool_mapping: Dict[str, Dict[str, Any]] = {}
        self._execution_history: List[ToolExecutionResult] = []

    async def convert_mcp_tools_to_gemini_functions(self) -> List[types.Tool]:
        """
        Convert all available MCP tools to Gemini function definitions.

        Returns:
            List of Gemini Tool objects that can be passed to the model
        """
        try:
            available_tools = []

            # Get external MCP tools if client is available and connected
            if hasattr(self.mcp_client_manager, 'list_all_tools'):
                try:
                    # AuraMCPClient has list_all_tools() method
                    mcp_tools = await self.mcp_client_manager.list_all_tools()
                    for tool_name, tool_info in mcp_tools.items():
                        # Convert to format expected by _convert_single_tool
                        available_tools.append({
                            'name': tool_name,
                            'description': tool_info.get('description', ''),
                            'server': tool_info.get('server', 'unknown'),
                            'parameters': tool_info.get('input_schema', {}),
                            'original_tool': tool_info
                        })
                        logger.debug(f"🔧 Added external MCP tool: {tool_name} from {tool_info.get('server')}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not get external MCP tools: {e}")

            # Get Aura internal tools if available
            if self.aura_internal_tools and hasattr(self.aura_internal_tools, 'get_tool_definitions'):
                try:
                    internal_tools = self.aura_internal_tools.get_tool_definitions()
                    for tool_name, tool_def in internal_tools.items():
                        available_tools.append({
                            'name': tool_name,
                            'description': tool_def.get('description', ''),
                            'server': 'aura-internal',
                            'parameters': tool_def.get('parameters', {}),
                            'original_tool': tool_def
                        })
                        logger.debug(f"🔧 Added internal tool: {tool_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not get internal tools: {e}")

            if not available_tools:
                logger.warning("⚠️ No MCP tools available for conversion")
                return []

            gemini_functions = []

            for tool in available_tools:
                try:
                    gemini_function = self._convert_single_tool(tool)
                    if gemini_function:
                        gemini_functions.append(gemini_function)

                        # Store mapping for execution
                        self._tool_mapping[tool['name']] = {
                            'server': tool.get('server', 'unknown'),
                            'original_tool': tool,
                            'mcp_name': tool['name']
                        }

                except Exception as e:
                    logger.error(f"❌ Failed to convert tool {tool.get('name', 'unknown')}: {e}")
                    continue

            self._gemini_functions = gemini_functions
            logger.info(f"✅ Converted {len(gemini_functions)} MCP tools to Gemini functions")

            return gemini_functions

        except Exception as e:
            logger.error(f"❌ Failed to convert MCP tools to Gemini functions: {e}")
            return []

    def _convert_single_tool(self, tool: Dict[str, Any]) -> Optional[types.Tool]:
        """
        Convert a single MCP tool to a Gemini function definition.

        Args:
            tool: MCP tool information dictionary

        Returns:
            Gemini Tool object or None if conversion fails
        """
        try:
            tool_name = tool['name']
            description = tool.get('description', 'No description available')
            parameters = tool.get('parameters', {})

            # Clean the function name for Gemini (no dots, special chars)
            clean_name = tool_name.replace('.', '_').replace('-', '_')
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')

            # Convert MCP parameters to Gemini schema format
            gemini_parameters = self._convert_parameters_schema(parameters)

            # Create Gemini function declaration
            function_declaration = types.FunctionDeclaration(
                name=clean_name,
                description=f"{description} (MCP tool: {tool_name})",
                parameters=gemini_parameters
            )

            # Store the mapping with clean name
            self._tool_mapping[clean_name] = {
                'server': tool.get('server', 'unknown'),
                'original_tool': tool,
                'mcp_name': tool_name,
                'clean_name': clean_name
            }

            logger.debug(f"🔄 Converted MCP tool '{tool_name}' to Gemini function '{clean_name}'")

            return types.Tool(function_declarations=[function_declaration])

        except Exception as e:
            logger.error(f"❌ Failed to convert tool {tool.get('name', 'unknown')}: {e}")
            return None

    def _convert_parameters_schema(self, mcp_schema: Dict[str, Any]) -> types.Schema:
        """
        Convert MCP parameter schema to Gemini schema format.

        Args:
            mcp_schema: MCP tool parameter schema

        Returns:
            Gemini Schema object
        """
        try:
            # Handle different MCP schema formats
            if not mcp_schema:
                return types.Schema(type=types.Type.OBJECT, properties={})

            # Extract properties and required fields
            properties = mcp_schema.get('properties', {})
            required = mcp_schema.get('required', [])

            # Convert properties to Gemini format
            gemini_properties = {}

            for prop_name, prop_schema in properties.items():
                gemini_prop = self._convert_property_schema(prop_schema)
                if gemini_prop:
                    gemini_properties[prop_name] = gemini_prop

            return types.Schema(
                type=types.Type.OBJECT,
                properties=gemini_properties,
                required=required if required else None
            )

        except Exception as e:
            logger.error(f"❌ Failed to convert parameter schema: {e}")
            return types.Schema(type=types.Type.OBJECT, properties={})

    def _convert_property_schema(self, prop_schema: Dict[str, Any]) -> Optional[types.Schema]:
        """
        Convert a single property schema from MCP to Gemini format.

        Args:
            prop_schema: MCP property schema

        Returns:
            Gemini Schema object or None
        """
        try:
            prop_type = prop_schema.get('type', 'string')
            description = prop_schema.get('description', '')

            # Map MCP types to Gemini types
            type_mapping = {
                'string': types.Type.STRING,
                'integer': types.Type.INTEGER,
                'number': types.Type.NUMBER,
                'boolean': types.Type.BOOLEAN,
                'array': types.Type.ARRAY,
                'object': types.Type.OBJECT
            }

            gemini_type = type_mapping.get(prop_type, types.Type.STRING)

            schema = types.Schema(
                type=gemini_type,
                description=description
            )

            # Handle array items
            if prop_type == 'array' and 'items' in prop_schema:
                items_schema = self._convert_property_schema(prop_schema['items'])
                if items_schema:
                    schema.items = items_schema

            # Handle enum values
            if 'enum' in prop_schema:
                schema.enum = prop_schema['enum']

            return schema

        except Exception as e:
            logger.error(f"❌ Failed to convert property schema: {e}")
            return None

    async def execute_function_call(
        self,
        function_call: types.FunctionCall,
        user_id: str
    ) -> ToolExecutionResult:
        """
        Execute a Gemini function call by routing it to the appropriate MCP tool.

        Args:
            function_call: Gemini function call object
            user_id: User ID for context

        Returns:
            ToolExecutionResult with execution outcome
        """
        start_time = datetime.now()

        try:
            function_name = function_call.name
            arguments = dict(function_call.args) if function_call.args else {}

            logger.info(f"🔧 Executing function call: {function_name} with args: {arguments}")

            # Look up the original MCP tool
            if function_name not in self._tool_mapping:
                error_msg = f"Function {function_name} not found in tool mapping"
                logger.error(f"❌ {error_msg}")
                return ToolExecutionResult(
                    tool_name=function_name or "",
                    success=False,
                    result=None,
                    error=error_msg
                )

            tool_info = self._tool_mapping[function_name]
            mcp_tool_name = tool_info['mcp_name']
            server = tool_info['server']

            # Handle parameter unwrapping - check if arguments are wrapped in a 'params' key
            if 'params' in arguments and len(arguments) == 1:
                try:
                    # Try to parse params as JSON string
                    params_value = arguments['params']
                    if isinstance(params_value, str):
                        arguments = json.loads(params_value)
                        logger.debug(f"🔄 Unwrapped JSON params for {function_name}: {arguments}")
                    elif isinstance(params_value, dict):
                        arguments = params_value
                        logger.debug(f"🔄 Unwrapped dict params for {function_name}: {arguments}")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"⚠️ Could not unwrap params for {function_name}: {e}")
                    # Continue with original arguments if parsing fails

            # Add user_id to arguments if not present and needed
            if 'user_id' not in arguments:
                # Check if the tool requires user_id
                original_tool = tool_info['original_tool']
                parameters = original_tool.get('parameters', {})
                properties = parameters.get('properties', {})
                if 'user_id' in properties:
                    arguments['user_id'] = user_id

            # Execute the MCP tool with appropriate parameter formatting
            if server == 'aura-internal' and self.aura_internal_tools:
                # Internal tools expect individual keyword arguments
                result = await self.aura_internal_tools.execute_tool(mcp_tool_name, arguments)
            else:
                # External MCP tools (FastMCP) have different parameter requirements
                if hasattr(self.mcp_client_manager, 'call_tool'):
                    # Check if tool expects parameters at all
                    original_tool = tool_info['original_tool']
                    parameters = original_tool.get('parameters', {})
                    properties = parameters.get('properties', {})
                    
                    if not properties and not arguments:
                        # Tool expects no parameters and we have no arguments - pass empty dict
                        mcp_arguments = {}
                        logger.debug(f"🔄 No parameters needed for {mcp_tool_name}: {mcp_arguments}")
                    else:
                        # Tool expects parameters - wrap in params structure
                        mcp_arguments = {'params': arguments}
                        logger.debug(f"🔄 Wrapping arguments for external tool {mcp_tool_name}: {mcp_arguments}")
                    
                    result = await self.mcp_client_manager.call_tool(mcp_tool_name, mcp_arguments)
                else:
                    raise ValueError(f"Cannot execute external tool {mcp_tool_name}: MCP client not properly configured")

            execution_time = (datetime.now() - start_time).total_seconds()

            execution_result = ToolExecutionResult(
                tool_name=function_name,
                success=True,
                result=result,
                execution_time=execution_time
            )

            self._execution_history.append(execution_result)
            logger.info(f"✅ Successfully executed {function_name} in {execution_time:.2f}s")

            return execution_result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)

            execution_result = ToolExecutionResult(
                tool_name=function_call.name or "",
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time
            )

            self._execution_history.append(execution_result)
            logger.error(f"❌ Failed to execute {function_call.name}: {error_msg}")

            return execution_result

    def get_available_functions(self) -> List[Dict[str, Any]]:
        """
        Get list of available functions for display/debugging.

        Returns:
            List of function information dictionaries
        """
        functions = []

        for tool in self._gemini_functions:
            if tool.function_declarations:
                for func_decl in tool.function_declarations:
                    functions.append({
                        'name': func_decl.name,
                        'description': func_decl.description,
                        'parameters': self._schema_to_dict(func_decl.parameters) if func_decl.parameters else {}
                    })

        return functions

    def _schema_to_dict(self, schema: types.Schema) -> Dict[str, Any]:
        """Convert Gemini Schema to dictionary for serialization"""
        if schema.type is not None and hasattr(schema.type, 'name'):
            type_name = schema.type.name
        else:
            type_name = str(schema.type)
        result = {
            'type': type_name
        }

        if schema.description:
            result['description'] = schema.description

        if schema.properties:
            result['properties'] = json.dumps({
                name: self._schema_to_dict(prop_schema)
                for name, prop_schema in schema.properties.items()
            })

        if schema.required:
            result['required'] = json.dumps(schema.required)

        if schema.enum:
            result['enum'] = json.dumps(schema.enum)

        return result

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for monitoring.

        Returns:
            Dictionary with execution statistics
        """
        total_executions = len(self._execution_history)
        successful_executions = len([r for r in self._execution_history if r.success])

        return {
            'total_functions': len(self._gemini_functions),
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'recent_executions': [
                {
                    'tool_name': r.tool_name,
                    'success': r.success,
                    'execution_time': r.execution_time,
                    'error': r.error
                }
                for r in self._execution_history[-10:]  # Last 10 executions
            ]
        }

def format_function_call_result_for_model(result: ToolExecutionResult) -> str:
    """
    Format the tool execution result for inclusion in the model conversation.

    Args:
        result: ToolExecutionResult from function execution

    Returns:
        Formatted string for the model
    """
    if result.success:
        # Format successful result
        if isinstance(result.result, dict):
            if 'result' in result.result:
                return f"Tool {result.tool_name} executed successfully:\n{json.dumps(result.result['result'], indent=2)}"
            else:
                return f"Tool {result.tool_name} executed successfully:\n{json.dumps(result.result, indent=2)}"
        elif isinstance(result.result, str):
            return f"Tool {result.tool_name} executed successfully:\n{result.result}"
        else:
            return f"Tool {result.tool_name} executed successfully:\n{str(result.result)}"
    else:
        # Format error result
        return f"Tool {result.tool_name} failed: {result.error}"
