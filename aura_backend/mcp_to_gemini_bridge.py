"""
MCP to Gemini Function Calling Bridge
===================================

This module bridges MCP tools with Google Gemini's function calling system.
It converts MCP tool schemas to Gemini-compatible function definitions and
handles the execution flow between Gemini and MCP servers.
"""

import logging
import asyncio
import os
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

# Import smart parameter handler
try:
    from smart_mcp_parameter_handler import get_smart_parameter_handler
    smart_parameter_handler = get_smart_parameter_handler()
except ImportError:
    smart_parameter_handler = None
    logger.warning("Smart parameter handler not available")

# Configuration for addressing Gemini 2.5 tool calling issues
TOOL_CALL_MAX_RETRIES = int(os.getenv('TOOL_CALL_MAX_RETRIES', '3'))
TOOL_CALL_RETRY_DELAY = float(os.getenv('TOOL_CALL_RETRY_DELAY', '1.0'))
TOOL_CALL_EXPONENTIAL_BACKOFF = os.getenv('TOOL_CALL_EXPONENTIAL_BACKOFF', 'true').lower() == 'true'
TOOL_CALL_SEQUENTIAL_MODE = os.getenv('TOOL_CALL_SEQUENTIAL_MODE', 'false').lower() == 'true'
TOOL_CALL_TIMEOUT = float(os.getenv('TOOL_CALL_TIMEOUT', '30.0'))

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
        Execute a Gemini function call with retry logic to handle known Gemini 2.5 tool calling issues.
        
        This method implements comprehensive error handling for the documented Gemini 2.5 problem
        where tool calls randomly fail or get cut off with no error messages.

        Args:
            function_call: Gemini function call object
            user_id: User ID for context

        Returns:
            ToolExecutionResult with execution outcome
        """
        # Use the enhanced execution method with retry logic
        return await self._execute_function_call_with_retry(function_call, user_id)

    async def _execute_function_call_with_retry(
        self,
        function_call: types.FunctionCall,
        user_id: str
    ) -> ToolExecutionResult:
        """
        Enhanced function call execution with retry logic for Gemini 2.5 stability issues.
        
        Implements exponential backoff and multiple retry strategies to handle:
        - Random tool call failures
        - Response cutoffs with no error messages  
        - Concurrent tool execution issues
        """
        start_time = datetime.now()
        last_error = None
        
        for attempt in range(TOOL_CALL_MAX_RETRIES + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    # Calculate retry delay with exponential backoff
                    if TOOL_CALL_EXPONENTIAL_BACKOFF:
                        delay = TOOL_CALL_RETRY_DELAY * (2 ** (attempt - 1))
                    else:
                        delay = TOOL_CALL_RETRY_DELAY
                    
                    logger.info(f"🔄 Retry attempt {attempt}/{TOOL_CALL_MAX_RETRIES} for {function_call.name} after {delay:.1f}s")
                    await asyncio.sleep(delay)
                
                # Execute with timeout to handle hanging calls
                try:
                    result = await asyncio.wait_for(
                        self._execute_single_function_call(function_call, user_id),
                        timeout=TOOL_CALL_TIMEOUT
                    )
                    
                    if result.success:
                        if attempt > 0:
                            logger.info(f"✅ Tool {function_call.name} succeeded on retry attempt {attempt}")
                        return result
                    else:
                        last_error = result.error
                        logger.warning(f"⚠️ Tool {function_call.name} failed on attempt {attempt + 1}: {result.error}")
                        
                except asyncio.TimeoutError:
                    last_error = f"Tool execution timed out after {TOOL_CALL_TIMEOUT}s"
                    logger.error(f"⏱️ Tool {function_call.name} timed out on attempt {attempt + 1}")
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"❌ Unexpected error on attempt {attempt + 1} for {function_call.name}: {e}")
        
        # All retries exhausted
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"💥 Tool {function_call.name} failed after {TOOL_CALL_MAX_RETRIES + 1} attempts")
        
        return ToolExecutionResult(
            tool_name=function_call.name or "",
            success=False,
            result=None,
            error=f"Failed after {TOOL_CALL_MAX_RETRIES + 1} attempts. Last error: {last_error}",
            execution_time=execution_time
        )

    async def _execute_single_function_call(
        self,
        function_call: types.FunctionCall,
        user_id: str
    ) -> ToolExecutionResult:
        """
        Execute a single function call attempt (original logic extracted for retry wrapper).
        """
        start_time = datetime.now()
        server = None
        mcp_tool_name = None
        formatted_arguments = None
        detected_format = 'direct'

        try:
            function_name = function_call.name
            arguments = dict(function_call.args) if function_call.args else {}

            logger.debug(f"🔧 Executing function call: {function_name}")
            logger.debug(f"📥 Raw arguments from Gemini: {arguments}")

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
            original_tool = tool_info['original_tool']

            logger.debug(f"🎯 Mapped to MCP tool: {mcp_tool_name} on server: {server}")

            # Add user_id to arguments if not present and the tool expects it
            if 'user_id' not in arguments:
                parameters = original_tool.get('parameters', {})
                properties = parameters.get('properties', {})
                if 'user_id' in properties:
                    arguments['user_id'] = user_id
                    logger.debug(f"📝 Added user_id to arguments: {user_id}")

            # Use smart parameter handler if available
            formatted_arguments = arguments

            if smart_parameter_handler:
                tool_input_schema = original_tool.get('input_schema') or original_tool.get('parameters', {})

                formatted_arguments = smart_parameter_handler.format_parameters(
                    tool_name=mcp_tool_name,
                    server_name=server,
                    arguments=arguments,
                    tool_schema=tool_input_schema
                )

                # Determine what format was applied
                if 'params' in formatted_arguments and len(formatted_arguments) == 1:
                    detected_format = 'wrapped_or_fastmcp'
                else:
                    detected_format = 'direct'

                logger.debug(f"📤 Smart handler applied format '{detected_format}' for {mcp_tool_name}")
            else:
                logger.debug("⚠️ Smart parameter handler not available, using raw arguments")

            # Execute the MCP tool
            result = None
            if server == 'aura-internal' and self.aura_internal_tools:
                logger.debug(f"🏠 Executing internal tool: {mcp_tool_name}")
                result = await self.aura_internal_tools.execute_tool(mcp_tool_name, formatted_arguments)
            else:
                logger.debug(f"🌐 Executing external MCP tool: {mcp_tool_name}")

                if hasattr(self.mcp_client_manager, 'call_tool'):
                    result = await self.mcp_client_manager.call_tool(mcp_tool_name, formatted_arguments)
                else:
                    raise ValueError(f"Cannot execute external tool {mcp_tool_name}: MCP client not properly configured")

            execution_time = (datetime.now() - start_time).total_seconds()

            logger.debug(f"✅ Tool {function_name} executed successfully in {execution_time:.2f}s")
            logger.debug(f"📋 Result type: {type(result)}")

            execution_result = ToolExecutionResult(
                tool_name=function_name,
                success=True,
                result=result,
                execution_time=execution_time
            )

            self._execution_history.append(execution_result)

            # Record success for format learning
            if smart_parameter_handler:
                smart_parameter_handler.record_success(
                    server_name=server,
                    tool_name=mcp_tool_name,
                    format_type=detected_format,
                    success=True
                )

            return execution_result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)

            logger.error(f"❌ Failed to execute {function_call.name}: {error_msg}")
            logger.debug(f"🔧 Function: {function_call.name}")
            logger.debug(f"📥 Raw args: {dict(function_call.args) if function_call.args else {}}")
            if formatted_arguments is not None:
                logger.debug(f"📤 Formatted args: {formatted_arguments}")
            logger.debug(f"🎯 MCP tool: {mcp_tool_name}")
            logger.debug(f"🖥️ Server: {server}")

            execution_result = ToolExecutionResult(
                tool_name=function_call.name or "",
                success=False,
                result=None,
                error=error_msg,
                execution_time=execution_time
            )

            self._execution_history.append(execution_result)

            # Record failure for format learning
            if smart_parameter_handler and server and mcp_tool_name:
                smart_parameter_handler.record_success(
                    server_name=server,
                    tool_name=mcp_tool_name,
                    format_type=detected_format,
                    success=False
                )

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

        stats = {
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

        # Add parameter handling stats if available
        if smart_parameter_handler:
            stats['parameter_handling'] = smart_parameter_handler.get_format_stats()

        return stats

def format_function_call_result_for_model(result: ToolExecutionResult) -> str:
    """
    Format the tool execution result for inclusion in the model conversation.

    Handles various result formats including complex nested structures from tools like brave_search.

    Args:
        result: ToolExecutionResult from function execution

    Returns:
        Formatted string for the model
    """
    if result.success:
        # Handle different result types
        if isinstance(result.result, dict):
            # Handle MCP result wrapper format
            if 'content' in result.result and len(result.result) == 1:
                # MCP tools often wrap results in {'content': [{'type': 'text', 'text': '...'}]}
                content = result.result['content']
                if isinstance(content, list) and len(content) > 0:
                    first_content = content[0]
                    if isinstance(first_content, dict) and 'text' in first_content:
                        actual_result = first_content['text']
                        logger.debug(f"📝 Extracted text from MCP content wrapper: {actual_result[:100]}...")
                        return f"Tool {result.tool_name} executed successfully:\n{actual_result}"

            # Special handling for brave_search results
            if result.tool_name and 'brave' in result.tool_name.lower():
                return _format_brave_search_result(result.tool_name, result.result)

            # Check if result is wrapped in a 'result' key
            if 'result' in result.result and len(result.result) == 1:
                actual_result = result.result['result']
            else:
                actual_result = result.result

            # Format based on content
            if isinstance(actual_result, (list, dict)) and actual_result:
                # Pretty print complex structures
                try:
                    formatted = json.dumps(actual_result, indent=2, ensure_ascii=False)
                    return f"Tool {result.tool_name} executed successfully:\n{formatted}"
                except (TypeError, ValueError):
                    # Fallback for non-JSON serializable objects
                    return f"Tool {result.tool_name} executed successfully:\n{str(actual_result)}"
            else:
                # Simple result
                return f"Tool {result.tool_name} executed successfully: {actual_result}"

        elif isinstance(result.result, str):
            # String results
            if result.result.strip():
                return f"Tool {result.tool_name} executed successfully:\n{result.result}"
            else:
                return f"Tool {result.tool_name} executed successfully (empty result)"

        elif isinstance(result.result, list):
            # List results
            if result.result:
                try:
                    formatted = json.dumps(result.result, indent=2, ensure_ascii=False)
                    return f"Tool {result.tool_name} executed successfully:\n{formatted}"
                except (TypeError, ValueError):
                    return f"Tool {result.tool_name} executed successfully:\n{str(result.result)}"
            else:
                return f"Tool {result.tool_name} executed successfully (empty list)"

        else:
            # Other types
            return f"Tool {result.tool_name} executed successfully:\n{str(result.result)}"
    else:
        # Format error result with more context
        error_msg = f"Tool {result.tool_name} failed: {result.error}"
        if result.execution_time:
            error_msg += f" (after {result.execution_time:.2f}s)"
        return error_msg

def _format_brave_search_result(tool_name: str, result: Dict[str, Any]) -> str:
    """
    Special formatting for brave_search results to make them more readable.

    Args:
        tool_name: Name of the brave search tool
        result: Raw result from brave search

    Returns:
        Formatted string for the model
    """
    try:
        # Check if result has the expected structure
        if isinstance(result, dict):
            # Handle wrapped result
            if 'result' in result and isinstance(result['result'], dict):
                search_data = result['result']
            else:
                search_data = result

            # Extract web results if available
            web_results = search_data.get('web', {}).get('results', [])

            if web_results:
                formatted_results = [f"Tool {tool_name} found {len(web_results)} results:\n"]

                for i, item in enumerate(web_results[:5], 1):  # Limit to first 5 results
                    title = item.get('title', 'No title')
                    url = item.get('url', 'No URL')
                    description = item.get('description', 'No description')

                    formatted_results.append(f"{i}. **{title}**")
                    formatted_results.append(f"   URL: {url}")
                    formatted_results.append(f"   {description}\n")

                if len(web_results) > 5:
                    formatted_results.append(f"... and {len(web_results) - 5} more results")

                return "\n".join(formatted_results)

            # Handle other result types (news, images, etc.)
            elif any(key in search_data for key in ['news', 'images', 'videos']):
                # Format other types of results
                formatted_parts = [f"Tool {tool_name} results:\n"]

                for key in ['news', 'images', 'videos']:
                    if key in search_data and search_data[key].get('results'):
                        results = search_data[key]['results']
                        formatted_parts.append(f"\n{key.capitalize()}: {len(results)} items found")

                return "\n".join(formatted_parts)

            # Fallback to generic formatting
            else:
                return f"Tool {tool_name} executed successfully:\n{json.dumps(search_data, indent=2)}"

        # Fallback for unexpected format
        return f"Tool {tool_name} executed successfully:\n{str(result)}"

    except Exception as e:
        logger.warning(f"⚠️ Error formatting brave_search result: {e}")
        # Fallback to simple JSON dump
        try:
            return f"Tool {tool_name} executed successfully:\n{json.dumps(result, indent=2)}"
        except (TypeError, ValueError) as e:
            return f"Tool {tool_name} executed successfully:\n{str(result)}"
