"""
Smart MCP Parameter Handler
==========================

This module provides intelligent parameter formatting for MCP tools by analyzing
their schemas instead of hardcoding server names. It learns from tool schemas
and caches the results for optimal performance.
"""

import logging
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ParameterFormat:
    """Represents how a tool expects its parameters"""
    format_type: str  # 'direct', 'wrapped', 'fastmcp'
    detected_at: datetime
    schema_hash: str
    confidence: float

class SmartMCPParameterHandler:
    """
    Intelligently handles parameter formatting for MCP tools by analyzing
    their schemas and learning from successful calls.
    """
    
    def __init__(self):
        # Cache for learned parameter formats
        # Key: (server_name, tool_name) -> ParameterFormat
        self._format_cache: Dict[Tuple[str, str], ParameterFormat] = {}
        
        # Success tracking for format validation
        # Key: (server_name, tool_name, format_type) -> (success_count, failure_count)
        self._format_success_tracking: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    
    def format_parameters(
        self,
        tool_name: str,
        server_name: str,
        arguments: Dict[str, Any],
        tool_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format parameters according to the tool's expected format by analyzing its schema.
        
        Args:
            tool_name: Name of the MCP tool
            server_name: Name of the MCP server
            arguments: Raw arguments from Gemini
            tool_schema: Tool schema containing parameter definitions
            
        Returns:
            Properly formatted parameters for the tool
        """
        try:
            # Check cache first
            cache_key = (server_name, tool_name)
            if cache_key in self._format_cache:
                cached_format = self._format_cache[cache_key]
                logger.debug(f"📋 Using cached format '{cached_format.format_type}' for {tool_name}")
                return self._apply_format(arguments, cached_format.format_type)
            
            # No cache hit - analyze the schema
            if tool_schema:
                format_type = self._detect_format_from_schema(tool_schema, tool_name)
                logger.info(f"🔍 Detected format '{format_type}' for {tool_name} from schema")
                
                # Cache the detection
                self._cache_format(server_name, tool_name, format_type, tool_schema)
                
                return self._apply_format(arguments, format_type)
            else:
                # No schema available - use heuristics
                format_type = self._detect_format_from_heuristics(
                    tool_name, server_name, arguments
                )
                logger.info(f"🎯 Using heuristic format '{format_type}' for {tool_name}")
                return self._apply_format(arguments, format_type)
                
        except Exception as e:
            logger.error(f"❌ Error formatting parameters for {tool_name}: {e}")
            # Return original arguments as fallback
            return arguments
    
    def _detect_format_from_schema(
        self,
        tool_schema: Dict[str, Any],
        tool_name: str
    ) -> str:
        """
        Detect parameter format from tool schema.
        
        Returns: 'direct', 'wrapped', or 'fastmcp'
        """
        # Get the input schema or parameters section
        input_schema = tool_schema.get('inputSchema') or tool_schema.get('parameters', {})
        properties = input_schema.get('properties', {})
        
        # Check if the tool expects no parameters
        if not properties:
            logger.debug(f"Tool {tool_name} expects no parameters")
            return 'direct'
        
        # Check if there's a single 'params' property that contains all parameters
        if len(properties) == 1 and 'params' in properties:
            params_prop = properties['params']
            
            # Check if params is an object with its own properties
            if params_prop.get('type') == 'object' and 'properties' in params_prop:
                logger.debug(f"Tool {tool_name} expects FastMCP-style wrapped params")
                return 'fastmcp'
            
            # Check if params is a simple wrapper
            if params_prop.get('type') in ['object', 'string']:
                logger.debug(f"Tool {tool_name} expects wrapped params")
                return 'wrapped'
        
        # Check for $defs pattern (common in FastMCP tools)
        if '$defs' in input_schema or '$ref' in str(input_schema):
            logger.debug(f"Tool {tool_name} uses $defs pattern - likely FastMCP")
            return 'fastmcp'
        
        # Check if all properties are direct parameters
        # This is the most common case for simple tools
        if all(
            isinstance(prop, dict) and 'type' in prop
            for prop in properties.values()
        ):
            logger.debug(f"Tool {tool_name} expects direct parameters")
            return 'direct'
        
        # Default to direct if unsure
        logger.debug(f"Tool {tool_name} format unclear, defaulting to direct")
        return 'direct'
    
    def _detect_format_from_heuristics(
        self,
        tool_name: str,
        server_name: str,
        arguments: Dict[str, Any]
    ) -> str:
        """
        Use heuristics when schema is not available.
        """
        # Check if tool name suggests a certain pattern
        tool_lower = tool_name.lower()
        
        # Common FastMCP tool patterns
        if any(pattern in tool_lower for pattern in ['brave_', 'web_', 'search']):
            return 'fastmcp'
        
        # Common direct parameter tools
        if any(pattern in tool_lower for pattern in ['read_', 'write_', 'execute_', 'list_']):
            return 'direct'
        
        # Check server implementation hints
        if 'mcp-server' in server_name.lower() or 'fastmcp' in server_name.lower():
            return 'fastmcp'
        
        # JavaScript/TypeScript servers often use wrapped params
        if any(hint in server_name.lower() for hint in ['npx', 'node', 'js', 'ts']):
            return 'wrapped'
        
        # Default to direct for Python servers
        return 'direct'
    
    def _apply_format(self, arguments: Dict[str, Any], format_type: str) -> Dict[str, Any]:
        """
        Apply the detected format to the arguments.
        """
        # Handle empty arguments
        if not arguments:
            return {}
        
        # Check if arguments are already in the expected format
        if format_type in ['wrapped', 'fastmcp'] and 'params' in arguments and len(arguments) == 1:
            # Already wrapped correctly
            return arguments
        
        # Apply formatting based on type
        if format_type == 'direct':
            # Unwrap if necessary
            if 'params' in arguments and len(arguments) == 1:
                params_value = arguments['params']
                if isinstance(params_value, dict):
                    return params_value
                elif isinstance(params_value, str):
                    try:
                        return json.loads(params_value)
                    except json.JSONDecodeError:
                        pass
            return arguments
            
        elif format_type == 'fastmcp':
            # Wrap in params for FastMCP
            if 'params' not in arguments:
                return {'params': arguments}
            return arguments
            
        elif format_type == 'wrapped':
            # Simple wrapper
            if 'params' not in arguments:
                return {'params': arguments}
            return arguments
        
        # Default to returning as-is
        return arguments
    
    def _cache_format(
        self,
        server_name: str,
        tool_name: str,
        format_type: str,
        tool_schema: Dict[str, Any]
    ):
        """Cache the detected format for future use."""
        cache_key = (server_name, tool_name)
        
        # Create a simple hash of the schema for validation
        schema_str = json.dumps(tool_schema, sort_keys=True)
        schema_hash = str(hash(schema_str))
        
        self._format_cache[cache_key] = ParameterFormat(
            format_type=format_type,
            detected_at=datetime.now(),
            schema_hash=schema_hash,
            confidence=1.0  # High confidence from schema analysis
        )
    
    def record_success(
        self,
        server_name: str,
        tool_name: str,
        format_type: str,
        success: bool
    ):
        """
        Record whether a format worked for a tool.
        This helps improve format detection over time.
        """
        tracking_key = (server_name, tool_name, format_type)
        
        if tracking_key not in self._format_success_tracking:
            self._format_success_tracking[tracking_key] = (0, 0)
        
        success_count, failure_count = self._format_success_tracking[tracking_key]
        
        if success:
            self._format_success_tracking[tracking_key] = (success_count + 1, failure_count)
            logger.debug(f"✅ Format '{format_type}' succeeded for {tool_name}")
        else:
            self._format_success_tracking[tracking_key] = (success_count, failure_count + 1)
            logger.debug(f"❌ Format '{format_type}' failed for {tool_name}")
            
            # If this format consistently fails, remove it from cache
            if failure_count >= 2 and success_count == 0:
                cache_key = (server_name, tool_name)
                if cache_key in self._format_cache:
                    del self._format_cache[cache_key]
                    logger.warning(f"🔄 Removed cached format for {tool_name} due to failures")
    
    def get_format_stats(self) -> Dict[str, Any]:
        """Get statistics about format detection and success rates."""
        stats = {
            'cached_formats': len(self._format_cache),
            'format_distribution': {},
            'success_rates': {}
        }
        
        # Count format types
        for format_info in self._format_cache.values():
            format_type = format_info.format_type
            stats['format_distribution'][format_type] = \
                stats['format_distribution'].get(format_type, 0) + 1
        
        # Calculate success rates
        for (server, tool, fmt), (success, failure) in self._format_success_tracking.items():
            key = f"{server}.{tool}.{fmt}"
            total = success + failure
            if total > 0:
                stats['success_rates'][key] = {
                    'success': success,
                    'failure': failure,
                    'rate': success / total
                }
        
        return stats

# Global instance for reuse
_smart_handler_instance = None

def get_smart_parameter_handler() -> SmartMCPParameterHandler:
    """Get or create the global smart parameter handler instance."""
    global _smart_handler_instance
    if _smart_handler_instance is None:
        _smart_handler_instance = SmartMCPParameterHandler()
    return _smart_handler_instance


# Example usage and tests
if __name__ == "__main__":
    # Test the smart parameter handler
    handler = SmartMCPParameterHandler()
    
    # Test schemas that would indicate different formats
    test_schemas = [
        # Direct parameters schema
        {
            "name": "read_file",
            "schema": {
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "encoding": {"type": "string"}
                    },
                    "required": ["path"]
                }
            },
            "expected": "direct"
        },
        # FastMCP wrapped params schema
        {
            "name": "brave_web_search",
            "schema": {
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "params": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "count": {"type": "number"}
                            },
                            "required": ["query"]
                        }
                    },
                    "required": ["params"]
                }
            },
            "expected": "fastmcp"
        },
        # No parameters
        {
            "name": "list_sessions",
            "schema": {
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            "expected": "direct"
        }
    ]
    
    print("🧪 Testing Smart Parameter Handler\n")
    print("=" * 60)
    
    for test in test_schemas:
        detected = handler._detect_format_from_schema(
            test["schema"]["inputSchema"],
            test["name"]
        )
        
        status = "✅" if detected == test["expected"] else "❌"
        print(f"{status} {test['name']}: detected '{detected}', expected '{test['expected']}'")
    
    print("\n" + "=" * 60)
    print("✅ Smart parameter detection ready!")
