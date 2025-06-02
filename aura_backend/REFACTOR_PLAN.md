#!/bin/bash
# Clean Startup Script Refactor Plan

## Current Issues:
# 1. Multiple redundant start scripts causing confusion
# 2. Aura Internal Server (aura_server.py) failing to start properly
# 3. MCP Client integration not connecting to servers
# 4. Architecture confusion between Aura core and MCP integration

## Recommended Actions:

### 1. Remove Redundant Scripts
# Keep only:
# - start.sh (main entry point)
# - start_api.sh (for API-only development)
# - start_frontend.sh (for frontend-only development)

### 2. Fix Architecture Issues
# The problem: aura_server.py is trying to run as a FastMCP server but it's blocking
# Solution: Don't run it as a separate process - integrate it directly into main.py

### 3. Proper Service Organization
# - main.py: Core API + integrated Aura capabilities + MCP client
# - No separate "Aura Internal Server" process needed
# - MCP client connects to external servers only

## To implement:
# 1. Modify main.py to include Aura's MCP tools directly
# 2. Remove the need for separate aura_server.py process
# 3. Simplify startup to just API + Frontend
