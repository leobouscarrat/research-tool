#!/bin/bash
cd "$(dirname "$0")"
exec uv run python mcp_server.py 2>>/tmp/mcp_research_tool.log