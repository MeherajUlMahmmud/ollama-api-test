import json
import re
from typing import Any, Dict

import requests

from config import LLM_CONFIG
from core.common.tool_registry import ToolRegistry
from logger import Logger

logger = Logger.get_logger()


class QueryAnalyzer:
    """Analyzes user queries and determines appropriate tools"""

    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.vlm_client = self._initialize_vlm_client()

    def _initialize_vlm_client(self):
        """Initialize VLM client for query analysis"""
        try:
            health_url = LLM_CONFIG["base_url"].replace(
                '/api/generate', '/api/tags')
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
            logger.info("VLM service is available for query analysis")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"VLM service unavailable: {str(e)}")
            return False

    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """Analyze user query and determine the best tool to use"""
        try:
            tools_info = self._get_tools_description()
            prompt = self._generate_analysis_prompt(user_query, tools_info)

            payload = {
                "model": LLM_CONFIG["model"],
                "prompt": prompt,
                "stream": False,
                "options": LLM_CONFIG["options"],
            }

            response = requests.post(
                LLM_CONFIG["base_url"],
                json=payload,
                timeout=LLM_CONFIG["timeout"]
            )
            response.raise_for_status()

            vlm_response = response.json().get('response', '')
            return self._parse_analysis_response(vlm_response)

        except Exception as e:
            logger.error(f"Query analysis failed: {str(e)}")
            return {
                "tool_name": "general_info",
                "confidence": 0.3,
                "parameters": {},
                "reasoning": f"Analysis failed, defaulting to general info: {str(e)}"
            }

    def _get_tools_description(self) -> str:
        """Get formatted description of all available tools"""
        tools_desc = []
        for tool in self.tool_registry.get_all_tools():
            tools_desc.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tools_desc)

    def _generate_analysis_prompt(self, user_query: str, tools_info: str) -> str:
        """Generate prompt for query analysis"""
        return f"""
You are an intelligent query analyzer that determines which tool should handle a user's request.

AVAILABLE TOOLS:
{tools_info}

USER QUERY: "{user_query}"

Analyze the user's query and determine:
1. Which tool is most appropriate
2. What parameters are needed
3. How confident you are in this choice
4. Your reasoning

RESPONSE FORMAT (JSON):
{{
    "tool_name": "exact_tool_name_from_list",
    "confidence": 0.95,
    "parameters": {{
        "key": "value for any specific parameters needed"
    }},
    "reasoning": "Why you chose this tool and what the user is trying to accomplish",
    "sql_query": "SELECT * FROM table WHERE condition" (only if tool_name is database_query),
    "api_params": {{"param": "value"}} (only if tool_name involves API calls)
}}

ANALYSIS RULES:
- For database queries: Include the SQL query in sql_query field
- For API calls: Include required parameters in api_params field
- For NID/liveness: Check if user mentions images or files
- For calculations: Look for mathematical operations or data analysis requests
- For general questions: Use general_info tool
- Always provide confidence between 0.0 and 1.0
- Be specific about what parameters are needed

Analyze the query and provide the JSON response:
"""

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse the VLM response for query analysis"""
        try:
            response = response.strip()
            json_match = re.search(r'\{.*}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                return json.loads(json_str)
            else:
                return {
                    "tool_name": "general_info",
                    "confidence": 0.3,
                    "parameters": {},
                    "reasoning": "Could not parse analysis response"
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis response: {e}")
            return {
                "tool_name": "general_info",
                "confidence": 0.3,
                "parameters": {},
                "reasoning": f"JSON parsing failed: {str(e)}"
            }
