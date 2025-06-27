import json
import re
from typing import Any, Dict

import requests

from config import LLM_CONFIG
from core.common.tool_registry import ToolRegistry
from logger import Logger

logger = Logger.get_logger()


class QueryAnalyzer:
    """
    Analyzes user queries to determine the appropriate tool and parameters for processing.

    This class interacts with a Vision-Language Model (VLM) service to analyze user queries,
    identify the best tool from the ToolRegistry, and extract necessary parameters. It handles
    communication with the VLM service, processes responses, and provides fallback behavior
    in case of errors.

    Attributes:
        tool_registry (ToolRegistry): Registry of available tools for query processing.
        vlm_client (bool): Indicates whether the VLM service is available.
        logger (Logger): Logger instance for recording analysis activities.
    """

    def __init__(self):
        """
        Initialize the QueryAnalyzer with a ToolRegistry and VLM client.

        Sets up the tool registry and checks the availability of the VLM service for query analysis.
        """
        self.tool_registry = ToolRegistry()
        self.logger = Logger.get_logger()
        self.logger.debug("Initializing QueryAnalyzer with ToolRegistry")
        self.vlm_client = self._initialize_vlm_client()
        self.logger.info(f"QueryAnalyzer initialized, VLM client available: {self.vlm_client}")

    def _initialize_vlm_client(self) -> bool:
        """
        Initialize the VLM client by checking the service's availability.

        Attempts to connect to the VLM service's health endpoint to confirm it is operational.

        Returns:
            bool: True if the VLM service is available, False otherwise.
        """
        self.logger.debug("Attempting to initialize VLM client")
        try:
            health_url = LLM_CONFIG["base_url"].replace('/api/generate', '/api/tags')
            self.logger.debug(f"Checking VLM service health at {health_url}")
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
            self.logger.info("VLM service health check successful")
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"VLM service initialization failed: {str(e)}", exc_info=True)
            return False

    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze a user query to determine the best tool and parameters.

        Sends the query to the VLM service with a formatted prompt, processes the response,
        and returns a dictionary with the tool name, parameters, and reasoning. Falls back
        to a default tool ('general_info') if analysis fails.

        Args:
            user_query (str): The raw query string provided by the user.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - tool_name (str): The name of the selected tool.
                - parameters (dict): Parameters required for the tool.
                - reasoning (str): Explanation of the tool selection.
                - sql_query (str, optional): SQL query if the tool is database-related.
                - api_params (dict, optional): Parameters for API-related tools.

        Raises:
            Exception: If query analysis fails, logs the error and returns a default response.
        """
        self.logger.info(f"Analyzing user query: {user_query}")
        try:
            self.logger.debug("Fetching tools description")
            tools_info = self._get_tools_description()
            self.logger.debug("Generating analysis prompt")
            prompt = self._generate_analysis_prompt(user_query, tools_info)

            if not self.vlm_client:
                self.logger.warning("VLM client unavailable, returning default response")
                return {
                    "tool_name": "general_info",
                    "parameters": {},
                    "reasoning": "VLM service unavailable, defaulting to general info"
                }

            self.logger.debug(f"Sending request to VLM service with payload: {prompt[:100]}...")
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
            self.logger.debug("Received response from VLM service")

            vlm_response = response.json().get('response', '')
            self.logger.debug(f"VLM response: {vlm_response[:100]}...")
            parsed_response = self._parse_analysis_response(vlm_response)
            self.logger.info(
                f"Query analysis completed: tool_name={parsed_response['tool_name']}, reasoning={parsed_response['reasoning']}")
            return parsed_response

        except Exception as e:
            self.logger.error(f"Query analysis failed for query '{user_query}': {str(e)}", exc_info=True)
            return {
                "tool_name": "general_info",
                "parameters": {},
                "reasoning": f"Analysis failed, defaulting to general info: {str(e)}"
            }

    def _get_tools_description(self) -> str:
        """
        Generate a formatted description of all available tools.

        Retrieves tool information from the ToolRegistry and formats it as a string,
        including tool names, descriptions, and tags (if available).

        Returns:
            str: A newline-separated string of tool descriptions.
        """
        self.logger.debug("Generating tools description")
        tools_desc = []
        for tool in self.tool_registry.get_all_tools():
            self.logger.debug(f"Processing tool: {tool.name}")
            if hasattr(tool, 'tags') and tool.tags:
                tags_str = ", ".join(sorted(tool.tags))
                tools_desc.append(f"- {tool.name}: {tool.description} - [Related tags: {tags_str}]")
            else:
                tools_desc.append(f"- {tool.name}: {tool.description}")
        description = "\n".join(tools_desc)
        self.logger.debug(f"Tools description generated: {description[:100]}...")
        return description

    def _generate_analysis_prompt(self, user_query: str, tools_info: str) -> str:
        """
        Generate a prompt for the VLM to analyze the user query.

        Constructs a prompt that includes the available tools, the user query, and
        instructions for the VLM to select a tool and provide parameters.

        Args:
            user_query (str): The raw query string provided by the user.
            tools_info (str): Formatted description of available tools.

        Returns:
            str: The formatted prompt string for the VLM.
        """
        self.logger.debug(f"Generating analysis prompt for query: {user_query}")
        prompt = f"""
You are an intelligent query analyzer that determines which tool should handle a user's request.

AVAILABLE TOOLS:
{tools_info}

USER QUERY: "{user_query}"

Analyze the user's query and determine:
1. Which tool is most appropriate
2. What parameters are needed
3. Your reasoning

RESPONSE FORMAT (JSON):
{{
    "tool_name": "exact_tool_name_from_list",
    "parameters": {{
        "key": "value for any specific parameters needed"
    }},
    "reasoning": "Why you chose this tool and what the user is trying to accomplish",
    "sql_query": "SELECT * FROM table WHERE condition" (only if tool_name is related databases),
    "api_params": {{"param": "value"}} (only if tool_name involves API calls)
}}

ANALYSIS RULES:
- For database queries: Include the SQL query in sql_query field
- For API calls: Include required parameters in api_params field
- For calculations: Look for mathematical operations or data analysis requests
- For general questions: Use general_info tool
- Be specific about what parameters are needed

Analyze the query and provide the JSON response:
"""
        self.logger.debug(f"Analysis prompt generated: {prompt[:100]}...")
        return prompt

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the VLM response to extract tool selection and parameters.

        Processes the VLM response to extract a JSON object containing the tool name,
        parameters, and reasoning. Falls back to a default response if parsing fails.

        Args:
            response (str): The raw response from the VLM service.

        Returns:
            Dict[str, Any]: A dictionary containing the parsed tool name, parameters,
                           and reasoning, or a default response if parsing fails.
        """
        self.logger.debug("Parsing VLM response")
        try:
            response = response.strip()
            json_match = re.search(r'\{.*}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                self.logger.debug(f"Extracted JSON string: {json_str[:100]}...")
                parsed_response = json.loads(json_str)
                self.logger.debug(f"Parsed response: {parsed_response}")
                return parsed_response
            else:
                self.logger.warning("No JSON found in VLM response")
                return {
                    "tool_name": "general_info",
                    "parameters": {},
                    "reasoning": "Could not parse analysis response"
                }
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse VLM response: {str(e)}", exc_info=True)
            return {
                "tool_name": "general_info",
                "parameters": {},
                "reasoning": f"JSON parsing failed: {str(e)}"
            }
