import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict

import requests

from config import LLM_CONFIG
from core.common.dataclass import QueryResult, Tool, ToolType
from core.common.tool_registry import ToolRegistry
from logger import Logger

logger = Logger.get_logger()


class QueryExecutor:
    """
    Executes queries using the specified tool and parameters.

    This class retrieves tools from the ToolRegistry and executes queries based on the tool type
    (database, API call, calculation, or general info). It formats the raw response into a
    human-readable format using an LLM service and returns a QueryResult object.

    Attributes:
        tool_registry (ToolRegistry): Registry of available tools for query execution.
        logger (Logger): Logger instance for recording execution activities.
    """

    def __init__(self):
        """
        Initialize the QueryExecutor with a ToolRegistry.

        Sets up the tool registry for accessing available tools during query execution.
        """
        self.tool_registry = ToolRegistry()
        self.logger = Logger.get_logger()
        self.logger.debug("QueryExecutor initialized with ToolRegistry")

    def execute_query(self, tool_name: str, parameters: Dict[str, Any], user_query: str) -> QueryResult:
        """
        Execute a query using the specified tool and parameters.

        Retrieves the tool from the ToolRegistry, executes the query based on the tool type,
        and formats the result into a human-readable response. Returns a QueryResult object
        with execution details.

        Args:
            tool_name (str): The name of the tool to use for execution.
            parameters (Dict[str, Any]): Parameters required for the tool execution.
            user_query (str): The original user query for context.

        Returns:
            QueryResult: An object containing:
                - success (bool): Whether the execution was successful.
                - tool_used (str): The name of the tool used.
                - raw_response (Any): The raw result from the tool execution.
                - formatted_response (str): Human-readable formatted response.
                - execution_time (float): Time taken to execute the query in seconds.
                - error_message (str, optional): Error message if execution failed.

        Raises:
            ValueError: If the specified tool is not found or invalid parameters are provided.
            Exception: For other execution errors, returns a failed QueryResult with error details.
        """
        start_time = datetime.now(timezone.utc)
        self.logger.info(f"Executing query with tool: {tool_name}, query: {user_query}")

        try:
            self.logger.debug(f"Retrieving tool: {tool_name}")
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                self.logger.error(f"Tool '{tool_name}' not found")
                raise ValueError(f"Tool '{tool_name}' not found")

            self.logger.debug(f"Tool type: {tool.type}")
            if tool.type == ToolType.DATABASE:
                result = self._execute_database_query(parameters, user_query)
            elif tool.type == ToolType.API_CALL:
                result = self._execute_api_call(tool, parameters)
            elif tool.type == ToolType.CALCULATION:
                result = self._execute_calculation(parameters, user_query)
            elif tool.type == ToolType.GENERAL_INFO:
                result = self._execute_general_info(user_query)
            else:
                self.logger.warning(f"Unsupported tool type: {tool.type.value}")
                result = f"Tool type {tool.type.value} requires file upload and should be called directly"

            self.logger.debug(f"Formatting response for tool: {tool_name}")
            formatted_response = self._format_response(result, user_query, tool_name)
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.info(
                f"Query executed successfully with tool: {tool_name}, execution time: {execution_time:.2f}s")

            return QueryResult(
                success=True,
                tool_used=tool_name,
                raw_response=result,
                formatted_response=formatted_response,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.error(f"Query execution failed for tool '{tool_name}', query '{user_query}': {str(e)}",
                              exc_info=True)
            return QueryResult(
                success=False,
                tool_used=tool_name,
                raw_response=None,
                formatted_response=f"Sorry, I encountered an error: {str(e)}",
                execution_time=execution_time,
                error_message=str(e)
            )

    def _execute_database_query(self, parameters: Dict[str, Any], user_query: str) -> Any:
        """
        Execute a database query using the provided SQL query.

        Connects to a SQLite database, executes the provided SQL query, and returns the results
        for SELECT queries or the affected row count for other queries.

        Args:
            parameters (Dict[str, Any]): Dictionary containing the SQL query in 'sql_query' key.
            user_query (str): The original user query for context.

        Returns:
            Any: List of dictionaries for SELECT queries, or a dictionary with affected row count
                 for other queries.

        Raises:
            ValueError: If no SQL query is provided in parameters.
            sqlite3.Error: If the database operation fails.
        """
        self.logger.debug(f"Executing database query for user query: {user_query}")
        sql_query = parameters.get('sql_query', '')
        if not sql_query:
            self.logger.error("No SQL query provided in parameters")
            raise ValueError("No SQL query provided")

        self.logger.debug(f"Connecting to SQLite database 'app_data.db'")
        conn = sqlite3.connect('app_data.db')
        cursor = conn.cursor()

        try:
            self.logger.debug(f"Executing SQL query: {sql_query}")
            cursor.execute(sql_query)
            if sql_query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                formatted_results = [dict(zip(columns, row)) for row in results]
                self.logger.debug(f"SELECT query returned {len(formatted_results)} rows")
                return formatted_results
            else:
                self.logger.debug(f"Non-SELECT queries are not allowed to be executed")
                return {"affected_rows": 0}

        finally:
            self.logger.debug("Closing database connection")
            conn.close()

    def _execute_api_call(self, tool: Tool, parameters: Dict[str, Any]) -> Any:
        """
        Execute an API call using the specified tool and parameters.

        Makes an HTTP request to the tool's endpoint with the provided API parameters.
        Currently supports a weather API as an example.

        Args:
            tool (Tool): The tool object containing the API endpoint.
            parameters (Dict[str, Any]): Dictionary containing API parameters in 'api_params' key.

        Returns:
            Any: The JSON response from the API call.

        Raises:
            ValueError: If the API call for the tool is not implemented.
            requests.RequestException: If the API call fails.
        """
        self.logger.debug(f"Executing API call for tool: {tool.name}")
        api_params = parameters.get('api_params', {})

        if tool.name == "weather_api":
            self.logger.debug(f"Preparing weather API call with params: {api_params}")
            url = tool.endpoint
            params = {
                'q': api_params.get('location', 'Dhaka'),
                'appid': 'your_api_key_here',  # Replace with actual API key
                'units': 'metric'
            }
            self.logger.debug(f"Sending GET request to {url}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()
            self.logger.debug(f"Weather API call successful, response: {json.dumps(result)[:100]}...")
            return result
        else:
            self.logger.error(f"API call for tool '{tool.name}' not implemented")
            raise ValueError(f"API call for {tool.name} not implemented")

    def _execute_calculation(self, parameters: Dict[str, Any], user_query: str) -> Any:
        """
        Execute a calculation query using the LLM service.

        Sends a prompt to the LLM service to perform mathematical calculations or data analysis
        based on the user query.

        Args:
            parameters (Dict[str, Any]): Parameters for the calculation (currently unused).
            user_query (str): The original user query for context.

        Returns:
            Any: The raw response from the LLM service.

        Raises:
            requests.RequestException: If the LLM service request fails.
        """
        self.logger.debug(f"Executing calculation for query: {user_query}")
        prompt = f"""
You are a mathematical calculator and data analyst. 

USER REQUEST: "{user_query}"

Perform the requested calculation or analysis and provide:
1. The calculation steps
2. The final result
3. Any relevant explanations

If it's a word problem, break it down step by step.
If it involves data analysis, explain your methodology.

Provide your response in a clear, structured format.
"""
        self.logger.debug(f"Generated calculation prompt: {prompt[:100]}...")

        payload = {
            "model": LLM_CONFIG["model"],
            "prompt": prompt,
            "stream": False,
            "options": LLM_CONFIG["options"],
        }

        self.logger.debug(f"Sending calculation request to LLM service")
        response = requests.post(
            LLM_CONFIG["base_url"],
            json=payload,
            timeout=LLM_CONFIG["timeout"]
        )
        response.raise_for_status()
        result = response.json().get('response', '')
        self.logger.debug(f"Calculation response received: {result[:100]}...")
        return result

    def _execute_general_info(self, user_query: str) -> Any:
        """
        Execute a general information query using the LLM service.

        Sends a prompt to the LLM service to answer general questions based on the user query.

        Args:
            user_query (str): The original user query.

        Returns:
            Any: The raw response from the LLM service.

        Raises:
            requests.RequestException: If the LLM service request fails.
        """
        self.logger.debug(f"Executing general info query: {user_query}")
        prompt = f"""
You are a helpful AI assistant. Answer the following question accurately and comprehensively:

QUESTION: "{user_query}"

Provide a clear, informative Ascertainable response that directly addresses the user's question.
If you're not certain about something, acknowledge the uncertainty.
Structure your response in a readable format with appropriate sections if needed.
"""
        self.logger.debug(f"Generated general info prompt: {prompt[:100]}...")

        payload = {
            "model": LLM_CONFIG["model"],
            "prompt": prompt,
            "stream": False,
            "options": LLM_CONFIG["options"],
        }

        self.logger.debug(f"Sending general info request to LLM service")
        response = requests.post(
            LLM_CONFIG["base_url"],
            json=payload,
            timeout=LLM_CONFIG["timeout"]
        )
        response.raise_for_status()
        result = response.json().get('response', '')
        self.logger.debug(f"General info response received: {result[:100]}...")
        return result

    def _format_response(self, raw_response: Any, user_query: str, tool_name: str) -> str:
        """
        Format a raw response into a human-readable format using the LLM service.

        Sends the raw response, user query, and tool information to the LLM service to generate
        a conversational, markdown-formatted response. Falls back to string conversion if formatting fails.

        Args:
            raw_response (Any): The raw result from the tool execution.
            user_query (str): The original user query for context.
            tool_name (str): The name of the tool used.

        Returns:
            str: The formatted human-readable response, or a stringified raw response if formatting fails.

        Raises:
            requests.RequestException: If the LLM service request fails.
        """
        self.logger.debug(f"Formatting response for query: {user_query}, tool: {tool_name}")
        try:
            prompt = f"""
You are a response formatter that converts raw data into human-readable responses.

USER'S ORIGINAL QUESTION: "{user_query}"
TOOL USED: {tool_name}
RAW RESPONSE: {json.dumps(raw_response, indent=2) if isinstance(raw_response, (dict, list)) else str(raw_response)}

Format this raw response into a clear, conversational answer that directly addresses the user's question.

FORMATTING GUIDELINES:
- Write in a natural, conversational tone
- Use markdown format
- Structure information clearly with appropriate headings or bullet points if needed
- Include all relevant information from the raw response
- Make numbers and data easy to understand
- If it's database results, present them in a readable table format
- If it's API data, extract and present the most relevant information
- Always end with a summary or conclusion if appropriate

Provide the formatted response:
"""
            self.logger.debug(f"Generated formatting prompt: {prompt[:100]}...")

            payload = {
                "model": LLM_CONFIG["model"],
                "prompt": prompt,
                "stream": False,
                "options": LLM_CONFIG["options"],
            }

            self.logger.debug(f"Sending formatting request to LLM service")
            response = requests.post(
                LLM_CONFIG["base_url"],
                json=payload,
                timeout=LLM_CONFIG["timeout"]
            )
            response.raise_for_status()
            formatted_response = response.json().get('response', str(raw_response))
            self.logger.debug(f"Formatted response received: {formatted_response[:100]}...")
            return formatted_response

        except Exception as e:
            self.logger.error(f"Response formatting failed for query '{user_query}', tool '{tool_name}': {str(e)}",
                              exc_info=True)
            if isinstance(raw_response, (dict, list)):
                formatted_response = json.dumps(raw_response, indent=2)
            else:
                formatted_response = str(raw_response)
            self.logger.debug(f"Fallback to stringified response: {formatted_response[:100]}...")
            return formatted_response
