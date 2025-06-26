from datetime import datetime
import json
import re
import sqlite3
from typing import Any, Dict

import requests

from config import LLM_CONFIG
from core.common.dataclass import QueryResult, Tool, ToolType
from core.common.tool_registry import ToolRegistry
from logger import Logger

logger = Logger.get_logger()


class QueryExecutor:
    """Executes queries using the determined tool"""

    def __init__(self):
        self.tool_registry = ToolRegistry()

    def execute_query(self, tool_name: str, parameters: Dict[str, Any], user_query: str) -> QueryResult:
        """Execute query using the specified tool"""
        start_time = datetime.now()

        try:
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                raise ValueError(f"Tool '{tool_name}' not found")

            if tool.type == ToolType.DATABASE:
                result = self._execute_database_query(parameters, user_query)
            elif tool.type == ToolType.API_CALL:
                result = self._execute_api_call(tool, parameters)
            elif tool.type == ToolType.CALCULATION:
                result = self._execute_calculation(parameters, user_query)
            elif tool.type == ToolType.GENERAL_INFO:
                result = self._execute_general_info(user_query)
            else:
                result = f"Tool type {tool.type.value} requires file upload and should be called directly"

            execution_time = (datetime.now() - start_time).total_seconds()

            return QueryResult(
                success=True,
                tool_used=tool_name,
                raw_response=result,
                formatted_response=self._format_response(
                    result, user_query, tool_name),
                execution_time=execution_time,
                confidence=0.9
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Query execution failed: {str(e)}")

            return QueryResult(
                success=False,
                tool_used=tool_name,
                raw_response=None,
                formatted_response=f"Sorry, I encountered an error: {str(e)}",
                execution_time=execution_time,
                confidence=0.0,
                error_message=str(e)
            )

    def _execute_database_query(self, parameters: Dict[str, Any], user_query: str) -> Any:
        """Execute database query"""
        sql_query = parameters.get('sql_query', '')
        if not sql_query:
            raise ValueError("No SQL query provided")

        # Simple SQLite example - replace with your actual database
        conn = sqlite3.connect('app_data.db')
        cursor = conn.cursor()

        try:
            cursor.execute(sql_query)
            if sql_query.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                columns = [description[0]
                           for description in cursor.description]
                return [dict(zip(columns, row)) for row in results]
            else:
                conn.commit()
                return {"affected_rows": cursor.rowcount}
        finally:
            conn.close()

    def _execute_api_call(self, tool: Tool, parameters: Dict[str, Any]) -> Any:
        """Execute API call"""
        api_params = parameters.get('api_params', {})

        if tool.name == "weather_api":
            # Example weather API call
            url = tool.endpoint
            params = {
                'q': api_params.get('location', 'Dhaka'),
                'appid': 'your_api_key_here',  # Replace with actual API key
                'units': 'metric'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        else:
            raise ValueError(f"API call for {tool.name} not implemented")

    def _execute_calculation(self, parameters: Dict[str, Any], user_query: str) -> Any:
        """Execute calculation using LLM"""
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

        return response.json().get('response', '')

    def _execute_general_info(self, user_query: str) -> Any:
        """Execute general information query using LLM"""
        prompt = f"""
You are a helpful AI assistant. Answer the following question accurately and comprehensively:

QUESTION: "{user_query}"

Provide a clear, informative response that directly addresses the user's question.
If you're not certain about something, acknowledge the uncertainty.
Structure your response in a readable format with appropriate sections if needed.
"""

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

        return response.json().get('response', '')

    def _format_response(self, raw_response: Any, user_query: str, tool_name: str) -> str:
        """Format raw response into human-readable format"""
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

            return response.json().get('response', str(raw_response))

        except Exception as e:
            logger.error(f"Response formatting failed: {str(e)}")
            # Fallback to simple string conversion
            if isinstance(raw_response, (dict, list)):
                return json.dumps(raw_response, indent=2)
            return str(raw_response)
