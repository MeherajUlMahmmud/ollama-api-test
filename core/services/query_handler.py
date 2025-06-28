from typing import Dict, Any

from core.common.query_analyzer import QueryAnalyzer
from core.common.query_executor import QueryExecutor
from logger import Logger


class QueryHandler:
    """
    A class to handle query processing by analyzing and executing queries.

    This class coordinates between a QueryAnalyzer to parse and analyze incoming queries
    and a QueryExecutor to execute the analyzed queries. It serves as the main interface
    for processing user queries in the system.

    Attributes:
        query_analyzer (QueryAnalyzer): An instance of QueryAnalyzer to analyze queries.
        query_executor (QueryExecutor): An instance of QueryExecutor to execute queries.
        logger (Logger): Logger instance for recording query handling activities.
    """

    def __init__(self, query_analyzer: QueryAnalyzer, query_executor: QueryExecutor):
        """
        Initialize the QueryHandler with analyzer and executor instances.

        Args:
            query_analyzer (QueryAnalyzer): The query analyzer to parse and analyze queries.
            query_executor (QueryExecutor): The query executor to run the analyzed queries.
        """
        self.query_analyzer = query_analyzer
        self.query_executor = query_executor
        self.logger = Logger.get_logger()
        self.logger.debug("QueryHandler initialized with QueryAnalyzer and QueryExecutor")

    def handle_query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a user query by analyzing and executing it.

        This method takes a query string, analyzes it using the QueryAnalyzer to determine
        the appropriate tool and parameters, and (if enabled) executes it using the QueryExecutor.

        Args:
            query_text (str): The raw query string provided by the user.

        Returns:
            Dict[str, Any]: A dictionary containing the query analysis results or execution results.

        Raises:
            Exception: If query analysis or execution fails, logs the error and re-raises it.
        """
        self.logger.info(f"Handling query: {query_text}")

        try:
            self.logger.debug(f"Analyzing query: {query_text}")
            tool_response = self.query_analyzer.analyze_query(query_text)
            self.logger.debug(f"Query analysis completed: {tool_response}")

            # query_result = self.query_executor.execute_query(
            #     tool_name=tool_response['tool_name'],
            #     parameters=tool_response['parameters'],
            #     user_query=query_text
            # )
            # self.logger.debug(f"Query execution completed with result: {query_result}")

            self.logger.info(f"Query handling completed successfully for: {query_text}")
            return tool_response  # Return tool_response for now, as execution is commented out

        except Exception as e:
            self.logger.error(f"Failed to handle query: {query_text} - Error: {str(e)}", exc_info=True)
            raise
