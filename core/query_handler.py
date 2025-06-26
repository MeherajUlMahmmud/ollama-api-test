from core.common.query_analyzer import QueryAnalyzer
from core.common.query_executor import QueryExecutor


class QueryHandler:
    def __init__(self, query_analyzer: QueryAnalyzer, query_executor: QueryExecutor):
        self.query_analyzer = query_analyzer
        self.query_executor = query_executor

    def handle_query(self, query_text: str) -> str:
        tool_response = self.query_analyzer.analyze_query(query_text)
        
        llm_response = self.query_executor.execute_query(
            tool_name=tool_response['tool_name'],
            parameters=tool_response['parameters'],
            user_query=query_text
        )
        
        return llm_response