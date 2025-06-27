from typing import Dict, List, Optional

from core.common.dataclass import Tool, ToolType


class ToolRegistry:
    """Registry for managing available tools"""

    def __init__(self):
        self.tools = self._initialize_tools()

    def _initialize_tools(self) -> Dict[str, Tool]:
        """Initialize available tools"""
        return {
            "hrms_db": Tool(
                name="Human Resource Management Database",
                type=ToolType.DATABASE,
                description="Execute SQL queries on the database to retrieve user data, transaction records, analytics, etc.",
                parameters={
                    "connection_string": "sqlite:///app_data.db",
                    "supported_tables": ["users", "nid_records", "liveness_checks", "api_logs"],
                    "supported_operations": ["SELECT", ],
                    "example_query": "SELECT * FROM users WHERE id = ?",
                },
                tags=["employees", "designations", "salary", "divisions", "staffs", "attendance", "absent", ]
            ),
            "cbs_db": Tool(
                name="Core Banking System Database",
                type=ToolType.DATABASE,
                description="Execute SQL queries on the database to retrieve user data, transaction records, analytics, etc.",
                parameters={
                    "connection_string": "sqlite:///app_data.db",
                    "supported_tables": ["users", "nid_records", "liveness_checks", "api_logs"],
                    "supported_operations": ["SELECT", ],
                    "example_query": "SELECT * FROM users WHERE id = ?",
                },
                tags=["core banking", "cbs", "core banking system", ]
            ),
            "mbs_db": Tool(
                name="Mobile Banking System Database",
                type=ToolType.DATABASE,
                description="Execute SQL queries on the database to retrieve user data, transaction records, analytics, etc.",
                parameters={
                    "connection_string": "sqlite:///app_data.db",
                    "supported_tables": ["users", "nid_records", "liveness_checks", "api_logs"],
                    "supported_operations": ["SELECT", ],
                    "example_query": "SELECT * FROM users WHERE id = ?",
                },
                tags=["mobile banking", "mobile banking system", "mbs", "rocket", "rocket app", "rocket system", ]
            ),
            "calculation": Tool(
                name="calculation",
                type=ToolType.CALCULATION,
                description="Perform mathematical calculations and data analysis",
                parameters={},
                tags=[]
            ),
            "general_info": Tool(
                name="general_info",
                type=ToolType.GENERAL_INFO,
                description="Provide general information and answers to questions",
                parameters={},
                tags=[]
            )
        }

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(tool_name)

    def get_all_tools(self) -> List[Tool]:
        """Get all available tools"""
        return list(self.tools.values())
