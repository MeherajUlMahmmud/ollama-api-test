import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional, List


class ToolType(Enum):
    """Available tool types for query processing"""
    DATABASE = "database"
    API_CALL = "api_call"
    GENERAL_INFO = "general_info"
    CALCULATION = "calculation"


@dataclass
class Tool:
    """Tool configuration"""
    name: str
    type: ToolType
    description: str
    parameters: Dict[str, Any]
    endpoint: Optional[str] = None
    tags: List[str] = None

@dataclass()
class QueryResult:
    """Result of query processing"""
    success: bool
    tool_used: str
    raw_response: Any
    formatted_response: str
    execution_time: float
    error_message: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)
