
from dataclasses import dataclass, asdict
from enum import Enum
import json
from typing import Any, Dict, Optional


class ToolType(Enum):
    """Available tool types for query processing"""
    DATABASE = "database"
    API_CALL = "api_call"
    NID_OCR = "nid_ocr"
    LIVENESS_CHECK = "liveness_check"
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


class QueryResult:
    """Result of query processing"""
    success: bool
    tool_used: str
    raw_response: Any
    formatted_response: str
    execution_time: float
    confidence: float
    error_message: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)
