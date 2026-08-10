from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    query: str
    plan: List[str]
    search_results: List[dict]
    extracted_content: List[dict]
    findings: str
    report: str
    steps_taken: List[str]
    error: Optional[str]
