from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    query: str                          # Original user query
    plan: List[str]                     # Sub-tasks planned
    search_results: List[dict]          # Raw search results
    extracted_content: List[dict]       # Extracted text from URLs
    findings: str                       # Synthesized findings
    report: str                         # Final written report
    steps_taken: List[str]              # Visible log for UI
    error: Optional[str]                # Any error that occurred
