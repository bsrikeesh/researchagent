"""
ResearchAgent — Autonomous 5G Research Assistant
Single-file Streamlit app with LangGraph agent.
All code inline to avoid namespace collisions.
"""

import streamlit as st
import os
import json
import re
from typing import TypedDict, List, Annotated
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# External imports (loaded inside functions where needed to avoid init issues)
# ---------------------------------------------------------------------------

def _get_tavily_client():
    from tavily import TavilyClient
    return TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

def _get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"],
    )

# ---------------------------------------------------------------------------
# Helper: safely extract string content from LLM response
# ---------------------------------------------------------------------------

def _safe_content(response) -> str:
    """Extract string content from LLM response, handling list responses."""
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)
    
    # Gemini sometimes returns a list of content parts
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    
    return str(content)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    plan: List[str]
    search_results: List[dict]
    extracted_findings: List[str]
    synthesis: str
    report: str
    current_step: str
    logs: Annotated[List[str], "append"]

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> List[dict]:
    """Search the web using Tavily."""
    client = _get_tavily_client()
    response = client.search(query=query, max_results=max_results, search_depth="advanced")
    results = []
    for r in response.get("results", []):
        results.append({
            "title": str(r.get("title", "")),
            "url": str(r.get("url", "")),
            "content": str(r.get("content", "")),
            "score": r.get("score", 0),
        })
    return results

def fetch_page_text(url: str) -> str:
    """Fetch and extract clean text from a URL."""
    import requests
    from bs4 import BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0 (ResearchAgent/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        return f"[Error fetching {url}: {e}]"

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def plan_node(state: AgentState) -> AgentState:
    llm = _get_llm()
    prompt = f"""You are a research planner. Given the user query, break it into 3-5 concrete sub-tasks for a research agent.
Return ONLY a JSON array of strings. No markdown, no explanation.

Query: {state['query']}
"""
    response = llm.invoke(prompt)
    text = _safe_content(response)
    # Extract JSON array
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            plan = json.loads(match.group())
            if not isinstance(plan, list):
                plan = [state["query"]]
        except json.JSONDecodeError:
            plan = [state["query"]]
    else:
        plan = [state["query"]]
    
    # Ensure all items are strings
    plan = [str(p).strip() for p in plan if p]
    
    state["plan"] = plan
    state["current_step"] = "plan"
    state["logs"].append(f"📋 Planned {len(plan)} sub-tasks: {plan}")
    return state

def search_node(state: AgentState) -> AgentState:
    query = state["query"]
    state["current_step"] = "search"
    state["logs"].append("🔍 Running web search...")
    results = web_search(query, max_results=5)
    state["search_results"] = results
    state["logs"].append(f"✅ Found {len(results)} sources")
    for r in results:
        title = str(r.get('title', 'Untitled'))
        url = str(r.get('url', ''))[:60]
        state["logs"].append(f"   • {title} ({url}...)")
    return state

def read_node(state: AgentState) -> AgentState:
    state["current_step"] = "read"
    state["logs"].append("📖 Reading sources...")
    findings = []
    for r in state["search_results"]:
        url = str(r["url"])
        text = fetch_page_text(url)
        # Truncate to avoid token explosion
        truncated = str(text)[:4000]
        title = str(r.get("title", "Untitled"))
        findings.append(f"SOURCE: {title}\nURL: {url}\nCONTENT: {truncated}\n---")
    state["extracted_findings"] = findings
    state["logs"].append(f"✅ Extracted text from {len(findings)} sources")
    return state

def synthesize_node(state: AgentState) -> AgentState:
    llm = _get_llm()
    findings_text = "\n\n".join(str(f) for f in state["extracted_findings"])
    prompt = f"""You are a technical research synthesizer. Based on the findings below, produce a concise synthesis that:
1. Summarizes key advances
2. Notes any limitations or gaps
3. Mentions how this relates to baseline methods (WBP, BP) if relevant

Findings:
{findings_text}

Synthesis:"""
    response = llm.invoke(prompt)
    synthesis = _safe_content(response)
    state["synthesis"] = synthesis
    state["current_step"] = "synthesize"
    state["logs"].append("🧠 Synthesized findings")
    return state

def write_node(state: AgentState) -> AgentState:
    llm = _get_llm()
    sources_md = ""
    for r in state["search_results"]:
        title = str(r.get('title', 'Untitled'))
        url = str(r.get('url', ''))
        sources_md += f"- [{title}]({url})\n"
    
    prompt = f"""You are a technical report writer. Write a structured research report in Markdown based on the synthesis and sources.

User Query: {state['query']}

Synthesis:
{state['synthesis']}

Sources:
{sources_md}

Report structure:
# Research Report: <Title>
## Executive Summary
## Key Findings
## Technical Details
## Research Gaps & Future Directions
## References

Report:"""

    response = llm.invoke(prompt)
    report = _safe_content(response)
    state["report"] = report
    state["current_step"] = "write"
    state["logs"].append("📝 Generated structured report")
    return state

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(AgentState)
    workflow.add_node("plan", plan_node)
    workflow.add_node("search", search_node)
    workflow.add_node("read", read_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("write", write_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "read")
    workflow.add_edge("read", "synthesize")
    workflow.add_edge("synthesize", "write")
    workflow.add_edge("write", END)

    return workflow.compile()

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def run_agent(query: str) -> AgentState:
    graph = build_graph()
    initial_state: AgentState = {
        "query": str(query),
        "plan": [],
        "search_results": [],
        "extracted_findings": [],
        "synthesis": "",
        "report": "",
        "current_step": "start",
        "logs": ["🚀 Agent started"],
    }
    # Use invoke() instead of stream() to get final state directly
    final_state = graph.invoke(initial_state)
    return final_state

def main():
    st.set_page_config(page_title="ResearchAgent", page_icon="🔬", layout="wide")
    st.title("🔬 ResearchAgent — Autonomous 5G Research Assistant")
    st.markdown("Powered by **LangGraph + Gemini 2.0 Flash + Tavily**")

    query = st.text_input(
        "Enter your research query:",
        value="Latest advances in GNN-based channel decoding for 5G NR",
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🪵 Agent Logs")
        log_container = st.container()

    with col2:
        st.subheader("📄 Final Report")
        report_container = st.container()

    if st.button("🚀 Run Agent", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        # Check secrets
        missing = []
        try:
            _ = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            missing.append("GOOGLE_API_KEY")
        try:
            _ = st.secrets["TAVILY_API_KEY"]
        except Exception:
            missing.append("TAVILY_API_KEY")
        if missing:
            st.error(f"Missing secrets: {', '.join(missing)}. Add them in .streamlit/secrets.toml")
            return

        with st.spinner("Agent is thinking..."):
            try:
                final_state = run_agent(query)
            except Exception as e:
                st.error(f"Agent error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

        with log_container:
            for log in final_state.get("logs", []):
                st.text(str(log))

        with report_container:
            report = final_state.get("report", "")
            if report:
                st.markdown(str(report))
                st.download_button(
                    "⬇️ Download Report",
                    data=str(report),
                    file_name="research_report.md",
                    mime="text/markdown",
                )
            else:
                st.info("No report generated.")

    st.divider()
    st.caption("Built with LangGraph | Gemini 2.0 Flash | Tavily | Streamlit")

if __name__ == "__main__":
    main()
