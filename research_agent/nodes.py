import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from .state import AgentState
from .tools import web_search, read_url

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        max_output_tokens=2048,
    )

def plan_node(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = f"""You are a research planner. Break this query into 3-4 specific search sub-tasks.
Return ONLY a numbered list, one sub-task per line, no extra text.

Query: {state['query']}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
    plan = [l.lstrip("0123456789.-) ") for l in lines if l]
    return {
        **state,
        "plan": plan,
        "steps_taken": state.get("steps_taken", []) + [
            f"📋 **Plan created** — {len(plan)} research tasks identified"
        ]
    }

def search_node(state: AgentState) -> AgentState:
    all_results = []
    steps = state.get("steps_taken", [])
    for task in state["plan"]:
        results = web_search(task, max_results=3)
        all_results.extend(results)
        steps = steps + [f"🔍 **Searched:** {task} → {len(results)} results"]
    seen = set()
    unique = []
    for r in all_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)
    return {
        **state,
        "search_results": unique,
        "steps_taken": steps + [f"✅ **Search complete** — {len(unique)} unique sources found"]
    }

def read_node(state: AgentState) -> AgentState:
    extracted = []
    steps = state.get("steps_taken", [])
    for result in state["search_results"][:6]:
        content = read_url(result["url"])
        extracted.append({
            "title":   result["title"],
            "url":     result["url"],
            "snippet": result["snippet"],
            "content": content
        })
        steps = steps + [f"📄 **Read:** {result['title'][:60]}..."]
    return {
        **state,
        "extracted_content": extracted,
        "steps_taken": steps + [f"✅ **Reading complete** — {len(extracted)} sources extracted"]
    }

def synthesize_node(state: AgentState) -> AgentState:
    llm = get_llm()
    sources_text = ""
    for i, src in enumerate(state["extracted_content"], 1):
        sources_text += f"\n\nSource {i}: {src['title']}\nURL: {src['url']}\n{src['content'][:1000]}"
    prompt = f"""You are a research analyst. Synthesize the following sources into key findings.

Original Query: {state['query']}

Sources:
{sources_text}

Produce:
1. KEY FINDINGS (5-7 bullet points)
2. RESEARCH GAPS (2-3 areas not well covered)
3. KEY PAPERS/SOURCES (most relevant ones with URLs)

Be precise and technical. Do not hallucinate."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        **state,
        "findings": response.content,
        "steps_taken": state.get("steps_taken", []) + ["🧠 **Synthesis complete** — key findings identified"]
    }

def write_node(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = f"""You are a technical research writer. Write a structured research report.

Query: {state['query']}

Synthesized Findings:
{state['findings']}

Write a professional report with these exact sections:
# [Report Title]

## Executive Summary
## Background
## Key Findings
## Current State of the Art
## Research Gaps & Future Directions
## Conclusion
## References

Be technical, precise, and cite sources throughout."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        **state,
        "report": response.content,
        "steps_taken": state.get("steps_taken", []) + ["✍️ **Report written** — ready to read"]
    }
