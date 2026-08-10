import sys
import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import TypedDict, List, Optional

# ── Load secrets ──────────────────────────────────────────────────────────────
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    plan: List[str]
    search_results: List[dict]
    extracted_content: List[dict]
    findings: str
    report: str
    steps_taken: List[str]
    error: Optional[str]

# ── Tools ─────────────────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 5) -> list:
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=False
    )
    results = []
    for r in response.get("results", []):
        results.append({
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "snippet": r.get("content", "")
        })
    return results

def read_url(url: str, max_chars: int = 3000) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header","aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
        return "\n".join(lines)[:max_chars]
    except Exception as e:
        return f"Could not read URL: {str(e)}"

def get_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        max_output_tokens=2048,
    )

# ── Nodes ─────────────────────────────────────────────────────────────────────
def plan_node(state: AgentState) -> AgentState:
    from langchain_core.messages import HumanMessage
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
    from langchain_core.messages import HumanMessage
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
    from langchain_core.messages import HumanMessage
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

# ── Graph ─────────────────────────────────────────────────────────────────────
def build_graph():
    from langgraph.graph import StateGraph, END
    graph = StateGraph(AgentState)
    graph.add_node("plan",       plan_node)
    graph.add_node("search",     search_node)
    graph.add_node("read",       read_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("write",      write_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan",       "search")
    graph.add_edge("search",     "read")
    graph.add_edge("read",       "synthesize")
    graph.add_edge("synthesize", "write")
    from langgraph.graph import END
    graph.add_edge("write", END)
    return graph.compile()

def run_agent(query: str) -> AgentState:
    agent = build_graph()
    initial_state: AgentState = {
        "query":             query,
        "plan":              [],
        "search_results":    [],
        "extracted_content": [],
        "findings":          "",
        "report":            "",
        "steps_taken":       [f"🚀 **Starting research** on: *{query}*"],
        "error":             None,
    }
    return agent.invoke(initial_state)

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchAgent — Autonomous Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ResearchAgent — Autonomous 5G Research Assistant")
st.markdown(
    "A multi-step AI agent that **autonomously searches, reads, synthesizes, and writes** "
    "a structured technical report on any query. "
    "*Powered by LangGraph + Gemini + Tavily*"
)
st.divider()

# Check keys
missing = []
if not os.environ.get("GOOGLE_API_KEY"):
    missing.append("GOOGLE_API_KEY")
if not os.environ.get("TAVILY_API_KEY"):
    missing.append("TAVILY_API_KEY")
if missing:
    st.error(f"⚠️ Missing API keys: {', '.join(missing)}. Add in Streamlit Cloud → Settings → Secrets.")
    st.stop()

with st.sidebar:
    st.header("About")
    st.markdown("""
**How it works:**
1. 📋 Plans research sub-tasks
2. 🔍 Searches the web (Tavily)
3. 📄 Reads each source
4. 🧠 Synthesizes findings
5. ✍️ Writes structured report

**Stack:** LangGraph · Gemini · Tavily · Streamlit
""")
    st.markdown("---")
    examples = [
        "Latest advances in GNN-based channel decoding for 5G NR",
        "LDPC vs Polar codes in 5G — current research landscape",
        "Transformer models for wireless channel estimation",
        "Federated learning for 6G network optimization",
        "AI-driven beam management in mmWave 5G networks",
    ]
    st.markdown("**Example queries:**")
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query_input"] = ex
    st.markdown("---")
    st.caption("Built by [B S Rikeesh](https://linkedin.com/in/bsrikeesh)")

query = st.text_input(
    "Enter your research query",
    placeholder="e.g. Latest advances in GNN-based channel decoding for 5G NR",
    key="query_input"
)

run_btn = st.button("🚀 Run Agent", type="primary", use_container_width=True)

if run_btn and query:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🔄 Agent Steps")
        steps_placeholder = st.empty()
    with col2:
        st.subheader("📄 Research Report")
        report_placeholder = st.empty()

    with st.spinner("Agent running — this takes 30-60 seconds..."):
        try:
            result = run_agent(query)
            with steps_placeholder.container():
                for step in result["steps_taken"]:
                    st.markdown(step)
            with report_placeholder.container():
                if result["report"]:
                    st.markdown(result["report"])
                    st.divider()
                    st.download_button(
                        label="📥 Download Report",
                        data=result["report"],
                        file_name=f"report_{query[:30].replace(' ','_')}.md",
                        mime="text/markdown"
                    )
            st.divider()
            st.subheader("🔗 Sources Retrieved")
            for i, src in enumerate(result["extracted_content"], 1):
                with st.expander(f"Source {i} — {src['title'][:70]}"):
                    st.markdown(f"**URL:** {src['url']}")
                    st.markdown(f"> {src['snippet'][:300]}...")
        except Exception as e:
            st.error(f"❌ Agent error: {str(e)}")

elif run_btn and not query:
    st.warning("Please enter a research query first.")

st.divider()
st.caption(
    "ResearchAgent | LangGraph + Gemini + Tavily · "
    "[GitHub](https://github.com/bsrikeesh/researchagent) · "
    "[LinkedIn](https://linkedin.com/in/bsrikeesh)"
)
