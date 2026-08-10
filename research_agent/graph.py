from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import plan_node, search_node, read_node, synthesize_node, write_node

def build_graph():
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
    graph.add_edge("write",      END)
    return graph.compile()

agent = build_graph()

def run_agent(query: str) -> AgentState:
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
