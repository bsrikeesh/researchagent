import streamlit as st
import os
from agent import run_agent

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

# Check API keys
missing = []
if not os.environ.get("GOOGLE_API_KEY"):
    missing.append("GOOGLE_API_KEY")
if not os.environ.get("TAVILY_API_KEY"):
    missing.append("TAVILY_API_KEY")
if missing:
    st.error(f"⚠️ Missing API keys: {', '.join(missing)}. Add them in Streamlit Cloud → Settings → Secrets.")
    st.stop()

# Sidebar
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
    st.markdown("**Example queries:**")
    examples = [
        "Latest advances in GNN-based channel decoding for 5G NR",
        "LDPC vs Polar codes in 5G — current research landscape",
        "Transformer models for wireless channel estimation",
        "Federated learning for 6G network optimization",
        "AI-driven beam management in mmWave 5G networks",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["query_input"] = ex
    st.markdown("---")
    st.caption("Built by [B S Rikeesh](https://linkedin.com/in/bsrikeesh)")

# Query input
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

            # Show steps
            with steps_placeholder.container():
                for step in result["steps_taken"]:
                    st.markdown(step)

            # Show report
            with report_placeholder.container():
                if result["report"]:
                    st.markdown(result["report"])
                    st.divider()
                    st.download_button(
                        label="📥 Download Report",
                        data=result["report"],
                        file_name=f"research_report_{query[:30].replace(' ','_')}.md",
                        mime="text/markdown"
                    )

            # Show sources
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
    "ResearchAgent | Built with LangGraph + Gemini + Tavily · "
    "[GitHub](https://github.com/bsrikeesh/researchagent) · "
    "[LinkedIn](https://linkedin.com/in/bsrikeesh)"
)
