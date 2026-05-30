"""Streamlit chat UI for the Airport Investment Intelligence Agent.

Run:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import Agent
from llm import get_provider

load_dotenv()

st.set_page_config(page_title="Airport Investment Intelligence", page_icon="✈️", layout="wide")
st.title("✈️ Airport Investment Intelligence Agent")
st.caption("Ranks US airports for renovation/expansion ROI using a deterministic Expansion "
           "Profitability Index, with an LLM for routing, explanation, and a bounded score nudge.")

# --------------------------------------------------------------------------- #
# Sidebar — model, λ, keys
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Model", ["Deterministic (no LLM)", "gemini", "claude"],
        help="Pick an LLM for NL routing + explanations, or run the deterministic fallback.",
    )

    # let the user paste a key in-session (avoids needing a .env for the demo)
    if model_choice == "gemini" and not os.getenv("GEMINI_API_KEY"):
        k = st.text_input("GEMINI_API_KEY", type="password")
        if k:
            os.environ["GEMINI_API_KEY"] = k
    if model_choice == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        k = st.text_input("ANTHROPIC_API_KEY", type="password")
        if k:
            os.environ["ANTHROPIC_API_KEY"] = k

    lam = st.slider(
        "LLM influence (λ)", 0.0, 1.0, 0.0, 0.05,
        help="How much the LLM may adjust deterministic scores. 0 = pure deterministic; "
             "1 = ±30% max, and only with a written justification.",
    )

    provider = None
    if model_choice != "Deterministic (no LLM)":
        provider = get_provider(model_choice)
        if provider is None:
            st.warning(f"No working {model_choice} key — using the deterministic fallback.")
        else:
            st.success(f"Using {provider.name} ({provider.model})")

    st.divider()
    st.markdown("**Try:**")
    st.markdown("- Which airports in New England are strong candidates for terminal expansion?\n"
                "- Compare LA and Santa Ana congestion levels.\n"
                "- What is the percentage of long haul flights out of Anchorage?\n"
                "- What is the unmet flight demand in SFO and why?")

# --------------------------------------------------------------------------- #
# Agent (persisted across reruns so history survives; settings updated live)
# --------------------------------------------------------------------------- #
if "agent" not in st.session_state:
    st.session_state.agent = Agent()
    st.session_state.messages = []
agent: Agent = st.session_state.agent
agent.provider = provider
agent.lam = lam


def render_result(result: dict):
    """Transparency panels: the route, the deterministic steps, and the scoring table."""
    if result.get("results") and result.get("intent") == "rank":
        rows = []
        for r in result["results"]:
            row = {"IATA": r["iata"], "City": r.get("city"), "EPI": r["epi"]}
            if "final_score" in r:
                row["Modifier"] = r.get("modifier")
                row["Final"] = r["final_score"]
                row["Why (LLM)"] = r.get("modifier_reason") or "-"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if result.get("steps"):
        with st.expander("Deterministic steps"):
            for s in result["steps"]:
                st.markdown(f"- {s}")
    if result.get("notes"):
        with st.expander("Assumptions & caveats"):
            for n in result["notes"]:
                st.markdown(f"- {n}")


# replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("result"):
            render_result(m["result"])

# new turn
if prompt := st.chat_input("Ask about US airport investment opportunities..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            out = agent.ask(prompt)
        st.markdown(out["answer"])
        with st.expander("How I read your question (route)"):
            st.json(out["route"])
        render_result(out["result"])
    st.session_state.messages.append(
        {"role": "assistant", "content": out["answer"], "result": out["result"]}
    )
