"""Streamlit chat UI for the Airport Investment Intelligence Agent.

Run:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from airport_intel import backtest
from airport_intel.agent import Agent
from airport_intel.llm import get_provider
from airport_intel.metrics import global_tracker
from airport_intel.scoring import MAX_SWING
from airport_intel.tools import DATA_NOTES

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Global methodology assumptions/caveats — shown once in the top-right "⋮" → About
# menu instead of being repeated under every answer.
_ASSUMPTIONS_MD = "### Assumptions & caveats\n\n" + "\n".join(f"- {n}" for n in DATA_NOTES)


def _metrics_md() -> str:
    """Lifetime LLM usage for the top-right "⋮" → About dialog.

    Streamlit's menu_items only allows the fixed keys About / Get help / Report a Bug, so
    metrics live as a section inside About. The whole script reruns after every answer, so
    set_page_config runs again and this section refreshes with the latest cumulative totals.
    Totals are persisted and monotonic — they accumulate across runs and never reset.
    """
    t = global_tracker().snapshot()
    return (
        "\n\n---\n\n### Metrics (lifetime)\n\n"
        f"- **LLM calls:** {t.calls:,}\n"
        f"- **Tokens:** {t.total_tokens:,} "
        f"({t.input_tokens:,} in / {t.output_tokens:,} out)\n"
        f"- **Estimated cost:** ${t.cost_usd:,.4f}\n\n"
        "_Cost is an estimate from public per-model rates; totals persist across runs "
        "and only ever increase._"
    )


def _disable_keyboard_shortcuts() -> None:
    """Make the app mouse-only by suppressing keyboard activations.

    Streamlit ships built-in hotkeys we don't define ourselves: "C" clears the
    cache, "R" reruns, and Enter/Return submits the chat box. This injects a
    capture-phase keydown blocker on the parent document so those keys do nothing
    — the user must click the ⋮ menu items / the chat send button (mouse) instead.
    Guarded so reruns don't stack duplicate listeners; scroll/tab keys are left
    alone since those navigate rather than activate.
    """
    components.html(
        """
        <script>
        const w = window.parent;
        if (!w.__mouseOnly) {
            w.__mouseOnly = true;
            w.document.addEventListener('keydown', function (e) {
                const t = e.target;
                const typing = t && (t.tagName === 'INPUT' ||
                                     t.tagName === 'TEXTAREA' || t.isContentEditable);
                if (typing) {
                    // Enter/Return would submit — block it; allow normal typing.
                    if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); }
                    return;
                }
                // Outside a field, swallow Streamlit's global hotkeys: R and C.
                const k = e.key.toLowerCase();
                if (k === 'r' || k === 'c') { e.preventDefault(); e.stopPropagation(); }
            }, true);
        }
        </script>
        """,
        height=0,
    )


st.set_page_config(
    page_title="Airport Investment Intelligence",
    page_icon="✈️",
    layout="wide",
    menu_items={"About": _ASSUMPTIONS_MD + _metrics_md()},
)
_disable_keyboard_shortcuts()
st.title("✈️ Airport Investment Intelligence Agent")
st.caption("Ranks US airports for renovation/expansion ROI using a deterministic Expansion "
           "Profitability Index, with an LLM for routing, explanation, and a bounded score nudge.")


# --------------------------------------------------------------------------- #
# Validation panel — does the EPI actually predict real-world funding decisions?
# A separate analysis surface (the chat agent + router are untouched): the EPI is
# backtested against the FAA Airport Terminal Program ($5B of competitive grants).
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _run_backtest() -> dict:
    return backtest.run_backtest()


with st.expander("📊 Is the EPI any good? — backtest vs $3.8B of real FAA terminal grants"):
    try:
        bt = _run_backtest()
    except FileNotFoundError:
        st.info("Funding ground truth not built yet — run `python -m etl.build_funding`.")
    else:
        st.caption(
            f"EPI stands for Expansion Profitability Index. It is a measurement created for evaluating airport investments."
            f"Validated against the FAA Airport Terminal Program (ATP), a *competitive* grant "
            f" that funds terminal explansion. The correlation between them shows the EPI is a valid measurement. "
            f"{bt['funded_in_universe']} of {bt['universe']} airports were funded "
            f"(base rate {bt['base_rate']:.0%})"
        )
        p20 = next((p for p in bt["precision"] if p["n"] == 20), bt["precision"][0])
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision@20 (EPI)", f"{p20['epi_precision']:.0%}",
                  help="Share of the top-20 EPI airports that actually received ATP funding.")
        c2.metric("vs. base rate", f"{bt['base_rate']:.0%}",
                  delta=f"{p20['epi_precision'] - bt['base_rate']:+.0%} over chance")
        c3.metric("Spearman(EPI, grant $)", f"{bt['spearman_epi_grant_usd']:+.2f}",
                  help="Rank correlation between EPI and ATP dollars across all airports.")

        st.markdown("**Precision — EPI vs. ATP grants**")
        st.dataframe(pd.DataFrame([
            {"Top": p["n"], "Funded": f"{p['epi_precision']:.0%}"}
            for p in bt["precision"]
        ]), hide_index=True, use_container_width=True)
        st.caption("The EPI tracks real funding strongly, well above the base rate. The "
                   "differentiated output is the *unfunded* high-EPI shortlist below: airports "
                   "the competitive process has not captured yet.")

        st.markdown(f"**🎯 Investment shortlist — high EPI, _not yet_ ATP-funded "
                    f"(≥ {bt['shortlist_min_passengers']:,} passengers)**")
        st.dataframe(pd.DataFrame([
            {"IATA": r["iata"], "City": r.get("city"), "State": r.get("state"),
             "EPI": r["epi"], "Passengers": r["passengers"]}
            for r in bt["shortlist"]
        ]), hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- #
# Sidebar — model, λ, keys
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Model", ["gemini", "claude"],
        help="The LLM that powers routing, explanation, and the bounded score nudge.",
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
        "LLM influence (λ)", 0.0, 1.0, 0.5, 0.05,
        help="How much the LLM may adjust deterministic scores, when calculating an investment score. 0 = pure deterministic; "
             "1 = ±40% max, and only with written justifications.",
    )


    # Dynamic explanation of what the current λ does to scoring.
    swing_pct = lam * MAX_SWING * 100
    if lam == 0.0:
        st.caption("**λ = 0** → pure deterministic: the LLM can't move scores at all "
                   "(modifier locked to 1.0).")
    else:
        st.caption(f"**λ = {lam:.2f}** → the LLM may nudge each score by at most "
                   f"**±{swing_pct:.0f}%**, and only with a written justification.")

    provider = get_provider(model_choice)
    if provider is None:
        st.warning(f"Add a {model_choice} API key above to start — the assistant requires an LLM.")
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
if provider is None:
    st.info("👈 Choose a model and add its API key in the sidebar to start chatting.")
    st.stop()

if "agent" not in st.session_state:
    st.session_state.agent = Agent(provider)
    st.session_state.messages = []
agent: Agent = st.session_state.agent
agent.provider = provider
agent.lam = lam


def render_result(result: dict):
    """Transparency panels: the route, the deterministic steps, and the scoring table."""
    if result.get("results") and result.get("intent") == "rank":
        rows = []
        # When ranked by a single KPI (e.g. growth), show that metric as the lead column.
        ranked_by = result.get("ranked_by")
        metric_col = ranked_by if ranked_by and ranked_by != "EPI" else None
        for r in result["results"]:
            row = {"IATA": r["iata"], "City": r.get("city")}
            if metric_col:
                row[metric_col] = r.get("metric_display")
            row["EPI"] = r["epi"]
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
    # Assumptions & caveats are global to the methodology, so they live once in the
    # top-right "⋮" → About menu rather than under every answer.


# --------------------------------------------------------------------------- #
# Turn handling — two phases so the input can be locked while we work:
#   1. Submit: stash the question in `pending` and rerun.
#   2. Busy rerun: the input renders disabled, then we answer synchronously.
# This prevents spamming a second question on top of an in-flight one without
# any background threads. (Streamlit's built-in top-right "running" indicator
# still lets the user abort a genuinely stuck run.)
# --------------------------------------------------------------------------- #
st.session_state.setdefault("pending", None)

# replay history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("route"):
            with st.expander("How I read your question (route)"):
                st.json(m["route"])
        if m.get("result"):
            render_result(m["result"])

busy = st.session_state.pending is not None

# Input is disabled while a question is in flight — prevents spamming a new one.
if user_input := st.chat_input(
    "Ask about US airport investment opportunities...", disabled=busy
):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.pending = user_input
    st.rerun()

# Answer the pending question. The input above is already rendered (disabled),
# so it stays locked for the duration of this blocking call.
if busy:
    prompt = st.session_state.pending
    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            try:
                out = agent.ask(prompt)
            except Exception as e:  # surface failures instead of hanging
                out = None
                st.session_state.messages.append(
                    {"role": "assistant",
                     "content": f"⚠️ Something went wrong while answering: {e}"}
                )
    if out is not None:
        st.session_state.messages.append(
            {"role": "assistant", "content": out["answer"],
             "route": out["route"], "result": out["result"]}
        )
    st.session_state.pending = None
    st.rerun()
