"""
Streamlit dashboard for the Autonomous LinkedIn Content Manager (CrewAI).

This wraps the existing 5-agent sequential pipeline defined in
`linkedin_content_manager.py` with an interactive, animated UI:

- Two inputs: Topic/Niche + Target Audience/Angle (optional).
- A live animated pipeline stepper (Research -> Writing -> Critique ->
  Optimization -> Scheduling) that updates while the crew runs in the
  background.
- A prominent final answer card once the crew finishes.
- An OPTIONAL verbose section (toggle) showing each agent's raw output
  (the agentic review chain) plus the full console log.
"""

from __future__ import annotations

import sys

import streamlit as st

st.set_page_config(
    page_title="LinkedIn Content Manager",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CrewAI/ChromaDB currently crash on Python 3.14 (pydantic v1 ConfigError).
# Streamlit Cloud ignores .python-version; the Python version must be 3.12
# in App settings → Advanced settings. Changing it requires delete + redeploy.
if sys.version_info >= (3, 14):
    st.error(
        f"This Cloud runtime is Python {sys.version.split()[0]}, which CrewAI "
        "does not support.\n\n"
        "Fix: Streamlit Cloud → **Manage app** → delete this app → **Deploy** again, "
        "then in **Advanced settings** set **Python version to 3.12**. "
        "Reboot alone will not change Python."
    )
    st.stop()

import contextlib
import io
import os
import time
from threading import Thread
from typing import Optional

from crewai import Crew, Process
from dotenv import load_dotenv

from app_config import ConfigurationError, validate_environment
from linkedin_content_manager import (
    build_llm,
    create_agents,
    create_tasks,
)

load_dotenv()


def apply_streamlit_secrets() -> None:
    """Copy Streamlit secrets into environment variables for CrewAI/Groq."""
    try:
        for key in st.secrets:
            value = st.secrets[key]
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if not os.getenv(str(nested_key)):
                        os.environ[str(nested_key)] = str(nested_value)
            elif not os.getenv(str(key)):
                os.environ[str(key)] = str(value)
    except FileNotFoundError:
        pass
    except Exception:
        pass


apply_streamlit_secrets()

STAGE_DEFS = [
    {"key": "research", "label": "Research", "icon": "🔍", "agent": "Trend Researcher"},
    {"key": "writing", "label": "Writing", "icon": "✍️", "agent": "Content Writer"},
    {"key": "critique", "label": "Critique", "icon": "🧐", "agent": "Content Critic"},
    {"key": "optimization", "label": "Optimization", "icon": "✨", "agent": "Post Optimizer"},
    {"key": "scheduling", "label": "Scheduling", "icon": "📅", "agent": "Publishing Strategist"},
]

EXAMPLE_TOPICS = [
    "Multi-agent AI systems",
    "Remote team leadership",
    "SaaS growth strategy",
    "Career advice for engineers",
    "AI in healthcare",
]


def rerun() -> None:
    """Compatibility wrapper across Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - older Streamlit fallback
        st.experimental_rerun()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: radial-gradient(1200px 600px at 10% -10%, #eef4ff 0%, #f7f9fc 45%, #ffffff 100%); }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .hero {
            background: linear-gradient(120deg, #0a66c2, #6a3fd8, #0a66c2);
            background-size: 220% 220%;
            animation: gradientShift 9s ease infinite;
            padding: 2.2rem 2.4rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 1.4rem;
            box-shadow: 0 14px 34px rgba(10, 102, 194, 0.30);
        }
        .hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px; }
        .hero p { margin-top: .5rem; opacity: .92; font-size: 1.02rem; }
        .hero .badge {
            display: inline-block; margin-top: .9rem; padding: .3rem .8rem;
            background: rgba(255,255,255,0.16); border-radius: 999px; font-size: .78rem;
            border: 1px solid rgba(255,255,255,0.3); letter-spacing: .3px;
        }

        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .card {
            animation: fadeSlideIn .55s ease-out;
            background: #ffffff;
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            border: 1px solid #eaecef;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.1rem;
        }
        .card h3 { margin-top: 0; }

        .chip-row { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: .6rem; }

        div.stButton > button {
            background: linear-gradient(120deg, #0a66c2, #0043a8);
            color: white; border: none; border-radius: 10px; padding: .65rem 1.4rem;
            font-weight: 700; transition: transform .18s ease, box-shadow .18s ease;
            box-shadow: 0 4px 14px rgba(10,102,194,0.28);
        }
        div.stButton > button:hover { transform: translateY(-2px) scale(1.015); box-shadow: 0 10px 22px rgba(10,102,194,0.38); }
        div.stButton > button:disabled { opacity: 0.55; transform: none; box-shadow: none; }

        .stepper {
            display: flex; align-items: center; justify-content: space-between;
            margin: 1.2rem 0 .4rem 0;
        }
        .step { display: flex; flex-direction: column; align-items: center; flex: 0 0 auto; width: 92px; }
        .step-circle {
            width: 54px; height: 54px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.5rem; background: #eef1f5; color: #8a94a6;
            border: 2px solid rgba(0,0,0,0.05); transition: all .35s ease;
        }
        @keyframes pulseGlow {
            0% { box-shadow: 0 0 0 0 rgba(10,102,194,.55); }
            70% { box-shadow: 0 0 0 16px rgba(10,102,194,0); }
            100% { box-shadow: 0 0 0 0 rgba(10,102,194,0); }
        }
        .step-circle.active { background: #0a66c2; color: white; animation: pulseGlow 1.5s infinite; transform: scale(1.08); }
        .step-circle.done { background: #12b76a; color: white; }
        .step-label { text-align: center; font-size: .78rem; margin-top: .45rem; color: #4a5468; font-weight: 700; }
        .step-line { flex: 1 1 auto; height: 4px; margin: 0 4px 30px 4px; border-radius: 3px; background: #e2e5ea; overflow: hidden; }
        .step-line.done { background: linear-gradient(90deg, #12b76a, #0a66c2); }

        @keyframes shimmer { 0% { background-position: -420px 0; } 100% { background-position: 420px 0; } }
        .progressbar-outer { height: 10px; border-radius: 6px; background: #eef1f5; overflow: hidden; margin: .3rem 0 1rem 0; }
        .progressbar-inner {
            height: 100%; border-radius: 6px;
            background: linear-gradient(90deg, #0a66c2 25%, #7aa9f0 50%, #0a66c2 75%);
            background-size: 820px 100%; animation: shimmer 2.1s linear infinite;
            transition: width .5s ease;
        }

        .status-line { font-size: .96rem; padding: .55rem .9rem; border-radius: 10px; background: #eef4ff; color: #0a3a75; font-weight: 600; }
        .status-line.success { background: #e9fbf1; color: #08703f; }
        .status-line.error { background: #fdecec; color: #9c1c1c; }

        .env-ok { color: #0e9f6e; font-weight: 700; }
        .env-bad { color: #d92d20; font-weight: 700; }

        footer, #MainMenu { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(placeholder, current_index: Optional[int], completed: set) -> None:
    parts = ["<div class='stepper'>"]
    for i, stage in enumerate(STAGE_DEFS):
        if i in completed:
            circle_cls, icon = "done", "✅"
        elif i == current_index:
            circle_cls, icon = "active", stage["icon"]
        else:
            circle_cls, icon = "", stage["icon"]
        parts.append(
            f"<div class='step'>"
            f"<div class='step-circle {circle_cls}'>{icon}</div>"
            f"<div class='step-label'>{stage['label']}</div>"
            f"</div>"
        )
        if i < len(STAGE_DEFS) - 1:
            line_cls = "done" if i in completed else ""
            parts.append(f"<div class='step-line {line_cls}'></div>")
    parts.append("</div>")
    placeholder.markdown("".join(parts), unsafe_allow_html=True)


def render_progress_bar(placeholder, percent: int) -> None:
    placeholder.markdown(
        f"""
        <div class="progressbar-outer">
            <div class="progressbar-inner" style="width:{percent}%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_completed_stages(log_text: str) -> set:
    completed = set()
    for i, stage in enumerate(STAGE_DEFS):
        if f"Completed Stage: {stage['label']}" in log_text:
            completed.add(i)
    return completed


def run_pipeline(topic: str, shared: dict) -> None:
    """Runs entirely in a background thread. Only mutates a plain dict."""
    buffer: io.StringIO = shared["buffer"]
    try:
        with contextlib.redirect_stdout(buffer):
            llm = build_llm()
            agents = create_agents(llm)
            tasks = create_tasks(agents, topic)
            crew = Crew(
                agents=list(agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=True,
                memory=True,
            )
            result = crew.kickoff(inputs={"topic": topic})

        shared["result"] = getattr(result, "raw", None) or str(result)

        stage_outputs = {}
        tasks_output = getattr(result, "tasks_output", None) or []
        for i, task_output in enumerate(tasks_output):
            stage_outputs[i] = getattr(task_output, "raw", None) or str(task_output)
        shared["stage_outputs"] = stage_outputs
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI
        shared["error"] = str(exc)
    finally:
        shared["running"] = False


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
inject_css()

for key, default in {
    "is_running": False,
    "shared": None,
    "last_result": None,
    "last_stage_outputs": {},
    "last_log": "",
    "last_error": None,
    "topic_input": "",
    "audience_input": "",
}.items():
    st.session_state.setdefault(key, default)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Environment Status")
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_serper = bool(os.getenv("SERPER_API_KEY"))
    st.markdown(
        f"- OPENAI_API_KEY (Groq): {'<span class=\"env-ok\">✔ Loaded</span>' if has_openai else '<span class=\"env-bad\">✘ Missing</span>'}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- SERPER_API_KEY: {'<span class=\"env-ok\">✔ Loaded</span>' if has_serper else '<span class=\"env-bad\">✘ Missing</span>'}",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 🔬 Verbose Output")
    show_verbose = st.checkbox(
        "Show detailed agent logs",
        value=False,
        help="Reveal each agent's raw output plus the full execution log.",
    )

    st.markdown("---")
    st.markdown("### 🧠 Pipeline")
    for stage in STAGE_DEFS:
        st.markdown(f"{stage['icon']} **{stage['label']}** — {stage['agent']}")

    st.markdown("---")
    if st.button("🔁 Reset dashboard"):
        st.session_state.is_running = False
        st.session_state.shared = None
        st.session_state.last_result = None
        st.session_state.last_stage_outputs = {}
        st.session_state.last_log = ""
        st.session_state.last_error = None
        rerun()

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
        <h1>🚀 Autonomous LinkedIn Content Manager</h1>
        <p>Five specialized AI agents research, write, critique, optimize, and schedule your next LinkedIn post — end to end.</p>
        <span class="badge">CrewAI · Sequential · Memory-Enabled · Agentic Review Chain</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Input card
# --------------------------------------------------------------------------- #
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("#### 🎯 What should the crew work on?")

st.markdown("<div class='chip-row'>", unsafe_allow_html=True)
chip_cols = st.columns(len(EXAMPLE_TOPICS))
for col, example in zip(chip_cols, EXAMPLE_TOPICS):
    if col.button(example, key=f"chip_{example}", disabled=st.session_state.is_running):
        st.session_state.topic_input = example
        rerun()
st.markdown("</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    topic = st.text_input(
        "Topic / Niche",
        key="topic_input",
        placeholder="e.g. Multi-agent AI systems",
        disabled=st.session_state.is_running,
    )
with col2:
    audience = st.text_input(
        "Target Audience / Angle (optional)",
        key="audience_input",
        placeholder="e.g. CTOs at mid-size startups",
        disabled=st.session_state.is_running,
    )

env_ready = True
try:
    validate_environment()
except ConfigurationError as exc:
    env_ready = False
    st.warning(f"⚠️ {exc}")

generate_clicked = st.button(
    "🚀 Generate LinkedIn Post",
    disabled=st.session_state.is_running or not env_ready or not topic.strip(),
)
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Kick off a new run
# --------------------------------------------------------------------------- #
if generate_clicked and topic.strip():
    combined_topic = topic.strip()
    if audience.strip():
        combined_topic = f"{combined_topic} — tailored for {audience.strip()}"

    shared = {
        "buffer": io.StringIO(),
        "result": None,
        "error": None,
        "running": True,
        "stage_outputs": {},
    }
    st.session_state.shared = shared
    st.session_state.is_running = True
    st.session_state.last_result = None
    st.session_state.last_stage_outputs = {}
    st.session_state.last_log = ""
    st.session_state.last_error = None

    thread = Thread(target=run_pipeline, args=(combined_topic, shared), daemon=True)
    thread.start()
    rerun()

# --------------------------------------------------------------------------- #
# Live pipeline view (blocks this script run until the crew finishes)
# --------------------------------------------------------------------------- #
if st.session_state.is_running and st.session_state.shared is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 🛰️ Live Pipeline")

    stepper_ph = st.empty()
    progress_ph = st.empty()
    status_ph = st.empty()
    log_ph = st.empty()

    shared = st.session_state.shared
    start_time = time.time()

    while True:
        log_text = shared["buffer"].getvalue()
        completed = compute_completed_stages(log_text)
        current = next((i for i in range(len(STAGE_DEFS)) if i not in completed), None)

        render_stepper(stepper_ph, current, completed)
        render_progress_bar(progress_ph, int(len(completed) / len(STAGE_DEFS) * 100))

        elapsed = time.time() - start_time
        if current is not None:
            stage_info = STAGE_DEFS[current]
            status_ph.markdown(
                f"<div class='status-line'>⏳ <b>{stage_info['agent']}</b> is working on "
                f"<b>{stage_info['label']}</b>… ({elapsed:0.1f}s elapsed)</div>",
                unsafe_allow_html=True,
            )
        else:
            status_ph.markdown(
                f"<div class='status-line'>🧵 Finalizing pipeline output… ({elapsed:0.1f}s elapsed)</div>",
                unsafe_allow_html=True,
            )

        if show_verbose:
            log_ph.code(log_text[-8000:] or "Waiting for agent output…", language="text")
        else:
            log_ph.empty()

        if not shared.get("running"):
            break
        time.sleep(0.4)

    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.last_error = shared.get("error")
    st.session_state.last_result = shared.get("result")
    st.session_state.last_stage_outputs = shared.get("stage_outputs", {})
    st.session_state.last_log = shared["buffer"].getvalue()
    st.session_state.is_running = False
    st.session_state.shared = None

    if not st.session_state.last_error:
        st.balloons()
    rerun()

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if st.session_state.last_error:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='status-line error'>❌ Pipeline failed: {st.session_state.last_error}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.last_result:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='status-line success'>🎉 Your publish-ready LinkedIn post is ready!</div>",
        unsafe_allow_html=True,
    )
    st.markdown("#### ✅ Final Answer")
    st.code(st.session_state.last_result, language="markdown")
    st.download_button(
        "⬇️ Download final brief (.md)",
        data=st.session_state.last_result,
        file_name="linkedin_publishing_brief.md",
        mime="text/markdown",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if show_verbose:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🔬 Agentic Review Chain (each agent's output)")
        tab_labels = [f"{s['icon']} {s['label']}" for s in STAGE_DEFS]
        tabs = st.tabs(tab_labels + ["🧾 Full Execution Log"])
        for i, tab in enumerate(tabs[:-1]):
            with tab:
                output = st.session_state.last_stage_outputs.get(i)
                if output:
                    st.markdown(output)
                else:
                    st.info("Output not captured for this stage.")
        with tabs[-1]:
            st.code(st.session_state.last_log or "No log captured.", language="text")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        "Enter a topic above and click **Generate LinkedIn Post** to watch the five agents "
        "research, write, critique, optimize, and schedule your content in real time."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#94a3b8; font-size:.8rem; margin-top:1.5rem;'>"
    "Built with CrewAI · Groq · Streamlit</p>",
    unsafe_allow_html=True,
)
