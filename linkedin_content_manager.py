"""
Autonomous LinkedIn Content Manager using CrewAI.

Pipeline:
Research -> Write -> Critique -> Optimize -> Schedule
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from dotenv import load_dotenv

from crewai import Agent, Crew, LLM, Process, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Stage:
    """Represents a named processing stage for pretty console output."""

    key: str
    title: str


STAGES = [
    Stage("research", "1/5 RESEARCH"),
    Stage("writing", "2/5 WRITING"),
    Stage("critique", "3/5 CRITIQUE"),
    Stage("optimization", "4/5 OPTIMIZATION"),
    Stage("scheduling", "5/5 SCHEDULING"),
]


def print_banner(title: str) -> None:
    """Print a visually clear section banner."""
    line = "=" * 90
    print(f"\n{line}\n{title}\n{line}")


def print_sub_banner(title: str) -> None:
    """Print a smaller subsection banner."""
    line = "-" * 90
    print(f"\n{line}\n{title}\n{line}")


def validate_environment() -> None:
    """
    Ensure all required environment variables are available before running.
    """
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("SERPER_API_KEY"):
        missing.append("SERPER_API_KEY")

    if missing:
        joined = ", ".join(missing)
        raise ConfigurationError(
            "Missing required environment variable(s): "
            f"{joined}. Set them in a local .env file, or in Streamlit Cloud "
            "App settings → Secrets (see .streamlit/secrets.toml.example)."
        )


def build_llm() -> LLM:
    """
    Build the shared Groq-backed LLM.

    CrewAI uses an OpenAI-compatible client, so we provide Groq's OpenAI API base URL
    while loading the key from OPENAI_API_KEY as requested.
    """
    # CrewAI may normalize provider/model prefixes before sending requests.
    # `openai/openai/gpt-oss-20b` reliably resolves to `openai/gpt-oss-20b`
    # against Groq's OpenAI-compatible endpoint in this environment.
    default_model = "openai/openai/gpt-oss-20b"
    return LLM(
        model=os.getenv("GROQ_MODEL", default_model),
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
        temperature=0.7,
    )


def stage_callback(stage_name: str) -> Callable:
    """
    Create a task callback that prints structured stage output.

    Callback signatures can vary by CrewAI version, so we accept *args/**kwargs
    and safely extract task output text where possible.
    """

    def _callback(*args, **kwargs) -> None:
        print_sub_banner(f"Completed Stage: {stage_name}")
        candidate_values = list(args) + list(kwargs.values())
        rendered = None
        for value in candidate_values:
            # TaskOutput often exposes `.raw`; fallback to string conversion.
            raw = getattr(value, "raw", None)
            if raw:
                rendered = str(raw)
                break
            if isinstance(value, str) and value.strip():
                rendered = value
                break
        if rendered:
            print(rendered.strip())
        else:
            print("Stage completed successfully.")

    return _callback


def create_agents(llm: LLM) -> Dict[str, Agent]:
    """
    Create exactly 5 specialized agents for the pipeline.
    """
    trend_researcher = Agent(
        role="LinkedIn Trend Researcher",
        goal="Research latest trending topics, hashtags, and content themes for a given niche",
        backstory=(
            "Expert social media researcher who monitors LinkedIn trends, viral posts, "
            "and industry news; knows what drives engagement on LinkedIn."
        ),
        tools=[SerperDevTool(), ScrapeWebsiteTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    content_writer = Agent(
        role="LinkedIn Content Writer",
        goal="Write engaging, high-quality LinkedIn posts based on research provided",
        backstory=(
            "Seasoned LinkedIn ghostwriter for industry leaders; expert in LinkedIn "
            "algorithm, hook writing, storytelling, CTA placement, conversational "
            "professional tone."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    content_critic = Agent(
        role="Content Quality Critic",
        goal=(
            "Review LinkedIn posts and provide detailed constructive feedback on "
            "engagement potential, tone, structure, clarity, hook strength, and "
            "CTA effectiveness"
        ),
        backstory=(
            "Harsh but fair editor with thousands of LinkedIn post reviews; distinguishes "
            "posts that get 10 likes from those with 10k+ impressions; delivers specific, "
            "actionable feedback."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    content_optimizer = Agent(
        role="LinkedIn Post Optimizer",
        goal="Rewrite posts incorporating critic feedback to maximize LinkedIn engagement",
        backstory=(
            "LinkedIn growth expert and copywriter; master of formatting (short lines, "
            "strategic breaks, emoji usage, hashtag optimization, hook patterns, "
            "mobile readability)."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    scheduling_agent = Agent(
        role="LinkedIn Publishing Strategist",
        goal=(
            "Determine optimal posting time, finalize formatting with hashtags, and "
            "create publishing-ready output with scheduling recommendations"
        ),
        backstory=(
            "LinkedIn analytics expert; understands optimal posting times by industry, "
            "audience timezone, and day of week."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return {
        "researcher": trend_researcher,
        "writer": content_writer,
        "critic": content_critic,
        "optimizer": content_optimizer,
        "scheduler": scheduling_agent,
    }


def create_tasks(agents: Dict[str, Agent], topic: str) -> list[Task]:
    """
    Create exactly 5 sequential tasks mapped to the required agents.

    Each downstream task receives explicit CrewAI context from earlier tasks.
    This makes every agent review and build on prior agent output, while the
    sequential process still preserves the strict research -> schedule order.
    """
    research_task = Task(
        description=(
            f"Research latest trends, viral content patterns, and hot topics on LinkedIn "
            f"for the niche: {topic}. Identify 3-5 trending angles, relevant hashtags, "
            "and content hooks currently performing well."
        ),
        expected_output=(
            "Structured research brief with trending topics, suggested angles, "
            "top-performing hashtags, and content hook ideas."
        ),
        agent=agents["researcher"],
        callback=stage_callback("Research"),
    )

    writing_task = Task(
        description=(
            f"Review the research brief from the Trend Researcher Agent, then write a "
            f"compelling LinkedIn post about {topic}. "
            "Include a strong hook (first 2 lines), storytelling or value-driven body, "
            "clear CTA, and 150-300 words. Use trending angles and hooks from research."
        ),
        expected_output=(
            "Complete LinkedIn post draft with hook, body, CTA, and suggested hashtags."
        ),
        agent=agents["writer"],
        context=[research_task],
        callback=stage_callback("Writing"),
    )

    critique_task = Task(
        description=(
            "Review the Content Writer Agent's LinkedIn post draft in the provided context. "
            'Critically evaluate hook strength (will people click "see more"?), storytelling '
            "quality, engagement potential, CTA effectiveness, tone consistency, LinkedIn "
            "formatting, and viral potential. Provide a score out of 10 and specific "
            "improvement suggestions."
        ),
        expected_output=(
            "Detailed critique with scores, strengths, weaknesses, and specific actionable "
            "improvement suggestions."
        ),
        agent=agents["critic"],
        context=[writing_task],
        callback=stage_callback("Critique"),
    )

    optimization_task = Task(
        description=(
            "Review both the original LinkedIn post from the Content Writer Agent and the "
            "critic's feedback from the Content Critic Agent. Rewrite the post by incorporating "
            "all feedback. Improve the hook, tighten copy, optimize formatting (short lines, "
            "line breaks, strategic emoji), strengthen CTA, and optimize hashtags. Produce "
            "final publish-ready version."
        ),
        expected_output=(
            "Final, polished, publish-ready LinkedIn post with optimized formatting, hashtags, "
            "and CTA."
        ),
        agent=agents["optimizer"],
        context=[writing_task, critique_task],
        callback=stage_callback("Optimization"),
    )

    scheduling_task = Task(
        description=(
            f"Review the optimized final post from the Content Optimizer Agent and analyze "
            f"the target audience for {topic}. Recommend best day and time to publish "
            "(with timezone), provide final formatted post ready for LinkedIn copy-paste, "
            "and include brief with hashtag strategy and first-hour engagement tips."
        ),
        expected_output=(
            "Complete publishing brief with recommended posting time, final formatted post, "
            "hashtag list, and first-hour engagement strategy."
        ),
        agent=agents["scheduler"],
        context=[optimization_task],
        callback=stage_callback("Scheduling"),
    )

    return [
        research_task,
        writing_task,
        critique_task,
        optimization_task,
        scheduling_task,
    ]


def run(topic: str) -> None:
    """Run the full crew pipeline for a given topic."""
    print_banner("Autonomous LinkedIn Content Manager")
    print(f"Topic/Niche: {topic}")
    print("Pipeline: Research -> Writing -> Critique -> Optimization -> Scheduling")

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

    # Show clear stage separators before CrewAI starts verbose execution.
    for stage in STAGES:
        print_sub_banner(f"Queued Stage: {stage.title}")

    try:
        result = crew.kickoff(inputs={"topic": topic})
    except Exception as exc:
        raise RuntimeError(
            "Crew execution failed. Check API keys, network connectivity, model availability, "
            f"or tool quotas. Root cause: {exc}"
        ) from exc

    print_banner("FINAL PUBLISHING OUTPUT")

    # CrewAI output structures can vary; normalize for reliable display.
    final_output = getattr(result, "raw", None)
    if final_output:
        print(final_output)
    else:
        print(str(result))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for non-interactive execution."""
    parser = argparse.ArgumentParser(
        description="Run the Autonomous LinkedIn Content Manager CrewAI pipeline."
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="LinkedIn topic or niche to generate content for.",
    )
    return parser.parse_args()


def main() -> int:
    """Application entry point with resilient error handling."""
    load_dotenv()

    try:
        validate_environment()
    except ConfigurationError as exc:
        print_banner("CONFIGURATION ERROR")
        print(exc)
        return 1

    args = parse_args()
    topic: Optional[str] = args.topic

    if not topic:
        topic = input("Enter the LinkedIn topic/niche: ").strip()

    if not topic:
        print_banner("INPUT ERROR")
        print("Topic cannot be empty.")
        return 1

    try:
        run(topic)
        return 0
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        return 130
    except Exception as exc:
        print_banner("RUNTIME ERROR")
        # Wrap text for cleaner terminal readability.
        print(textwrap.fill(str(exc), width=100))
        return 1


if __name__ == "__main__":
    sys.exit(main())
