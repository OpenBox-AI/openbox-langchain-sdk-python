#!/usr/bin/env python3
"""
Content Builder Agent (LangChain + OpenBox Governance)

A content writer agent configured entirely through files on disk:
- AGENTS.md defines brand voice and style guide
- skills/ provides specialized workflows (blog posts, social media)
- subagents.yaml defines the researcher subagent configuration

Usage:
    uv run python content_writer.py "Write a blog post about AI agents"
    uv run python content_writer.py "Create a LinkedIn post about prompt engineering"
"""

import logging
import os
import sys
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from openbox_langchain import create_openbox_langchain_middleware

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("openbox_langchain").setLevel(logging.DEBUG)

EXAMPLE_DIR = Path(__file__).parent
console = Console()


# ═══════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════


@tool
def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
) -> dict:
    """Search the web for current information.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: "general" for most queries, "news" for current events

    Returns:
        Search results with titles, URLs, and content excerpts.
    """
    try:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return {"error": "TAVILY_API_KEY not set"}

        client = TavilyClient(api_key=api_key)
        return client.search(query, max_results=max_results, topic=topic)
    except Exception as e:
        return {"error": f"Search failed: {e}"}


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file. Creates parent directories as needed.

    Args:
        file_path: Relative path from the project root (e.g., 'blogs/my-post/post.md')
        content: The content to write
    """
    try:
        path = EXAMPLE_DIR / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return f"File written to {path}"
    except Exception as e:
        return f"Error: {e}"


@tool
def read_file(file_path: str) -> str:
    """Read content from a file.

    Args:
        file_path: Relative path from the project root (e.g., 'research/topic.md')
    """
    try:
        path = EXAMPLE_DIR / file_path
        return path.read_text()
    except Exception as e:
        return f"Error: {e}"


@tool
def generate_cover(prompt: str, slug: str) -> str:
    """Generate a cover image for a blog post.

    Args:
        prompt: Detailed description of the image to generate.
        slug: Blog post slug. Image saves to blogs/<slug>/hero.png
    """
    try:
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )

        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                output_path = EXAMPLE_DIR / "blogs" / slug / "hero.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                return f"Image saved to {output_path}"

        return "No image generated"
    except Exception as e:
        return f"Error: {e}"


@tool
def generate_social_image(prompt: str, platform: str, slug: str) -> str:
    """Generate an image for a social media post.

    Args:
        prompt: Detailed description of the image to generate.
        platform: Either "linkedin" or "tweets"
        slug: Post slug. Image saves to <platform>/<slug>/image.png
    """
    try:
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )

        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                output_path = EXAMPLE_DIR / platform / slug / "image.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(output_path))
                return f"Image saved to {output_path}"

        return "No image generated"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════
# Skill & subagent loading
# ═══════════════════════════════════════════════════════════════════


def load_skills(skills_dir: Path) -> str:
    """Load all SKILL.md files and return their combined content."""
    skills_text = []
    for skill_file in sorted(skills_dir.rglob("SKILL.md")):
        skills_text.append(skill_file.read_text())
    return "\n\n---\n\n".join(skills_text)


def load_subagent_config(config_path: Path) -> dict:
    """Load subagent definitions from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════
# Researcher subagent
# ═══════════════════════════════════════════════════════════════════


def run_researcher(topic: str, save_to: str) -> str:
    """Create and run the researcher subagent synchronously."""
    config = load_subagent_config(EXAMPLE_DIR / "subagents.yaml")
    researcher_spec = config.get("researcher", {})
    system_prompt = researcher_spec.get("system_prompt", "You are a research assistant.")
    model_name = researcher_spec.get("model", "openai:gpt-4o-mini")

    model = init_chat_model(model_name, temperature=0)
    researcher = create_agent(
        model=model,
        tools=[web_search, write_file],
        system_prompt=system_prompt,
    )

    task_description = f"Research {topic} and save findings to {save_to}"
    result = researcher.invoke({"messages": [("user", task_description)]})

    # Extract last AI message as summary
    from langchain_core.messages import AIMessage

    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if isinstance(content, str):
                return f"Research complete. Summary: {content[:300]}..."
            break

    return "Research complete."


@tool
def research(topic: str, save_to: str) -> str:
    """Delegate research to the researcher subagent. ALWAYS use this first before writing any content.

    Args:
        topic: The topic to research (be specific)
        save_to: File path to save research results (e.g., 'research/ai-agents.md')
    """
    console.print(f"  [bold magenta]>> Researching:[/] {topic[:60]}...")
    result = run_researcher(topic, save_to)
    console.print("  [green]✓ Research complete[/]")
    return result


# ═══════════════════════════════════════════════════════════════════
# Main agent
# ═══════════════════════════════════════════════════════════════════


def create_content_writer():
    """Create a content writer agent configured by filesystem files.

    Returns:
        Tuple of (agent, openbox_handler)
    """
    # Load memory (brand voice & style guide)
    agents_md = (EXAMPLE_DIR / "AGENTS.md").read_text()

    # Load skills (blog-post, social-media workflows)
    skills_text = load_skills(EXAMPLE_DIR / "skills")

    # Build system prompt combining memory + skills
    system_prompt = f"""{agents_md}

## Available Skills (loaded from skills/)

{skills_text}

## Tool Usage Instructions

- Use the `research` tool FIRST before writing any content
- Use `write_file` to save content to the appropriate directory
- Use `read_file` to read research results before writing
- Use `generate_cover` for blog post cover images (saves to blogs/<slug>/hero.png)
- Use `generate_social_image` for social media images (saves to <platform>/<slug>/image.png)
"""

    # Create OpenBox governance middleware
    middleware = create_openbox_langchain_middleware(
        api_url=os.environ["OPENBOX_URL"],
        api_key=os.environ["OPENBOX_API_KEY"],
        agent_name=os.environ.get("OPENBOX_AGENT_NAME", "ContentWriter"),
        tool_type_map={"web_search": "http"},
    )

    model = init_chat_model("openai:gpt-4o-mini", temperature=0)

    agent = create_agent(
        model=model,
        tools=[research, write_file, read_file, generate_cover, generate_social_image],
        system_prompt=system_prompt,
        middleware=[middleware],
    )

    return agent


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════


def main():
    """Run the content writer agent with progress output."""
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "Write a blog post about how AI agents are transforming software development"

    console.print()
    console.print("[bold blue]Content Builder Agent[/] [dim](LangChain + OpenBox)[/]")
    console.print(f"[dim]Task: {task}[/]")
    console.print()

    agent = create_content_writer()

    # Stream with governance middleware (injected via create_agent)
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from rich.markdown import Markdown
    from rich.panel import Panel

    printed_count = 0
    for chunk in agent.stream(
        {"messages": [("user", task)]},
        stream_mode="values",
    ):
        if "messages" in chunk:
            messages = chunk["messages"]
            for msg in messages[printed_count:]:
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content
                    if isinstance(content, list):
                        content = "\n".join(
                            p.get("text", "") for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        )
                    if content.strip():
                        console.print(Panel(Markdown(content), title="Agent", border_style="green"))
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get("name", "")
                        if name == "research":
                            console.print(f"  >> Research: {tc.get('args', {}).get('topic', '')[:60]}...")
                        elif name == "write_file":
                            console.print(f"  >> Writing: {tc.get('args', {}).get('file_path', '')}")
            printed_count = len(messages)

    console.print()
    console.print("[bold green]✓ Done![/]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
