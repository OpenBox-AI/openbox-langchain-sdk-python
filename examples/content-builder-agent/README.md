# Content Builder Agent

A content writing agent for writing blog posts, LinkedIn posts, and tweets with cover images included.

**This example demonstrates how to use LangChain with OpenBox governance through three filesystem primitives:**
- **Memory** (`AGENTS.md`) – persistent context like brand voice and style guidelines
- **Skills** (`skills/*/SKILL.md`) – workflows for specific tasks, loaded on demand
- **Subagents** (`subagents.yaml`) – specialized agents for delegated tasks like research

The `content_writer.py` script shows how to combine these into a working agent using LangChain with OpenBox governance.

## Quick Start

```bash
# Set API keys
export OPENAI_API_KEY="..."
export OPENBOX_URL="https://core.openbox.ai"
export OPENBOX_API_KEY="obx_live_..."
export OPENBOX_WORKLOAD_PRIVATE_KEY="$(cat /path/to/workload-private-key.pem)"
export GOOGLE_API_KEY="..."      # For image generation
export TAVILY_API_KEY="..."      # For web search (optional)

# Run with this repository's SDK and its published base-SDK dependency
cd examples/content-builder-agent
uv run --with-editable ../.. python content_writer.py "Write a blog post about prompt engineering"
```

**More examples:**
```bash
uv run --with-editable ../.. python content_writer.py "Create a LinkedIn post about AI agents"
uv run --with-editable ../.. python content_writer.py "Write a Twitter thread about the future of coding"
```

Use the PKCS8 PEM RSA key associated with the agent's active IAM v3 workload
identity. The middleware reads `OPENBOX_LANGCHAIN_WORKLOAD_PRIVATE_KEY` before
`OPENBOX_WORKLOAD_PRIVATE_KEY` and validates the identity at startup. The base SDK
handles Core bootstrap, Keycloak token exchange, and authenticated v3 requests.

The `--with-editable ../..` option loads the SDK under development from this
repository. Its `openbox-sdk-python>=1.3.1` dependency comes from PyPI. This makes
the new middleware configuration available before a LangChain SDK release.

For an existing DID-signed agent, supply `OPENBOX_AGENT_DID` and
`OPENBOX_AGENT_PRIVATE_KEY` instead. API-key-only agents can omit identity keys
when their Core configuration permits it.

## How It Works

The agent is configured by files on disk, not code:

```
content-builder-agent/
├── AGENTS.md                    # Brand voice & style guide
├── subagents.yaml               # Subagent definitions
├── skills/
│   ├── blog-post/
│   │   └── SKILL.md             # Blog writing workflow
│   └── social-media/
│       └── SKILL.md             # Social media workflow
└── content_writer.py            # Wires it together (includes tools)
```

| File | Purpose | When Loaded |
|------|---------|-------------|
| `AGENTS.md` | Brand voice, tone, writing standards | Always (system prompt) |
| `subagents.yaml` | Research subagent config | When research tool runs |
| `skills/*/SKILL.md` | Content-specific workflows | Always (appended to system prompt) |

## Architecture

```python
# Load memory + skills into system prompt
agents_md = Path("AGENTS.md").read_text()
skills_text = load_skills(Path("skills/"))

# Create OpenBox governance middleware
middleware = create_openbox_langchain_middleware(
    api_url=os.environ["OPENBOX_URL"],
    api_key=os.environ["OPENBOX_API_KEY"],
    # Workload credentials are resolved from the environment.
    agent_name="ContentWriter",
)

# Create agent with middleware
agent = create_agent(
    model=init_chat_model("openai:gpt-4o-mini"),
    tools=[research, write_file, read_file, generate_cover, generate_social_image],
    system_prompt=agents_md + skills_text,
    middleware=[middleware],
)

# Run with governance applied automatically
result = agent.invoke({"messages": [("user", task)]})
```

**Flow:**
1. Agent receives task → loads relevant skill (blog-post or social-media)
2. Calls `research` tool → spawns researcher subagent → saves to `research/`
3. Reads research findings → writes content → saves to `blogs/` or `linkedin/`
4. Generates cover image with Gemini → saves alongside content

## Output

```
blogs/
└── prompt-engineering/
    ├── post.md       # Blog content
    └── hero.png      # Generated cover image

linkedin/
└── ai-agents/
    ├── post.md       # Post content
    └── image.png     # Generated image

research/
└── prompt-engineering.md   # Research notes
```

## Customizing

**Change the voice:** Edit `AGENTS.md` to modify brand tone and style.

**Add a content type:** Create `skills/<name>/SKILL.md` with YAML frontmatter:
```yaml
---
name: newsletter
description: Use this skill when writing email newsletters
---
# Newsletter Skill
...
```

**Add a subagent:** Add to `subagents.yaml`:
```yaml
editor:
  description: Review and improve drafted content
  model: openai:gpt-4o-mini
  system_prompt: |
    You are an editor. Review the content and suggest improvements...
  tools: []
```

**Add a tool:** Define it in `content_writer.py` with the `@tool` decorator and add to the `create_agent(tools=[...])` list.

## Key Differences from deepagents Version

| Aspect | deepagents | LangChain |
|--------|-----------|-----------|
| Agent creation | `create_deep_agent()` | `create_agent()` from langchain |
| Memory/Skills | Native middleware | Loaded manually into system prompt |
| Subagents | Native `task` tool | Custom `research` tool spawning sub-agent |
| Governance | `create_openbox_middleware()` | `create_openbox_langchain_middleware()` via AgentMiddleware |
| Execution | Async (`astream`) | Sync (`stream`) |
| File I/O | `FilesystemBackend` | Custom `write_file`/`read_file` tools |

## Requirements

- Python 3.11+
- `OPENAI_API_KEY` - For the main agent (GPT-4o-mini)
- `OPENBOX_URL` + `OPENBOX_API_KEY` - For OpenBox governance
- `OPENBOX_WORKLOAD_PRIVATE_KEY` - PKCS8 PEM RSA key for IAM v3 workload authentication
- `GOOGLE_API_KEY` - For image generation (Gemini's Imagen)
- `TAVILY_API_KEY` - For web search (optional, research still works without it)
