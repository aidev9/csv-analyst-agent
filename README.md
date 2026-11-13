# CSV Analyst

A conversational data analyst agent, built with the [Mistral Agents SDK](https://docs.mistral.ai/studio/agents/overview), that answers plain-language questions about the support-ticket file `data/ams-tickets.csv` in the user's Google Drive.

## What it does

- Reads `data/ams-tickets.csv` through the `google_drive_mcp` connector (as the session user, read-only).
- Loads the CSV with a `post_tool_call` hook (`DriveTicketsLoader`): when the connector's `download_file_content` returns, the hook parses the file into the session's dataset and hands the model only a summary. The model never reads or re-sends the raw file, which keeps model calls short.
- Answers questions with real numbers computed by three fixed tools only — `describe_columns`, `filter_rows`, `group_by_aggregate`. No arbitrary code execution (programmatic tool calling is disabled).
- Specialized in ticket analysis: root-cause analysis, top fix priorities, automation opportunities, and short/medium/long-term plans.
- Sessions are durable, so follow-up questions in the same conversation remember the file and the discussion.

## Run locally

```bash
uv sync
uv run --env-file .env agents chat
```

## Deploy (Mistral-managed)

```bash
git push origin main
uv run --env-file .env agents deploy
```

The deploy registers the agent as `csv-analyst` on a hardened managed deployment (the Google Drive connector acts on behalf of the session user). Chat with it in Vibe Code Web at `https://chat.mistral.ai/managed-agents/csv-analyst?source=agents_api`, or load it from code with `agents.Agent.load_from(agent_name="csv-analyst")`.

## Test

```bash
uv run python tests/test_tools.py
```
