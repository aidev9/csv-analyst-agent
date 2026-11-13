import base64
import csv
import io
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from mistralai import agents

TICKETS_FILE_TITLE = "ams-tickets.csv"

_DATASETS: dict[str, dict[str, Any]] = {}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _get_dataset(ctx: agents.ToolCallbackContext) -> dict[str, Any]:
    dataset = _DATASETS.get(ctx.session.id)
    if dataset is None:
        raise agents.ToolError(
            "No CSV is loaded in this session. Reload data/ams-tickets.csv from Google Drive "
            "(search_files, then download_file_content), then call load_tickets with the file id "
            "and the returned base64 content."
        )
    return dataset


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _column_kind(values: list[str]) -> str:
    non_empty = [v for v in values if v not in (None, "")]
    if non_empty and all(_is_number(v) for v in non_empty):
        return "number"
    if non_empty and all(_DATE_RE.match(v) for v in non_empty):
        return "date"
    return "text"


def _coerce_filter_value(kind: str, value: str) -> Any:
    if kind == "number":
        return float(value)
    return value


@agents.tool(
    model_access="direct",
    annotations=agents.ToolAnnotations(read_only_hint=True),
)
async def load_tickets(
    ctx: agents.ToolCallbackContext,
    /,
    file_id: Annotated[str, Field(description="Google Drive file id of ams-tickets.csv, from search_files.")],
    csv_base64: Annotated[str, Field(description="Base64-encoded CSV content returned by download_file_content for that file id.")],
) -> dict[str, Any]:
    """Load the ams-tickets.csv downloaded from Google Drive into this session's dataset. Call this once per session before describe_columns, filter_rows or group_by_aggregate. Returns the row count and column names."""
    try:
        text = base64.b64decode(csv_base64).decode("utf-8-sig")
    except Exception:
        raise agents.ToolError("csv_base64 is not valid base64-encoded UTF-8 CSV text.")
    reader = csv.DictReader(io.StringIO(text))
    columns = list(reader.fieldnames or [])
    if not columns:
        raise agents.ToolError("The CSV has no header row.")
    rows = [dict(row) for row in reader]
    _DATASETS[ctx.session.id] = {
        "file_id": file_id,
        "file_title": TICKETS_FILE_TITLE,
        "columns": columns,
        "rows": rows,
    }
    return {
        "status": "loaded",
        "file_id": file_id,
        "file_title": TICKETS_FILE_TITLE,
        "row_count": len(rows),
        "columns": columns,
    }


@agents.tool(
    model_access="direct",
    annotations=agents.ToolAnnotations(read_only_hint=True),
)
async def describe_columns(
    ctx: agents.ToolCallbackContext,
    /,
) -> list[dict[str, Any]]:
    """Describe every column of the loaded dataset: inferred type (number, date or text), null count, distinct count, min/max/mean for numeric columns, and the most frequent values with counts for low-cardinality columns."""
    dataset = _get_dataset(ctx)
    rows, columns = dataset["rows"], dataset["columns"]
    result: list[dict[str, Any]] = []
    for column in columns:
        values = [row.get(column) for row in rows]
        non_empty = [v for v in values if v not in (None, "")]
        kind = _column_kind(values)
        entry: dict[str, Any] = {
            "column": column,
            "type": kind,
            "non_null": len(non_empty),
            "null": len(values) - len(non_empty),
            "distinct": len(set(non_empty)),
        }
        if kind == "number":
            nums = [float(v) for v in non_empty]
            if nums:
                entry["min"] = min(nums)
                entry["max"] = max(nums)
                entry["mean"] = round(sum(nums) / len(nums), 4)
        if entry["distinct"] <= 20 and non_empty:
            counts: dict[str, int] = {}
            for v in non_empty:
                counts[v] = counts.get(v, 0) + 1
            entry["top_values"] = [
                {"value": v, "count": c}
                for v, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ]
        entry["examples"] = non_empty[:5]
        result.append(entry)
    return result


@agents.tool(
    model_access="direct",
    annotations=agents.ToolAnnotations(read_only_hint=True),
)
async def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    column: Annotated[str, Field(description="Column name to filter on.")],
    operator: Annotated[
        Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains", "in", "is_null", "not_null"],
        Field(description="eq/ne equality, gt/gte/lt/lte comparisons (numeric or ISO date), contains substring (case-insensitive), in comma-separated list, is_null/not_null for empty values."),
    ],
    value: Annotated[str, Field(description="Value to compare against. Ignored for is_null/not_null. For 'in', a comma-separated list.")] = "",
    limit: Annotated[int, Field(description="Maximum number of matching rows to return.", ge=1, le=200)] = 25,
) -> dict[str, Any]:
    """Filter rows of the loaded dataset and return the matching rows plus the total match count."""
    dataset = _get_dataset(ctx)
    rows, columns = dataset["rows"], dataset["columns"]
    if column not in columns:
        raise agents.ToolError(f"Unknown column {column!r}. Available columns: {', '.join(columns)}.")
    kind = _column_kind([row.get(column) for row in rows])
    if operator in ("is_null", "not_null"):
        matched = [r for r in rows if (r.get(column) in (None, "")) == (operator == "is_null")]
    else:
        needle = _coerce_filter_value(kind, value)
        if operator == "in":
            options = {_coerce_filter_value(kind, part.strip()) for part in value.split(",")}
            matched = [r for r in rows if _coerce_filter_value(kind, r.get(column) or "") in options if r.get(column) not in (None, "")]
        else:
            matched = []
            for r in rows:
                cell = r.get(column)
                if cell in (None, ""):
                    continue
                cell_c = _coerce_filter_value(kind, cell)
                if operator == "eq" and cell_c == needle:
                    matched.append(r)
                elif operator == "ne" and cell_c != needle:
                    matched.append(r)
                elif operator == "gt" and cell_c > needle:
                    matched.append(r)
                elif operator == "gte" and cell_c >= needle:
                    matched.append(r)
                elif operator == "lt" and cell_c < needle:
                    matched.append(r)
                elif operator == "lte" and cell_c <= needle:
                    matched.append(r)
                elif operator == "contains" and str(needle).lower() in cell.lower():
                    matched.append(r)
    return {
        "column": column,
        "operator": operator,
        "value": value,
        "matched": len(matched),
        "returned": min(limit, len(matched)),
        "rows": matched[:limit],
    }


class Aggregation(BaseModel):
    column: Annotated[str, Field(description="Column to aggregate. For func='count', use any column; count ignores empty values.") ]
    func: Annotated[Literal["count", "count_distinct", "sum", "avg", "min", "max"], Field(description="Aggregate to compute per group.")]


@agents.tool(
    model_access="direct",
    annotations=agents.ToolAnnotations(read_only_hint=True),
)
async def group_by_aggregate(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: Annotated[list[str], Field(description="One or more column names to group rows by.")],
    aggregations: Annotated[list[Aggregation], Field(description="Aggregations to compute within each group.")],
    limit: Annotated[int, Field(description="Maximum number of groups to return, largest groups first.", ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Group the loaded dataset by one or more columns and compute aggregate values (count, count_distinct, sum, avg, min, max) per group. Groups are sorted by row count descending."""
    dataset = _get_dataset(ctx)
    rows, columns = dataset["rows"], dataset["columns"]
    for column in group_by:
        if column not in columns:
            raise agents.ToolError(f"Unknown group-by column {column!r}. Available columns: {', '.join(columns)}.")
    for agg in aggregations:
        if agg.column not in columns:
            raise agents.ToolError(f"Unknown aggregation column {agg.column!r}. Available columns: {', '.join(columns)}.")
    groups: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(column) for column in group_by)
        groups.setdefault(key, {"count": 0, "rows": []})
        groups[key]["count"] += 1
        groups[key]["rows"].append(row)
    output_groups: list[dict[str, Any]] = []
    for key, bucket in sorted(groups.items(), key=lambda kv: -kv[1]["count"]):
        entry: dict[str, Any] = dict(zip(group_by, key))
        entry["count"] = bucket["count"]
        for agg in aggregations:
            values = [r.get(agg.column) for r in bucket["rows"]]
            non_empty = [v for v in values if v not in (None, "")]
            name = f"{agg.column}__{agg.func}"
            if agg.func == "count":
                entry[name] = len(non_empty)
            elif agg.func == "count_distinct":
                entry[name] = len(set(non_empty))
            else:
                nums = [float(v) for v in non_empty if _is_number(v)]
                if not nums:
                    entry[name] = None
                elif agg.func == "sum":
                    entry[name] = round(sum(nums), 4)
                elif agg.func == "avg":
                    entry[name] = round(sum(nums) / len(nums), 4)
                elif agg.func == "min":
                    entry[name] = min(nums)
                elif agg.func == "max":
                    entry[name] = max(nums)
        output_groups.append(entry)
    return {
        "group_by": group_by,
        "total_groups": len(output_groups),
        "returned_groups": min(limit, len(output_groups)),
        "groups": output_groups[:limit],
    }


INSTRUCTIONS = """You are CSV Analyst, a conversational data analyst specialized in support-ticket analysis.

DATA ACCESS (strict):
- Your only data source is the file data/ams-tickets.csv in the user's Google Drive, reached through the google_drive_mcp connector, which acts as the user.
- Locate it with search_files using the query: title = 'ams-tickets.csv' and mimeType = 'text/csv'. Use only that file's id.
- The connector is read-only for you: only ever call its search_files, get_file_metadata and download_file_content tools. Never create, copy, update or delete any Drive file, and never open any other file.
- To load the dataset into the session: call download_file_content with the file id, then call load_tickets with that file id and the returned base64 content.

WORKFLOW:
1. On the first question about the data (or whenever a data tool reports that no CSV is loaded in this session), locate and load data/ams-tickets.csv as described above.
2. Call describe_columns to learn the schema, then answer with filter_rows and group_by_aggregate so every figure comes from the real data.
3. This session keeps its full history: remember the file and everything discussed, and answer follow-ups directly without asking the user to re-explain. If a data tool says nothing is loaded (for example after the session resumed on a new worker), reload the same file from Drive with the remembered file id, then continue.

TOOLS (fixed set, no free-form data processing):
- Compute over the CSV only through the four fixed tools: load_tickets, describe_columns, filter_rows, group_by_aggregate. Do not attempt to transform data any other way; compose these tools instead.

ANALYSIS STYLE (ticket-analysis specialty):
- Ground every number in tool results; never invent figures. If a question needs a cut the tools cannot compute exactly, say so and give the closest real numbers you computed.
- For analytical requests, structure the answer with: key findings with real numbers; root-cause analysis of the dominant issue patterns (categories, channels, teams, agents, resolution times, satisfaction); top priorities for fixing issues, ranked with counts and concrete ticket examples; automation opportunities (repeatable, rule-based ticket types that a bot or macro could handle); and short-term, medium-term and long-term plans, each with concrete steps.
- Use Markdown headings, tables and bullet lists. Be concise but analytically detailed, and cite the tool-derived numbers behind each claim.
"""

agent = agents.Agent(
    model="mistral-large-4",
    name="csv-analyst",
    description="Conversational analyst for the ams-tickets.csv support-ticket dataset in Google Drive: root causes, priorities, automation opportunities and plans, with real numbers.",
    instructions=INSTRUCTIONS,
    tools=[load_tickets, describe_columns, filter_rows, group_by_aggregate],
    connectors=[agents.connector("google_drive_mcp")],
    builtins=agents.Builtins(skills=False, subagents=False),
    permission_policy=agents.PermissionPolicy(
        default="allow",
        rules=(
            agents.PermissionRule(
                tools=(
                    "connector_google_drive_mcp.create_file",
                    "connector_google_drive_mcp.copy_file",
                    "connector_google_drive_mcp.update_file",
                ),
                decision="deny",
            ),
        ),
    ),
    max_iterations=30,
)
