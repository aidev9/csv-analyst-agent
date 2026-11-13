import asyncio
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mistralai import agents

import main

CSV = (
    "ticket_id,created_date,resolved_date,category,priority,status,channel,subject,"
    "assigned_team,assigned_agent,resolution_hours,customer_id,satisfaction_score\n"
    "TKT-1,2024-01-01,2024-01-02,Order Issue,P3,Resolved,Live Chat,Order cancelled incorrectly,"
    "Order Management,Tom W.,30.3,CUST-1,4\n"
    "TKT-2,2024-01-02,2024-01-03,Payment & Billing,P4,Resolved,Self-Service Portal,Double charged,"
    "Finance Ops,Jane Z.,35.5,CUST-2,3\n"
    "TKT-3,2024-01-08,,Order Issue,P1,Open,Live Chat,Wrong item shipped,"
    "Order Management,James K.,,CUST-3,\n"
)


def _download_call(result: agents.ToolResult, tool_id: str = "connector_google_drive_mcp.download_file_content") -> agents.hook.PostToolCall:
    return agents.hook.PostToolCall(
        turn_id="local",
        tool_name=tool_id.rsplit(".", 1)[-1],
        tool_id=tool_id,
        tool_call_id="call-1",
        arguments={"file_id": "file-1"},
        tool_result=result,
    )


def _hook_ctx(session_id: str = "local") -> agents.HookCallbackContext:
    return agents.HookCallbackContext(agents.CallbackSession(id=session_id, turn_id="local"))


async def _load(result: agents.ToolResult, session_id: str = "local") -> dict:
    main._DATASETS.pop(session_id, None)
    out = await main.DriveTicketsLoader().post_tool_call(_hook_ctx(session_id), _download_call(result))
    assert out is not None
    return json.loads(out.tool_result.content[0].text)


def test_hook_loads_every_payload_shape() -> None:
    b64 = base64.b64encode(CSV.encode()).decode()
    shapes = {
        "embedded blob": agents.ToolResult(
            content=[agents.EmbeddedResourceContentBlock(uri="gdrive:///file-1", media_type="text/csv", blob=b64)]
        ),
        "embedded text": agents.ToolResult(
            content=[agents.EmbeddedResourceContentBlock(uri="gdrive:///file-1", media_type="text/csv", text=CSV)]
        ),
        "text base64": agents.ToolResult(content=[agents.TextContentBlock(text=b64)]),
        "text raw csv": agents.ToolResult(content=[agents.TextContentBlock(text=CSV)]),
        "text json": agents.ToolResult(
            content=[agents.TextContentBlock(text=json.dumps({"file_id": "file-1", "mimeType": "text/csv", "content": b64}))]
        ),
        "structured": agents.ToolResult(structured_content={"file": {"name": "ams-tickets.csv", "data": b64}}),
    }

    async def run() -> None:
        for label, result in shapes.items():
            summary = await _load(result)
            assert summary["status"] == "loaded", label
            assert summary["row_count"] == 3, label
            assert "category" in summary["columns"], label
            assert b64 not in json.dumps(summary) and "Wrong item shipped" not in json.dumps(summary), label

        bad = await main.DriveTicketsLoader().post_tool_call(
            _hook_ctx("bad"), _download_call(agents.ToolResult(content=[agents.TextContentBlock(text="not a csv")]))
        )
        assert bad is not None and bad.tool_result.is_error
        assert "bad" not in main._DATASETS

        other = await main.DriveTicketsLoader().post_tool_call(
            _hook_ctx("other"),
            _download_call(agents.ToolResult(content=[agents.TextContentBlock(text=CSV)]), tool_id="connector_google_drive_mcp.search_files"),
        )
        assert other is None
        assert "other" not in main._DATASETS

    asyncio.run(run())


def test_tools_end_to_end() -> None:
    async def run() -> None:
        ctx = agents.ToolCallbackContext.local()
        b64 = base64.b64encode(CSV.encode()).decode()
        loaded = await _load(agents.ToolResult(content=[agents.TextContentBlock(text=b64)]))
        assert loaded["status"] == "loaded"
        assert loaded["row_count"] == 3

        columns = await main.describe_columns(ctx)
        by_name = {c["column"]: c for c in columns}
        assert by_name["resolution_hours"]["type"] == "number"
        assert by_name["created_date"]["type"] == "date"
        assert by_name["status"]["type"] == "text"
        assert by_name["resolution_hours"]["null"] == 1

        resolved = await main.filter_rows(ctx, column="status", operator="eq", value="Resolved")
        assert resolved["matched"] == 2
        assert resolved["returned"] == 2

        open_p1 = await main.filter_rows(ctx, column="priority", operator="eq", value="P1")
        assert open_p1["matched"] == 1
        assert open_p1["rows"][0]["ticket_id"] == "TKT-3"

        slow = await main.filter_rows(ctx, column="resolution_hours", operator="gte", value="35")
        assert slow["matched"] == 1

        groups = await main.group_by_aggregate(
            ctx,
            group_by=["category"],
            aggregations=[
                main.Aggregation(column="ticket_id", func="count"),
                main.Aggregation(column="resolution_hours", func="avg"),
                main.Aggregation(column="satisfaction_score", func="min"),
            ],
        )
        assert groups["total_groups"] == 2
        order = groups["groups"][0]
        assert order["category"] == "Order Issue"
        assert order["count"] == 2
        assert order["ticket_id__count"] == 2
        assert order["resolution_hours__avg"] == 30.3

        try:
            await main.filter_rows(agents.ToolCallbackContext.local(), column="status", operator="eq", value="Open")
        except agents.ToolError:
            pass
        else:
            local_ctx = agents.ToolCallbackContext(agents.CallbackSession(id="other", turn_id="t"), tool_call_id="c")
            try:
                await main.describe_columns(local_ctx)
            except agents.ToolError:
                pass
            else:
                raise AssertionError("expected ToolError for a session with no dataset")

    asyncio.run(run())


if __name__ == "__main__":
    test_hook_loads_every_payload_shape()
    test_tools_end_to_end()
    print("ok")
