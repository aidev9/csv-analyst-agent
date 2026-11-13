import asyncio
import base64
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


def test_tools_end_to_end() -> None:
    async def run() -> None:
        ctx = agents.ToolCallbackContext.local()
        b64 = base64.b64encode(CSV.encode()).decode()
        loaded = await main.load_tickets(ctx, file_id="file-1", csv_base64=b64)
        assert loaded["status"] == "loaded"
        assert loaded["row_count"] == 3
        assert "category" in loaded["columns"]

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
    test_tools_end_to_end()
