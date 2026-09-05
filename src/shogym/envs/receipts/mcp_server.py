"""In-process MCP server for the receipts env: the single ``submit_filing`` tool.

A receipts episode is one filing. ``submit_filing`` is the env's score terminal:
the serve layer validates its args against the schema this tool advertises,
atomically seals the episode, then runs the env's ``finalize``, so this handler
body is never dispatched for a sealed episode. It exists to publish the tool's
name and argument schema in the manifest ``describe()`` reads.

The tool returns nothing evaluative on either channel. Whether the filing matched is
what an experiment delivers afterwards, in a controlled arm, as one cell of a fork;
an env whose terminal carried the verdict would be putting a graded receipt in every
arm including the one that is meant to be empty. The result's content says the filing
landed and how many rows it named, and its ``_meta`` sidecar carries the terminate
flag alone, because env_v1 declares ``inband_terminal_feedback = False``. That is the
shape of a successful filing; an abort, a failed terminal transaction, a schema
refusal and a post-seal call each have their own, and env_v1's README lists them.
"""

from __future__ import annotations

from typing import Any, Dict

from fastmcp import FastMCP

SUBMIT_TOOL_NAME = "submit_filing"

server: FastMCP = FastMCP(name="receipts")


@server.tool
def submit_filing(filing: str, _session_id: str) -> Dict[str, Any]:
    """File your answer. **This ends the episode.**

    ``filing`` is one line per record: the record id, a comma, and the value, with
    no header and no other text. File every record, in the order they appear in the
    schedule. A record you are leaving empty still gets a line, with nothing after
    the comma.

    Filing seals the episode. There is no second filing, no verdict in what this
    returns and no further step to take, so file your best answer.
    """
    # Never reached for a sealed episode: the serve layer intercepts this score
    # terminal, validates and seals and runs the env's `finalize` instead of
    # dispatching here. Kept as a defensive, non-grading stub.
    return {"filed": True, "bytes": len(filing)}
