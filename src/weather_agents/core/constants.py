"""Shared constants across the Skyloom framework.

Central home for values that cross module boundaries — keeps
circular imports in check and avoids duplication.
"""

# Sentinel returned by the task_done tool. When the LLM calls
# task_done, the chat-stream loop detects this value in the tool
# result and terminates the turn cleanly (no truncation warning).
TASK_DONE_SENTINEL = "__TASK_DONE__"
