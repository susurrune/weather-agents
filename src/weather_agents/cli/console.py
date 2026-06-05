"""Shared Rich console for the CLI.

A single ``Console`` instance is shared across the CLI package so that Live
displays, prompts, and printed output all coordinate on one terminal handle
(Rich allows only one active Live per console). Importing ``console`` from here
— rather than each module constructing its own ``Console()`` — keeps that
invariant and lets the CLI be split into focused modules without fragmenting
terminal state.
"""

from __future__ import annotations

from rich.console import Console

console = Console()
