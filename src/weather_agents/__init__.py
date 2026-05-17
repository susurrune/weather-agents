"""Weather Agents — Multi-agent AI orchestration framework.

Five specialized agents (Fog, Rain, Frost, Snow, Dew) collaborate
through an event-driven message bus to accomplish complex tasks.
"""

# LiteLLM pre-loads Bedrock/SageMaker shapes and prints a WARNING to stderr
# when botocore is absent. We don't ship those integrations, so the lines are
# pure noise on every `wa` startup. Silence them as early as possible —
# before anything in this package imports litellm (e.g. core.llm).
import logging as _logging

_logging.getLogger("LiteLLM").setLevel(_logging.ERROR)

__version__ = "1.0.0"
