"""Configuration management for Weather Agents."""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

USER_CONFIG_DIR = Path.home() / ".weather-agents"

# All agent names — single source of truth for config iteration and validation.
AGENT_NAMES = ("fog", "rain", "frost", "snow", "dew", "sunshine")


def _find_config_dir() -> Path:
    """Locate the config/ directory reliably in dev and installed modes."""
    from importlib.resources import files

    try:
        ref = files("weather_agents") / "config"
        path = Path(str(ref))
        if (path / "default.yaml").exists():
            return path
    except Exception:
        pass

    dev_path = Path(__file__).parent.parent.parent.parent / "config"
    if (dev_path / "default.yaml").exists():
        return dev_path

    return USER_CONFIG_DIR / "config"


CONFIG_DIR = _find_config_dir()


# ── Model Catalog ──────────────────────────────────────────────────────────


def load_model_catalog() -> dict[str, list[dict]]:
    """Load available models from models.yaml, grouped by provider.

    Each entry includes: name, provider, context_window, max_output,
    and optionally: input_cost_per_1k, output_cost_per_1k, fallback (list of model names).
    """
    path = CONFIG_DIR / "models.yaml"
    if not path.exists():
        return {}
    data = _load_yaml(path)
    catalog: dict[str, list[dict]] = {}
    for provider, models in data.items():
        if isinstance(models, dict):
            catalog[provider] = []
            for name, info in models.items():
                entry: dict = {"name": name}
                if isinstance(info, dict):
                    entry.update(info)
                else:
                    entry["provider"] = info
                catalog[provider].append(entry)
    return catalog


def format_models_for_display(catalog: dict[str, list[dict]]) -> str:
    """Pretty-print the model catalog for CLI display."""
    lines = []
    for provider, models in catalog.items():
        lines.append(f"  [{provider.upper()}]")
        for m in models:
            cost_parts = []
            if m.get("input_cost_per_1k"):
                cost_parts.append(f"${m['input_cost_per_1k']:.4f}/1k in")
            if m.get("output_cost_per_1k"):
                cost_parts.append(f"${m['output_cost_per_1k']:.4f}/1k out")
            cost_str = f"  cost=({', '.join(cost_parts)})" if cost_parts else ""
            fallback_str = ""
            if m.get("fallback"):
                fallback_str = f"  fallback->{' > '.join(m['fallback'])}"
            lines.append(
                f"    {m['name']}  (ctx={m.get('context_window', '?')}, max={m.get('max_output', '?')}){cost_str}{fallback_str}"
            )
    return "\n".join(lines)


_CTX_CACHE: dict[str, int] = {}


def get_model_context_window(model_name: str) -> int:
    """Look up context window for a model. Returns 128K if unknown."""
    if not isinstance(model_name, str):
        return 128000
    if model_name in _CTX_CACHE:
        return _CTX_CACHE[model_name]
    catalog = load_model_catalog()
    for models in catalog.values():
        for m in models:
            if m["name"] == model_name:
                val = int(m.get("context_window", 128000))
                _CTX_CACHE[model_name] = val
                return val
    # Strip provider prefix and retry
    if "/" in model_name:
        return get_model_context_window(model_name.split("/", 1)[1])
    _CTX_CACHE[model_name] = 128000
    return 128000


# ── Config dataclasses ─────────────────────────────────────────────────────


@dataclass
class LLMConfig:
    default_model: str = "deepseek/deepseek-v4-flash"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 2
    api_keys: dict[str, str] = field(default_factory=dict)
    language: str = "zh"


@dataclass
class AgentModelConfig:
    model: str | None = None
    specialty: str = ""
    max_tool_rounds: int = 10


@dataclass
class AgentConfigs:
    fog: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(specialty="探索研究"))
    rain: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(specialty="生成创造"))
    frost: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(specialty="审查优化"))
    snow: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(specialty="规划编排"))
    dew: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(specialty="运维集成"))
    sunshine: AgentModelConfig = field(
        default_factory=lambda: AgentModelConfig(specialty="情感陪伴")
    )


@dataclass
class BusConfig:
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class MemoryConfig:
    db_path: str = "~/.weather-agents/memory.db"
    short_term_limit: int = 50
    max_persisted_messages: int = 1000


@dataclass
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class WorkspaceConfig:
    path: str = "auto"  # "auto" = detect best drive; or absolute path


@dataclass
class TTSConfig:
    enabled: bool = False
    provider: str = "doubao"
    access_token: str = ""
    api_key: str = ""
    app_id: str = ""
    resource_id: str = "seed-tts-2.0"
    voice_type: str = "zh_female_sajiaoxuemei_uranus_bigtts"
    encoding: str = "mp3"
    sample_rate: int = 24000
    speed_ratio: float = 1.0
    volume_ratio: float = 1.0
    pitch_ratio: float = 1.0
    emotion: str = "happy"


@dataclass
class PluginConfig:
    enabled: bool = True
    directories: list[str] = field(default_factory=lambda: ["~/.weather-agents/plugins"])


@dataclass
class MCPServerItem:
    name: str = ""
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class MCPConfig:
    servers: list[dict] = field(default_factory=list)


@dataclass
class CLIConfig:
    # Persisted interactive-mode pick (default | plan | auto). Read by
    # cli.mode.ModeController on startup; written when the user toggles.
    interactive_mode: str = "default"


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agents: AgentConfigs = field(default_factory=AgentConfigs)
    bus: BusConfig = field(default_factory=BusConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    web: WebConfig = field(default_factory=WebConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)


# ── Load / Save helpers ────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_user_cfg(data: dict) -> None:
    """Write data to the user config file, merging with existing."""
    path = USER_CONFIG_DIR / "config.yaml"
    existing = _load_yaml(path)
    _deep_merge(existing, data)
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(existing, f, allow_unicode=True, default_flow_style=False)
    invalidate_cache()


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _write_yaml(path: Path, data: dict) -> None:
    """Write dict to YAML file directly (not merge). Set restrictive permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    # Restrict permissions to owner-only for sensitive data (API keys)
    with contextlib.suppress(PermissionError):
        os.chmod(path, 0o600)
    invalidate_cache()


def _resolve_env(value: str) -> str:
    """Resolve ${VAR} placeholders to environment variables.

    Logs a warning when the variable is missing so misconfiguration is visible
    instead of silently substituting an empty string.
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var = value[2:-1]
        resolved = os.getenv(var)
        if resolved is None:
            import logging

            logging.getLogger("weather_agents.config").warning(
                "env_var_missing: %s referenced but not set; using empty string", var
            )
            return ""
        return resolved
    return value


# ── Config cache ────────────────────────────────────────────────────────

_config_cache: AppConfig | None = None
_config_cache_time: float = 0
_CONFIG_CACHE_TTL: float = 2.0  # seconds


def invalidate_cache() -> None:
    """Force next load_config() to re-read from disk."""
    global _config_cache, _config_cache_time
    _config_cache = None
    _config_cache_time = 0.0


# ── Public API ─────────────────────────────────────────────────────────────


def load_config() -> AppConfig:
    """Load config from default + user overrides + env vars, with TTL cache."""
    global _config_cache, _config_cache_time

    now = time.monotonic()
    if _config_cache is not None and (now - _config_cache_time) < _CONFIG_CACHE_TTL:
        return _config_cache

    cfg = _load_config_uncached()
    _config_cache = cfg
    _config_cache_time = now
    return cfg


def _load_dotenv() -> None:
    """Load .env file into os.environ, respecting existing env vars.

    Priority: existing env var > .env file.
    Looks for .env in current working directory, then user config directory.
    """
    for base in (Path.cwd(), USER_CONFIG_DIR):
        dotenv_path = base / ".env"
        if not dotenv_path.exists():
            continue
        try:
            with open(dotenv_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except OSError:
            pass


def _load_config_uncached() -> AppConfig:
    cfg = AppConfig()

    _load_dotenv()

    default_data = _load_yaml(CONFIG_DIR / "default.yaml")
    user_data = _load_yaml(USER_CONFIG_DIR / "config.yaml")

    # Merge: user overrides defaults
    merged = {**default_data}
    _deep_merge(merged, user_data)

    # LLM settings
    if llm := merged.get("llm"):
        cfg.llm.default_model = llm.get("default_model", cfg.llm.default_model)
        cfg.llm.temperature = llm.get("temperature", cfg.llm.temperature)
        cfg.llm.max_tokens = llm.get("max_tokens", cfg.llm.max_tokens)
        cfg.llm.timeout = llm.get("timeout", cfg.llm.timeout)
        cfg.llm.language = llm.get("language", cfg.llm.language)
        if keys := llm.get("api_keys"):
            cfg.llm.api_keys = {k: _resolve_env(v) for k, v in keys.items()}

    # Per-agent overrides
    if agents := merged.get("agents"):
        for name in AGENT_NAMES:
            if agent_cfg := agents.get(name):
                attr: AgentModelConfig = getattr(cfg.agents, name)
                if m := agent_cfg.get("model"):
                    attr.model = m
                if s := agent_cfg.get("specialty"):
                    attr.specialty = s
                if mr := agent_cfg.get("max_tool_rounds"):
                    attr.max_tool_rounds = int(mr)

    # Web
    if web := merged.get("web"):
        cfg.web.host = web.get("host", cfg.web.host)
        cfg.web.port = web.get("port", cfg.web.port)

    # TTS (Doubao / Volcano Engine)
    if tts_cfg := merged.get("tts"):
        cfg.tts.enabled = tts_cfg.get("enabled", False)
        cfg.tts.provider = tts_cfg.get("provider", cfg.tts.provider)
        cfg.tts.access_token = tts_cfg.get("access_token", "")
        cfg.tts.api_key = tts_cfg.get("api_key", "")
        cfg.tts.app_id = tts_cfg.get("app_id", "")
        cfg.tts.resource_id = tts_cfg.get("resource_id", cfg.tts.resource_id)
        cfg.tts.voice_type = tts_cfg.get("voice_type", cfg.tts.voice_type)
        cfg.tts.encoding = tts_cfg.get("encoding", cfg.tts.encoding)
        cfg.tts.sample_rate = int(tts_cfg.get("sample_rate", 24000))
        cfg.tts.speed_ratio = float(tts_cfg.get("speed_ratio", 1.0))
        cfg.tts.volume_ratio = float(tts_cfg.get("volume_ratio", 1.0))
        cfg.tts.pitch_ratio = float(tts_cfg.get("pitch_ratio", 1.0))
        cfg.tts.emotion = tts_cfg.get("emotion", cfg.tts.emotion)

    # Workspace
    if ws := merged.get("workspace"):
        cfg.workspace.path = ws.get("path", cfg.workspace.path)

    # Memory
    if mem := merged.get("memory"):
        cfg.memory.db_path = mem.get("db_path", cfg.memory.db_path)
        cfg.memory.short_term_limit = mem.get("short_term_limit", cfg.memory.short_term_limit)
        cfg.memory.max_persisted_messages = mem.get(
            "max_persisted_messages", cfg.memory.max_persisted_messages
        )

    # CLI (interactive mode persisted across sessions)
    if cli_cfg := merged.get("cli"):
        cfg.cli.interactive_mode = cli_cfg.get("interactive_mode", cfg.cli.interactive_mode)

    # MCP (with env var resolution — only for enabled servers)
    if (mcp := merged.get("mcp")) and (servers := mcp.get("servers")):
        resolved = []
        for s in servers:
            if s.get("enabled", True):
                env = {k: _resolve_env(v) for k, v in s.get("env", {}).items()}
                s["env"] = env
            else:
                s["env"] = s.get("env", {})
            resolved.append(s)
        cfg.mcp.servers = resolved

    # API keys from env vars (lowest priority)
    if not cfg.llm.api_keys.get("openai") and os.getenv("OPENAI_API_KEY"):
        cfg.llm.api_keys["openai"] = os.getenv("OPENAI_API_KEY", "")
    if not cfg.llm.api_keys.get("anthropic") and os.getenv("ANTHROPIC_API_KEY"):
        cfg.llm.api_keys["anthropic"] = os.getenv("ANTHROPIC_API_KEY", "")
    if not cfg.llm.api_keys.get("deepseek") and os.getenv("DEEPSEEK_API_KEY"):
        cfg.llm.api_keys["deepseek"] = os.getenv("DEEPSEEK_API_KEY", "")
    if not cfg.llm.api_keys.get("google") and os.getenv("GOOGLE_API_KEY"):
        cfg.llm.api_keys["google"] = os.getenv("GOOGLE_API_KEY", "")

    _sync_api_keys_to_env(cfg.llm.api_keys)

    return cfg


_ENV_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def _sync_api_keys_to_env(api_keys: dict[str, str]) -> None:
    """Push config API keys into environment so LiteLLM can find them."""
    for provider, key in api_keys.items():
        if not key:
            continue
        env_var = _ENV_KEY_MAP.get(provider)
        if env_var:
            os.environ[env_var] = key
        else:
            os.environ[f"{provider.upper()}_API_KEY"] = key


def set_config(key: str, value: str) -> tuple[bool, str]:
    """Set a config key and persist to user config.

    Supported keys:
      default_model, temperature, max_tokens, timeout
      model.<agent>      (fog/rain/frost/snow/dew/sunshine)
      api_key.<provider> (openai/anthropic/deepseek/google)
    """
    parts = key.split(".")

    # api_key.<provider>
    if len(parts) == 2 and parts[0] == "api_key":
        provider = parts[1]
        _save_user_cfg({"llm": {"api_keys": {provider: value}}})
        return True, f"api_key.{provider} saved"

    # workspace.path
    if len(parts) == 2 and parts[0] == "workspace":
        if parts[1] == "path":
            # Validate: must be "auto" or an absolute path
            if value.lower() != "auto":
                expanded = os.path.expanduser(value)
                # Check the raw value is absolute so relative paths like
                # "foo/bar" are caught (Path.resolve() would make them
                # absolute by prepending cwd on all platforms).
                if not Path(expanded).is_absolute():
                    return False, f"workspace path must be absolute or 'auto', got: {value}"
            _save_user_cfg({"workspace": {"path": value}})
            return True, f"workspace.path → {value}"
        return False, f"unknown workspace key: {parts[1]}"

    # model.<agent>
    if len(parts) == 2 and parts[0] == "model":
        agent_name = parts[1]
        VALID_AGENTS = AGENT_NAMES
        if agent_name not in VALID_AGENTS:
            return False, f"unknown agent '{agent_name}', use: model.{', model.'.join(AGENT_NAMES)}"
        _save_user_cfg({"agents": {agent_name: {"model": value}}})
        return True, f"{agent_name} model → {value}"

    # Simple keys under llm
    SIMPLE_LLM_KEYS = ("default_model", "temperature", "max_tokens", "timeout")
    if key in SIMPLE_LLM_KEYS:
        typed_val: str | float | int = value
        try:
            if key == "temperature":
                typed_val = float(value)
                if not 0.0 <= typed_val <= 2.0:
                    return False, "temperature must be in [0.0, 2.0]"
            elif key == "max_tokens":
                typed_val = int(value)
                if not 1 <= typed_val <= 200_000:
                    return False, "max_tokens must be in [1, 200000]"
            elif key == "timeout":
                typed_val = int(value)
                if not 1 <= typed_val <= 600:
                    return False, "timeout must be in [1, 600] seconds"
        except ValueError:
            return False, f"invalid value for {key}: {value!r}"
        _save_user_cfg({"llm": {key: typed_val}})
        return True, f"{key} → {value}"

    return False, f"unknown config key: {key}"


def delete_config(key: str) -> tuple[bool, str]:
    """Delete a config key from user config.

    Supported keys: same as set_config(), plus:
      api_key.<provider>  (removes the key)
    """
    path = USER_CONFIG_DIR / "config.yaml"
    data = _load_yaml(path)
    if not data:
        return True, "nothing to delete (config is empty)"

    parts = key.split(".")

    if len(parts) == 2 and parts[0] == "api_key":
        provider = parts[1]
        removed = data.get("llm", {}).get("api_keys", {}).pop(provider, None)
        if removed:
            _write_yaml(path, data)
        env_var = _ENV_KEY_MAP.get(provider, f"{provider.upper()}_API_KEY")
        os.environ.pop(env_var, None)
        if removed:
            return True, f"api_key.{provider} deleted"
        return True, f"api_key.{provider} not set"

    if len(parts) == 2 and parts[0] == "workspace" and parts[1] == "path":
        removed = data.get("workspace", {}).pop("path", None)
        if removed:
            _write_yaml(path, data)
            return True, "workspace.path reset to auto"
        return True, "workspace.path already at auto"

    if len(parts) == 2 and parts[0] == "model":
        agent_name = parts[1]
        VALID_AGENTS = AGENT_NAMES
        if agent_name not in VALID_AGENTS:
            return False, "unknown agent"
        removed = data.get("agents", {}).get(agent_name, {}).pop("model", None)
        if removed:
            _write_yaml(path, data)
            return True, f"{agent_name} model reset to default"
        return True, f"{agent_name} already using default"

    SIMPLE_LLM_KEYS = ("default_model", "temperature", "max_tokens", "timeout")
    if key in SIMPLE_LLM_KEYS:
        removed = data.get("llm", {}).pop(key, None)
        if removed:
            _write_yaml(path, data)
            return True, f"{key} reset to default"
        return True, f"{key} already at default"

    return False, f"unknown config key: {key}"
