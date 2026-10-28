"""Tracks whether each component (STT, LLM, TTS) is loaded and usable."""

from app.schemas import ComponentHealth


class Readiness:
    """Component states written by the model loader."""

    def __init__(self) -> None:
        self._components: dict[str, ComponentHealth] = {}

    def set(self, name: str, state: str, detail: str = "") -> None:
        self._components[name] = ComponentHealth(state=state, detail=detail)

    def snapshot(self) -> dict[str, ComponentHealth]:
        return dict(self._components)


def overall_status(components: dict[str, ComponentHealth]) -> str:
    states = {c.state for c in components.values()}
    if "error" in states:
        return "degraded"
    if not states or "loading" in states:
        return "starting"
    return "ok"


def all_ready(components: dict[str, ComponentHealth]) -> bool:
    return bool(components) and all(c.state == "ready" for c in components.values())


def speech_models_ready(components: dict[str, ComponentHealth]) -> bool:
    """STT and TTS must be loaded before a turn can run. The LLM is not required here:
    if it is down the turn fails with a specific llm_error instead of "not ready"."""
    return all(name in components and components[name].state == "ready" for name in ("stt", "tts"))
