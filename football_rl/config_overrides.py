"""`--config-override section.field=value` parsing and value coercion against the config dataclasses."""

from __future__ import annotations

from dataclasses import fields, replace
import math
from typing import Any, Dict, List, Sequence, Tuple

from football_rl.config import DEFAULT_CONFIG, EnvConfig, PPOHyperparams, RewardConfig


CONFIG_OVERRIDE_SECTIONS = {"ppo": PPOHyperparams, "rewards": RewardConfig}


def coerce_override_value(raw: str, annotation: Any) -> Any:
    """Parse the text ``raw`` as the dataclass field type ``annotation`` (``float``, ``int``, ``str`` or ``bool``).

    Booleans are spelled ``true``/``false`` (case-insensitive); floats must be finite. Anything else raises ValueError.
    """
    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    if name == "bool":
        lowered = raw.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        raise ValueError(f"expected 'true' or 'false', got {raw!r}")
    if name == "int":
        return int(raw)
    if name == "float":
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"expected a finite number, got {raw!r}")
        return value
    if name == "str":
        return raw
    raise ValueError(f"field type {name} cannot be set from the command line")


def parse_config_overrides(
    specs: Sequence[str], base: EnvConfig = DEFAULT_CONFIG
) -> Tuple[EnvConfig, Dict[str, Any]]:
    """Apply ``SECTION.KEY=VALUE`` overrides (sections ``rewards`` and ``ppo``) to ``base``.

    Returns the new config and the applied overrides as a dict sorted by ``SECTION.KEY`` with values coerced
    to the field types. Any malformed spec, unknown section or key, duplicate, unparseable value or
    failed dataclass validation raises ValueError naming the offending item.
    """
    values: Dict[str, Dict[str, Any]] = {section: {} for section in CONFIG_OVERRIDE_SECTIONS}
    sources: Dict[str, List[str]] = {section: [] for section in CONFIG_OVERRIDE_SECTIONS}
    applied: Dict[str, Any] = {}
    for spec in specs:
        target, eq, raw = spec.partition("=")
        section, dot, key = target.partition(".")
        if not eq or not dot or not section or not key:
            raise ValueError(f"--config-override {spec!r}: expected SECTION.KEY=VALUE")
        cls = CONFIG_OVERRIDE_SECTIONS.get(section)
        if cls is None:
            raise ValueError(
                f"--config-override {spec!r}: unknown section {section!r} "
                f"(allowed: {', '.join(sorted(CONFIG_OVERRIDE_SECTIONS))})"
            )
        field_types = {f.name: f.type for f in fields(cls)}
        if key not in field_types:
            raise ValueError(f"--config-override {spec!r}: {section!r} has no key {key!r}")
        if target in applied:
            raise ValueError(f"--config-override {spec!r}: {target} is set more than once")
        try:
            value = coerce_override_value(raw, field_types[key])
        except ValueError as exc:
            raise ValueError(f"--config-override {spec!r}: cannot read {raw!r} as {field_types[key]}: {exc}") from exc
        applied[target] = value
        values[section][key] = value
        sources[section].append(spec)

    cfg = base
    for section, changes in values.items():
        if not changes:
            continue
        try:
            cfg = replace(cfg, **{section: replace(getattr(cfg, section), **changes)})
        except ValueError as exc:
            raise ValueError(f"--config-override {', '.join(sources[section])}: {exc}") from exc
    return cfg, dict(sorted(applied.items()))
