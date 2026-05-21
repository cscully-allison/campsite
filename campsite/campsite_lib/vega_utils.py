"""Vega-Lite specification traversal and inspection utilities.

Pure, stateless functions for walking composed vega-lite specs and
extracting structural information used by the artifact extractors
in si_checkers.py.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Core traversal
# ---------------------------------------------------------------------------


def walk_units(spec: dict) -> list[dict]:
    """Flatten a composed vega-lite spec into terminal unit specs.

    A unit spec is one that has a ``mark`` key. Composition operators
    (layer, hconcat, vconcat, concat) are recursed.  Faceted specs
    (``facet`` key with a ``spec`` sub-key) are also walked.
    """
    units: list[dict] = []

    if "mark" in spec:
        units.append(spec)

    for key in ("layer", "hconcat", "vconcat", "concat"):
        for child in spec.get(key, []):
            if isinstance(child, dict):
                units.extend(walk_units(child))

    facet_spec = spec.get("spec")
    if isinstance(facet_spec, dict) and "facet" in spec:
        units.extend(walk_units(facet_spec))

    return units


def get_mark_type(unit: dict) -> Optional[str]:
    """Extract the mark type string from a unit spec.

    Handles shorthand (``"mark": "point"``) and object form
    (``"mark": {"type": "point", ...}``).
    """
    mark = unit.get("mark")
    if isinstance(mark, str):
        return mark
    if isinstance(mark, dict):
        return mark.get("type")
    return None


def extract_all_marks(spec: dict) -> list[str]:
    """Extract all mark type strings from a possibly-composed spec."""
    marks: list[str] = []
    for unit in walk_units(spec):
        mt = get_mark_type(unit)
        if mt is not None:
            marks.append(mt)
    return marks


# ---------------------------------------------------------------------------
# Encoding inspection
# ---------------------------------------------------------------------------


def get_encodings(unit: dict) -> dict:
    """Get the encoding dict from a unit spec."""
    return unit.get("encoding", {}) or {}


def get_encoding_type(encoding_def: dict) -> Optional[str]:
    """Get the type (quantitative/nominal/ordinal/temporal) from an encoding definition."""
    if isinstance(encoding_def, dict):
        return encoding_def.get("type")
    return None


def get_encoding_field(encoding_def: dict) -> Optional[str]:
    """Get the field name from an encoding channel definition."""
    if isinstance(encoding_def, dict):
        return encoding_def.get("field")
    return None


def get_encoding_aggregate(encoding_def: dict) -> Optional[str]:
    """Get the aggregate function from an encoding channel definition."""
    if isinstance(encoding_def, dict):
        return encoding_def.get("aggregate")
    return None


# ---------------------------------------------------------------------------
# Transform collection
# ---------------------------------------------------------------------------


def get_transforms(spec: dict) -> list[dict]:
    """Collect all transforms from the entire composition tree."""
    transforms: list[dict] = []

    for t in spec.get("transform", []):
        if isinstance(t, dict):
            transforms.append(t)

    for key in ("layer", "hconcat", "vconcat", "concat"):
        for child in spec.get(key, []):
            if isinstance(child, dict):
                transforms.extend(get_transforms(child))

    facet_spec = spec.get("spec")
    if isinstance(facet_spec, dict) and "facet" in spec:
        transforms.extend(get_transforms(facet_spec))

    return transforms


def get_filter_transforms(spec: dict) -> list[dict]:
    """Get only filter transforms from the composition tree."""
    return [t for t in get_transforms(spec) if "filter" in t]


def get_calculate_transforms(spec: dict) -> list[dict]:
    """Get only calculate transforms from the composition tree."""
    return [t for t in get_transforms(spec) if "calculate" in t]


def get_aggregate_groupby_fields(spec: dict) -> list[str]:
    """Extract groupby field names from aggregate transforms."""
    fields: list[str] = []
    for t in get_transforms(spec):
        if "aggregate" in t:
            for field_name in t.get("groupby", []):
                if isinstance(field_name, str):
                    fields.append(field_name)
    return fields


# ---------------------------------------------------------------------------
# Grouping and faceting
# ---------------------------------------------------------------------------


def get_facet_fields(spec: dict) -> list[str]:
    """Extract row/column/facet field names from the spec."""
    fields: list[str] = []

    for unit in walk_units(spec):
        enc = get_encodings(unit)
        for channel in ("row", "column"):
            ch_def = enc.get(channel)
            if isinstance(ch_def, dict):
                field_name = ch_def.get("field")
                if field_name:
                    fields.append(field_name)

    # Explicit facet at top level
    facet = spec.get("facet")
    if isinstance(facet, dict):
        field_name = facet.get("field")
        if field_name:
            fields.append(field_name)
        for sub in ("row", "column"):
            sub_def = facet.get(sub)
            if isinstance(sub_def, dict) and sub_def.get("field"):
                fields.append(sub_def["field"])

    return fields


def has_grouping(spec: dict) -> bool:
    """Check if any unit spec has a nominal/ordinal grouping encoding.

    Grouping channels: color, row, column mapped to nominal or ordinal data.
    """
    for unit in walk_units(spec):
        enc = get_encodings(unit)
        for channel in ("color", "row", "column"):
            ch_def = enc.get(channel)
            if isinstance(ch_def, dict):
                if ch_def.get("type") in ("nominal", "ordinal"):
                    return True
    return False


def get_color_grouping_fields(spec: dict) -> list[str]:
    """Get field names from color encodings with nominal/ordinal type."""
    fields: list[str] = []
    for unit in walk_units(spec):
        enc = get_encodings(unit)
        color_def = enc.get("color")
        if isinstance(color_def, dict):
            if color_def.get("type") in ("nominal", "ordinal"):
                field_name = color_def.get("field")
                if field_name:
                    fields.append(field_name)
    return fields


# ---------------------------------------------------------------------------
# Semantic detection (higher-level, reused across checkers)
# ---------------------------------------------------------------------------


def _is_uncertainty_band(unit: dict) -> bool:
    """Check if a unit is an uncertainty band (area/rule with y+y2)."""
    mark = get_mark_type(unit)
    if mark not in ("area", "rule"):
        return False
    enc = get_encodings(unit)
    return "y" in enc and "y2" in enc


def detect_referent_pattern(spec: dict) -> str:
    """Determine referent type from structural patterns in a vega-lite spec.

    Returns one of: ``"threshold"``, ``"quantity"``, ``"missing"``.

    Detection priority:
    1. rule mark present → ``"threshold"`` (reference line)
    2. text mark present → ``"threshold"`` (annotation)
    3. hconcat/vconcat with >1 child → ``"quantity"``
    4. Multiple primary layers (excl. rule/text/uncertainty bands) → ``"quantity"``
    5. Otherwise → ``"missing"``
    """
    marks = extract_all_marks(spec)

    if "rule" in marks:
        return "threshold"

    if "text" in marks:
        return "threshold"

    for key in ("hconcat", "vconcat"):
        children = spec.get(key)
        if isinstance(children, list) and len(children) > 1:
            return "quantity"

    units = walk_units(spec)
    if len(units) > 1:
        primary_units = [
            u for u in units
            if get_mark_type(u) not in ("rule", "text", None)
            and not _is_uncertainty_band(u)
        ]
        if len(primary_units) >= 2:
            return "quantity"

    return "missing"


def has_uncertainty_encoding(spec: dict) -> bool:
    """Check for y2/band-style uncertainty encodings.

    Detects area marks with y+y2 (confidence bands) and
    rule marks with y+y2 (error bars).
    """
    for unit in walk_units(spec):
        if _is_uncertainty_band(unit):
            return True
    return False


def has_division_calculate(spec: dict) -> bool:
    """Check for calculate transforms with division operations."""
    for t in get_calculate_transforms(spec):
        expr = t.get("calculate", "")
        if isinstance(expr, str) and "/" in expr:
            return True
    return False


def detect_partition_ordering(spec: dict) -> bool:
    """Check if the primary grouping dimension uses temporal or ordinal type.

    Returns True if the grouping axis is temporal or ordinal, False otherwise.
    """
    for unit in walk_units(spec):
        enc = get_encodings(unit)
        for channel in ("x", "color", "row", "column"):
            ch_def = enc.get(channel, {})
            if isinstance(ch_def, dict):
                ch_type = ch_def.get("type")
                if ch_type in ("temporal", "ordinal"):
                    return True
    return False
