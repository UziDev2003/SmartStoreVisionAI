"""
Activity Rule Engine
====================
Composable, YAML-driven rule engine. Each rule evaluates a context
dictionary (track features, zone, history snapshot) and contributes
a confidence score if the condition is satisfied.

Example rule:
    - name: "high_value_zone_loiter"
      when:
        zone_type: "general"
        zone_name_contains: "high_value"
        dwell_seconds: ">120"
        motion_var: "<0.05"
      then:
        activity: "loitering"
        severity: "MEDIUM"
        confidence: 0.75
"""
from __future__ import annotations

from typing import Dict, List, Any, Callable, Optional
import operator
import re


_OPS: Dict[str, Callable] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _resolve_path(ctx: Dict[str, Any], path: str) -> Any:
    """Resolve a dotted path like 'features.avg_speed' against the context."""
    cur: Any = ctx
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
        if cur is None:
            return None
    return cur


def _eval_condition(value: Any, spec: Any) -> bool:
    """
    Evaluate a single condition. A bare value implies equality, while
    strings of the form '>120', '<0.05', '==X' imply comparisons.
    """
    if isinstance(spec, str) and spec.startswith((">", "<", "==", "!=")):
        for op_sym, op_fn in _OPS.items():
            if spec.startswith(op_sym):
                rhs_str = spec[len(op_sym):].strip()
                try:
                    rhs = float(rhs_str)
                    return op_fn(float(value), rhs)
                except ValueError:
                    return op_fn(str(value), rhs_str)
        return False
    if isinstance(spec, list):
        return value in spec
    return value == spec


def _matches(ctx: Dict[str, Any], when: Dict[str, Any]) -> bool:
    for key, spec in when.items():
        if key.endswith("_contains"):
            base = key[: -len("_contains")]
            v = _resolve_path(ctx, base)
            if v is None or not isinstance(v, str):
                return False
            if not re.search(str(spec), v, re.IGNORECASE):
                return False
            continue
        if key.endswith("_regex"):
            base = key[: -len("_regex")]
            v = _resolve_path(ctx, base)
            if v is None or not isinstance(v, str):
                return False
            if not re.search(str(spec), v):
                return False
            continue
        v = _resolve_path(ctx, key)
        if v is None:
            return False
        if not _eval_condition(v, spec):
            return False
    return True


class Rule:
    """A single suspicious-activity rule."""
    def __init__(self, spec: Dict[str, Any]):
        self.name: str = spec.get("name", "unnamed_rule")
        self.when: Dict[str, Any] = spec.get("when", {}) or {}
        self.then: Dict[str, Any] = spec.get("then", {}) or {}

    def matches(self, ctx: Dict[str, Any]) -> bool:
        return _matches(ctx, self.when)

    def produce(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "activity": self.then.get("activity", "suspicious"),
            "severity": self.then.get("severity", "LOW"),
            "confidence": float(self.then.get("confidence", 0.5)),
            "message": self.then.get("message", f"Rule {self.name} triggered"),
        }


class RuleEngine:
    """Evaluates a list of Rule instances against contexts."""
    def __init__(self, rules: Optional[List[Dict[str, Any]]] = None):
        self.rules: List[Rule] = []
        if rules:
            self.load(rules)

    def load(self, rules: List[Dict[str, Any]]) -> None:
        self.rules = [Rule(r) for r in rules]

    def evaluate(self, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return list of rule outputs whose `when` conditions match."""
        results: List[Dict[str, Any]] = []
        for r in self.rules:
            try:
                if r.matches(ctx):
                    results.append(r.produce())
            except Exception:
                # Rule evaluation should never crash the pipeline
                continue
        return results
