"""Static coherence scan of a source package (read-only, AST based).

Finds:
  * config dataclass fields that are never read outside their definition
    (and fields read only by tests),
  * `self.X` attributes that are written but never read anywhere,
  * class-level aliases (`old_name = new_name`) that nothing references,
  * groups of fields in one config class sharing the same non-trivial default value,
    ranked by name similarity (likely duplicated parameters).

A read is any attribute load (`obj.X`) or a getattr/hasattr string that is not a
write pattern. Writes: assignment, augmented assignment, subscript stores
(`self.X[i] = v`), in-place mutator calls (`self.X.fill(...)`) and `out=self.X`.
Matching is by attribute name, so a name shared by two unrelated objects counts as read.

Usage:
  python tools/audit/orphan_scan.py [--source-dir football_rl] [--config config.py] [--json OUT]
"""

from __future__ import annotations

import argparse
import ast
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.audit._common import REPO_ROOT, write_json  # noqa: E402

__all__ = ["scan", "main"]

MUTATORS = frozenset({
    "fill", "append", "extend", "clear", "update", "insert", "remove", "add", "discard", "sort",
    "setdefault", "appendleft", "extendleft", "popitem",
})
TRIVIAL_VALUES = (0, 1, -1)


def _is_test_file(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return "tests" in rel.parts[:-1] or path.name.startswith("test_") or path.name == "conftest.py"


def _parents(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    par: Dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            par[child] = node
    return par


def _is_write_use(node: ast.Attribute, par: Dict[ast.AST, ast.AST]) -> bool:
    if isinstance(node.ctx, (ast.Store, ast.Del)):
        return True
    cur: ast.AST = node
    parent = par.get(cur)
    while isinstance(parent, ast.Subscript) and parent.value is cur:
        if isinstance(parent.ctx, (ast.Store, ast.Del)):
            return True
        cur = parent
        parent = par.get(cur)
    parent = par.get(node)
    if isinstance(parent, ast.keyword) and parent.arg == "out":
        return True
    if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr in MUTATORS:
        call = par.get(parent)
        if isinstance(call, ast.Call) and call.func is parent:
            return True
    return False


def _is_self_attr(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"


class _FileFacts:
    def __init__(self) -> None:
        self.reads: Set[str] = set()
        self.name_loads: Set[str] = set()
        self.self_writes: Set[Tuple[str, str]] = set()   # (class, attr)
        self.aliases: Set[Tuple[str, str]] = set()       # (class, alias name)


def _collect(tree: ast.AST) -> _FileFacts:
    facts = _FileFacts()
    par = _parents(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if _is_write_use(node, par):
                if _is_self_attr(node):
                    cls = _enclosing_class(node, par)
                    if cls is not None:
                        facts.self_writes.add((cls, node.attr))
            else:
                facts.reads.add(node.attr)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            facts.name_loads.add(node.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr"):
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                facts.reads.add(node.args[1].value)
        elif isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                        and isinstance(stmt.value, ast.Name)):
                    facts.aliases.add((node.name, stmt.targets[0].id))
    return facts


def _enclosing_class(node: ast.AST, par: Dict[ast.AST, ast.AST]) -> Optional[str]:
    cur = par.get(node)
    while cur is not None:
        if isinstance(cur, ast.ClassDef):
            return cur.name
        cur = par.get(cur)
    return None


def _config_fields(tree: ast.AST) -> Dict[str, List[Tuple[str, Any]]]:
    """{class_name: [(field, literal_default_or_None)]} for @dataclass classes."""
    out: Dict[str, List[Tuple[str, Any]]] = {}
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.ClassDef):
            continue
        is_dc = any(
            (isinstance(d, ast.Name) and d.id == "dataclass")
            or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
            for d in node.decorator_list
        )
        if not is_dc:
            continue
        fields = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                val: Any = None
                if stmt.value is not None:
                    try:
                        val = ast.literal_eval(stmt.value)
                    except (ValueError, SyntaxError, TypeError):
                        val = None
                fields.append((stmt.target.id, val))
        out[node.name] = fields
    return out


def _tokens(name: str) -> Set[str]:
    return {t for t in name.lower().split("_") if t}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if (ta or tb) else 0.0


def _duplicate_groups(fields: Dict[str, List[Tuple[str, Any]]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for cls, fl in fields.items():
        by_val: Dict[Any, List[str]] = {}
        for name, val in fl:
            if isinstance(val, bool) or val is None:
                continue
            if isinstance(val, (int, float)):
                if val in TRIVIAL_VALUES:
                    continue
                key: Any = float(val)
            elif isinstance(val, str):
                key = val
            else:
                continue
            by_val.setdefault(key, []).append(name)
        for val, names in by_val.items():
            if len(names) < 2:
                continue
            sim = max(_jaccard(a, b) for a, b in combinations(names, 2))
            groups.append({"class": cls, "value": val, "fields": sorted(names), "similarity": round(sim, 4)})
    groups.sort(key=lambda g: (-g["similarity"], g["class"], str(g["value"])))
    return groups


def _parse(path: Path) -> Optional[ast.AST]:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return None


def scan(source_dir: Path | str, config_name: str = "config.py") -> Dict[str, Any]:
    root = Path(source_dir)
    cfg_path = root / config_name
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config module not found: {cfg_path}")
    files = sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)

    src_facts: List[_FileFacts] = []
    test_facts: List[_FileFacts] = []
    cfg_tree: Optional[ast.AST] = None
    for p in files:
        tree = _parse(p)
        if tree is None:
            continue
        if p == cfg_path:
            cfg_tree = tree
        (test_facts if _is_test_file(p, root) else src_facts).append(_collect(tree))
    assert cfg_tree is not None

    def union(fs: Iterable[_FileFacts], attr: str) -> Set[Any]:
        acc: Set[Any] = set()
        for f in fs:
            acc |= getattr(f, attr)
        return acc

    src_reads = union(src_facts, "reads")
    test_reads = union(test_facts, "reads")
    src_names = union(src_facts, "name_loads")

    fields = _config_fields(cfg_tree)
    unread, test_only = [], []
    for cls, fl in fields.items():
        for name, _ in fl:
            if name in src_reads:
                continue
            (test_only if name in test_reads else unread).append(f"{cls}.{name}")

    writes = union(src_facts, "self_writes")
    write_only, attr_test_only = [], []
    for cls, attr in sorted(writes):
        if attr in src_reads:
            continue
        (attr_test_only if attr in test_reads else write_only).append(f"{cls}.{attr}")

    aliases = sorted(f"{c}.{a}" for c, a in union(src_facts, "aliases")
                     if a not in src_reads and a not in src_names and a not in test_reads)

    return {
        "source_dir": str(root),
        "files_scanned": len(files),
        "unread_config_fields": sorted(unread),
        "config_fields_read_only_in_tests": sorted(test_only),
        "write_only_attributes": write_only,
        "attributes_read_only_in_tests": attr_test_only,
        "unused_class_aliases": aliases,
        "duplicate_value_groups": _duplicate_groups(fields),
    }


def _summary(res: Dict[str, Any], cap: int = 15) -> List[str]:
    lines = [f"orphan_scan: {res['source_dir']} ({res['files_scanned']} files)"]
    for key in ("unread_config_fields", "config_fields_read_only_in_tests", "write_only_attributes",
                "attributes_read_only_in_tests", "unused_class_aliases"):
        items = res[key]
        more = f" (+{len(items) - cap} more)" if len(items) > cap else ""
        lines.append(f"  {key} [{len(items)}]: {', '.join(items[:cap])}{more}")
    similar = [g for g in res["duplicate_value_groups"] if g["similarity"] > 0]
    lines.append(f"  duplicate_value_groups [{len(res['duplicate_value_groups'])}], name-similar [{len(similar)}]:")
    for g in similar[:cap]:
        lines.append(f"    {g['class']}: {g['value']} <- {', '.join(g['fields'])} (sim {g['similarity']:.2f})")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", default=str(REPO_ROOT / "football_rl"))
    ap.add_argument("--config", default="config.py")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    res = scan(args.source_dir, args.config)
    print("\n".join(_summary(res)))
    if args.json:
        write_json(args.json, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
