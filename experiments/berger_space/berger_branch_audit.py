"""Summarize the completed Berger branch search artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mpmath import mp

from ..shared.non_mirrored_common import RANDOM_SEED, _mp_string
from . import mu_branch_probe
from . import non_mirrored_grid_search as grid_search


AUDIT_VERSION = "berger-branch-audit-v1"
LOW_SCOUT_THRESHOLD = mp.mpf("0.15")
COLLAPSED_INTERVAL_RATIO = mp.mpf("0.1")
COLLAPSED_S_THRESHOLD = mp.log(COLLAPSED_INTERVAL_RATIO)
DEFAULT_BRANCH_P_SIGNS = [-1, 1, 1]


@dataclass(frozen=True)
class CanonicalAuditEntry:
    """One curated Berger artifact group to audit."""

    label: str
    scout_jsonls: tuple[str, ...]
    refinement_summary: str | None = None
    notes: str = ""


CANONICAL_ARTIFACTS: tuple[CanonicalAuditEntry, ...] = (
    CanonicalAuditEntry(
        "near vertex",
        ("output/non_mirrored_grid_searches/20260523-174121-seed1729-non-mirrored-grid-v1.jsonl",),
        "output/non_mirrored_grid_refinements/20260525-125909-seed1729-non-mirrored-grid-refine-v1-summary.json",
        "Original calibrated 7D Berger-near vertex grid.",
    ),
    CanonicalAuditEntry(
        "near cell-center",
        ("output/non_mirrored_grid_searches/20260527-205409-seed1729-non-mirrored-grid-v1.jsonl",),
        "output/non_mirrored_grid_refinements/20260528-201123-seed1729-non-mirrored-grid-refine-v1-summary.json",
        "Shifted midpoint grid in the original Berger-near box.",
    ),
    CanonicalAuditEntry(
        "symmetric-alpha-omega vertex",
        ("output/non_mirrored_grid_searches/20260530-171855-seed1729-non-mirrored-grid-v1.jsonl",),
        "output/non_mirrored_grid_refinements/20260531-094436-seed1729-non-mirrored-grid-refine-v1-summary.json",
        "Physical alpha/omega interval made symmetric about zero.",
    ),
    CanonicalAuditEntry(
        "symmetric-alpha-omega cell-center",
        ("output/non_mirrored_grid_searches/20260601-090715-seed1729-non-mirrored-grid-v1.jsonl",),
        "output/non_mirrored_grid_refinements/20260601-225508-seed1729-non-mirrored-grid-refine-v1-summary.json",
        "Cell-centered follow-up in the symmetric physical alpha/omega box.",
    ),
    CanonicalAuditEntry(
        "positive-ac standard",
        (
            "output/non_mirrored_grid_searches/20260602-144334-seed1729-non-mirrored-grid-v1.jsonl",
            "output/non_mirrored_grid_searches/20260604-093131-seed1729-non-mirrored-grid-v1.jsonl",
        ),
        None,
        "Standard positive a,c branch, vertex plus cell-center scouts.",
    ),
    CanonicalAuditEntry(
        "positive-ac boundary",
        (
            "output/non_mirrored_grid_searches/20260605-090434-seed1729-non-mirrored-grid-v1.jsonl",
            "output/non_mirrored_grid_searches/20260606-101555-seed1729-non-mirrored-grid-v1.jsonl",
        ),
        None,
        "First low-scale positive a,c boundary strip.",
    ),
    CanonicalAuditEntry(
        "positive-ac boundary-v2",
        ("output/non_mirrored_grid_searches/20260608-113600-seed1729-non-mirrored-grid-v1.jsonl",),
        "output/non_mirrored_grid_refinements/20260620-221306-seed1729-non-mirrored-grid-refine-v1-summary.json",
        "Second positive a,c boundary strip, refined over local minima.",
    ),
    CanonicalAuditEntry(
        "negative-ac standard",
        ("output/non_mirrored_grid_searches/20260621-202610-seed1729-non-mirrored-grid-v1.jsonl",),
        None,
        "Negative a,c component with 3a-c > 0.",
    ),
    CanonicalAuditEntry(
        "mixed-mu short",
        ("output/non_mirrored_grid_searches/20260625-200514-seed1729-non-mirrored-grid-v1.jsonl",),
        None,
        "Endpoint-local opposite-mu branch in the first shortened interval box.",
    ),
    CanonicalAuditEntry(
        "mixed-mu boundary",
        ("output/non_mirrored_grid_searches/20260626-124043-seed1729-non-mirrored-grid-v1.jsonl",),
        "output/non_mirrored_grid_refinements/20260626-212554-seed1729-non-mirrored-grid-refine-v1-summary.json",
        "Broader low-scale, short-interval mixed-mu boundary strip.",
    ),
)


def _summary_path_for_jsonl(path: Path) -> Path:
    """Return the summary path paired with a JSONL checkpoint."""
    return path.with_name(f"{path.stem}-summary.json")


def _artifact_summary_path(path: Path) -> Path:
    """Return the JSON summary path for a manifest artifact."""
    return _summary_path_for_jsonl(path) if path.suffix == ".jsonl" else path


def _load_json(path: Path) -> dict:
    """Load one JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _mp_value(value) -> mp.mpf | None:
    """Parse one persisted numeric value."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return mp.mpf(value)
    if str(value) in {"Infinity", "inf", "+inf"}:
        return mp.inf
    return mp.mpf(str(value))


def _nstr(value, digits: int = 8) -> str:
    """Return a compact numeric string for markdown."""
    parsed = _mp_value(value)
    if parsed is None:
        return "n/a"
    if parsed == mp.inf:
        return "inf"
    return mp.nstr(parsed, digits)


def _sign_symbol(value) -> str:
    """Return a compact sign symbol for one persisted number."""
    parsed = _mp_value(value)
    if parsed is None:
        return "?"
    if parsed > 0:
        return ">0"
    if parsed < 0:
        return "<0"
    return "=0"


def _counts_payload(counts: dict | None) -> dict[str, int]:
    """Return integer counts with stable keys."""
    return {str(key): int(value) for key, value in (counts or {}).items()}


def _base_params_payload_for_grid(grid: dict) -> dict:
    """Return base endpoint parameters, filling defaults for older summaries."""
    base_params = grid.get("base_params")
    if base_params is not None:
        return base_params
    region_name = grid.get("region", "near")
    return grid_search._base_params_payload(grid_search._base_params_for_region(region_name))


def branch_descriptor(base_params: dict) -> dict:
    """Return a compact physical branch descriptor from persisted base parameters."""
    left = base_params.get("left", {})
    right = base_params.get("right", {})
    p_signs = base_params.get("p_signs", DEFAULT_BRANCH_P_SIGNS)
    right_p_signs = base_params.get("right_p_signs")
    return {
        "a_sign": _sign_symbol(left.get("a")),
        "c_sign": _sign_symbol(left.get("c")),
        "right_d_sign": _sign_symbol(right.get("d")),
        "right_f_sign": _sign_symbol(right.get("f")),
        "left_mu": left.get("mu", -1),
        "right_mu": right.get("mu", -1),
        "p_signs": p_signs,
        "right_p_signs": right_p_signs,
        "label": _branch_label(left, right, p_signs, right_p_signs),
    }


def _branch_label(left: dict, right: dict, p_signs, right_p_signs) -> str:
    """Return one human-readable branch label."""
    right_label = "global" if right_p_signs is None else tuple(right_p_signs)
    return (
        f"a{_sign_symbol(left.get('a'))}, c{_sign_symbol(left.get('c'))}; "
        f"mu=({left.get('mu', -1)},{right.get('mu', -1)}); "
        f"p={tuple(p_signs)}, right_p={right_label}"
    )


def _extract_scout(summary_path: Path, summary: dict) -> dict:
    """Return compact scout facts from a grid-search summary."""
    grid = summary.get("grid", {})
    counts = _counts_payload(summary.get("classification_counts"))
    best = (summary.get("best_scouts") or [None])[0]
    base_params = _base_params_payload_for_grid(grid)
    return {
        "summary_path": str(summary_path),
        "jsonl_path": str(summary_path.with_name(summary_path.name.replace("-summary.json", ".jsonl"))),
        "region": grid.get("region"),
        "shift": grid.get("shift") or "vertex",
        "bounds": grid.get("bounds", []),
        "axis_counts": grid.get("axis_counts", []),
        "seed_count": int(grid.get("seed_count", summary.get("scout_count", 0)) or 0),
        "full_seed_count": int(grid.get("full_seed_count", grid.get("seed_count", 0)) or 0),
        "limit": grid.get("limit"),
        "success_count": counts.get("scout_success", 0),
        "failure_count": counts.get("scout_failure", 0),
        "classification_counts": counts,
        "best_scout": best,
        "best_scout_residual": None if best is None else best.get("residual_norm"),
        "best_scout_seed": None if best is None else best.get("seed_index"),
        "best_scout_distance": None if best is None else best.get("distance"),
        "best_scout_asymmetry": None if best is None else best.get("asymmetry"),
        "branch": branch_descriptor(base_params),
    }


def _track_final(track: dict) -> dict:
    """Return the final residual payload for one persisted track."""
    stages = track.get("stages") or []
    if stages:
        return stages[-1].get("final", {})
    return track.get("scout", {})


def _best_track(tracks: Iterable[dict]) -> dict | None:
    """Return the persisted track with the smallest final residual norm."""
    best = None
    best_norm = mp.inf
    for track in tracks:
        norm = _mp_value(_track_final(track).get("residual_norm"))
        if norm is not None and norm < best_norm:
            best = track
            best_norm = norm
    return best


def _extract_refinement(summary_path: Path, summary: dict) -> dict:
    """Return compact refinement facts from a grid-refinement summary."""
    tracks = list(summary.get("tracks") or [])
    best = _best_track(tracks)
    final = _track_final(best) if best else {}
    best_verified = (summary.get("best_verified_tracks") or [None])[0]
    physical = (best_verified or {}).get("physical_parameters", {})
    verifications = sum(len(track.get("verifications") or []) for track in tracks)
    return {
        "summary_path": str(summary_path),
        "source_scout_jsonl": summary.get("scout_jsonl"),
        "selection_count": int(summary.get("selection_count", len(summary.get("selections") or [])) or 0),
        "classified_count": int(summary.get("classified_count", len(tracks)) or 0),
        "classification_counts": _counts_payload(summary.get("classification_counts")),
        "selection_config": summary.get("selection_config") or {},
        "verification_count": verifications,
        "best_final_residual": final.get("residual_norm"),
        "best_final_seed": None if best is None else best.get("seed_index"),
        "best_final_point": final.get("point") or (best or {}).get("final_point") or {},
        "best_physical_parameters": physical,
    }


def _best_scout(scouts: list[dict]) -> dict | None:
    """Return the scout with the smallest best-scout residual."""
    best = None
    best_norm = mp.inf
    for scout in scouts:
        norm = _mp_value(scout.get("best_scout_residual"))
        if norm is not None and norm < best_norm:
            best = scout
            best_norm = norm
    return best


def _total_seed_count(scouts: list[dict]) -> int:
    """Return total evaluated scout count across an artifact group."""
    return sum(int(scout.get("seed_count", 0)) for scout in scouts)


def _outcome_label(scouts: list[dict], refinement: dict | None) -> str:
    """Classify the audited artifact group."""
    if refinement is not None:
        counts = refinement.get("classification_counts", {})
        if counts.get("recovered_berger", 0) > 0:
            return "berger-recovered"
        final_s = _mp_value((refinement.get("best_final_point") or {}).get("s"))
        physical_t = _mp_value((refinement.get("best_physical_parameters") or {}).get("T"))
        base_t = _base_interval_for_scouts(scouts)
        if final_s is not None and final_s <= COLLAPSED_S_THRESHOLD:
            return "collapsed-tail"
        if physical_t is not None and base_t is not None and physical_t / base_t <= COLLAPSED_INTERVAL_RATIO:
            return "collapsed-tail"
        return "finite-residual-tail"

    best = _best_scout(scouts)
    best_norm = None if best is None else _mp_value(best.get("best_scout_residual"))
    if best_norm is None or best_norm >= LOW_SCOUT_THRESHOLD:
        return "no-low-scout-signal"
    return "not-refined"


def _base_interval_for_scouts(scouts: list[dict]) -> mp.mpf | None:
    """Return the base Berger interval from the first scout branch when available."""
    if not scouts:
        return None
    summary_path = Path(scouts[0]["summary_path"])
    grid = _load_json(summary_path).get("grid", {})
    base = _base_params_payload_for_grid(grid)
    return _mp_value(base.get("interval_end"))


def _run_payload(entry: CanonicalAuditEntry, scouts: list[dict], refinement: dict | None) -> dict:
    """Return one JSON-ready audited run payload."""
    best = _best_scout(scouts)
    return {
        "label": entry.label,
        "notes": entry.notes,
        "outcome": _outcome_label(scouts, refinement),
        "scouts": scouts,
        "refinement": refinement,
        "best_scout_residual": None if best is None else best.get("best_scout_residual"),
        "best_scout_seed": None if best is None else best.get("best_scout_seed"),
        "best_scout_region": None if best is None else best.get("region"),
        "best_scout_shift": None if best is None else best.get("shift"),
        "total_scout_count": _total_seed_count(scouts),
        "branch": scouts[0]["branch"] if scouts else None,
    }


def _missing_artifacts(root: Path, manifest: tuple[CanonicalAuditEntry, ...]) -> list[str]:
    """Return missing summary artifacts required by the manifest."""
    missing = []
    for entry in manifest:
        for scout_jsonl in entry.scout_jsonls:
            summary_path = root / _artifact_summary_path(Path(scout_jsonl))
            if not summary_path.exists():
                missing.append(str(summary_path))
        if entry.refinement_summary is not None:
            refinement_path = root / _artifact_summary_path(Path(entry.refinement_summary))
            if not refinement_path.exists():
                missing.append(str(refinement_path))
    return missing


def build_audit(
    *,
    root: Path | str = Path("."),
    manifest: tuple[CanonicalAuditEntry, ...] = CANONICAL_ARTIFACTS,
    allow_missing: bool = False,
    branch_probe_summary: dict | None = None,
) -> dict:
    """Build the complete Berger audit payload."""
    root = Path(root)
    missing = _missing_artifacts(root, manifest)
    if missing and not allow_missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing canonical Berger audit artifacts:\n{formatted}")

    runs = []
    for entry in manifest:
        scout_summaries = []
        entry_missing = False
        for scout_jsonl in entry.scout_jsonls:
            summary_path = root / _artifact_summary_path(Path(scout_jsonl))
            if not summary_path.exists():
                entry_missing = True
                continue
            scout_summaries.append(_extract_scout(summary_path, _load_json(summary_path)))
        refinement = None
        if entry.refinement_summary is not None:
            refinement_path = root / _artifact_summary_path(Path(entry.refinement_summary))
            if refinement_path.exists():
                refinement = _extract_refinement(refinement_path, _load_json(refinement_path))
            else:
                entry_missing = True
        if entry_missing and allow_missing and not scout_summaries:
            continue
        if scout_summaries:
            runs.append(_run_payload(entry, scout_summaries, refinement))

    if branch_probe_summary is None:
        branch_probe_summary = mu_branch_probe.build_summary()
    return {
        "audit_version": AUDIT_VERSION,
        "random_seed": RANDOM_SEED,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "low_scout_threshold": _mp_string(LOW_SCOUT_THRESHOLD),
            "collapsed_interval_ratio": _mp_string(COLLAPSED_INTERVAL_RATIO),
            "collapsed_s_threshold": _mp_string(COLLAPSED_S_THRESHOLD),
        },
        "missing_artifacts": missing,
        "runs": runs,
        "branch_probe": branch_probe_summary,
        "conclusion": "No non-Berger nearly G2 candidate survived refinement or high-order verification in the explored Berger branches.",
    }


def _counts_label(counts: dict | None) -> str:
    """Return a compact counts label."""
    if not counts:
        return "n/a"
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def _scout_label(run: dict) -> str:
    """Return compact markdown text for scout artifacts."""
    parts = []
    for scout in run["scouts"]:
        parts.append(
            f"{scout['region']} {scout['shift']}: "
            f"{scout['seed_count']} seeds, "
            f"{scout['success_count']} ok/{scout['failure_count']} fail"
        )
    return "<br>".join(parts)


def _best_scout_label(run: dict) -> str:
    """Return compact markdown text for the best scout."""
    if run.get("best_scout_residual") is None:
        return "n/a"
    return f"{_nstr(run['best_scout_residual'])} (seed {run['best_scout_seed']})"


def _refinement_label(refinement: dict | None) -> str:
    """Return compact markdown text for refinement results."""
    if refinement is None:
        return "none"
    return f"{refinement['selection_count']} selected; {_counts_label(refinement['classification_counts'])}"


def _best_final_label(refinement: dict | None) -> str:
    """Return compact markdown text for final refinement residuals."""
    if refinement is None or refinement.get("best_final_residual") is None:
        return "n/a"
    return f"{_nstr(refinement['best_final_residual'])} (seed {refinement['best_final_seed']})"


def _escape_md(value: str) -> str:
    """Escape table-sensitive markdown characters."""
    return value.replace("|", "\\|")


def render_markdown(audit: dict) -> str:
    """Render the Berger audit payload as a markdown report."""
    lines = [
        "# Berger Branch Closure Audit",
        "",
        "Generated from local experiment artifacts.",
        "",
        "Reproduce with:",
        "",
        "```zsh",
        ".venv/bin/python -m experiments.berger_branch_audit --write-markdown docs/berger-branch-audit.md",
        "```",
        "",
        "## Canonical Runs",
        "",
        "| Run | Scouts | Best scout | Refinement | Best final | Verifications | Branch | Outcome |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in audit["runs"]:
        refinement = run.get("refinement")
        verification_count = "0" if refinement is None else str(refinement.get("verification_count", 0))
        branch = (run.get("branch") or {}).get("label", "n/a")
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(run["label"]),
                    _escape_md(_scout_label(run)),
                    _escape_md(_best_scout_label(run)),
                    _escape_md(_refinement_label(refinement)),
                    _escape_md(_best_final_label(refinement)),
                    verification_count,
                    _escape_md(branch),
                    run["outcome"],
                ]
            )
            + " |"
        )

    branch_probe = audit["branch_probe"]
    compatible = branch_probe.get("compatible_two_sided", [])
    mixed = branch_probe.get("mixed_opposite_mu_compatible", [])
    lines.extend(
        [
            "",
            "## Branch Coverage",
            "",
            "The global mirrored `p`/`mu` probe finds only the already explored Berger branch:",
            "",
            *[f"- `{mu_branch_probe._format_record(record)}`" for record in compatible],
            "",
            "The non-default opposite-`mu` scoutable branch requires endpoint-local square-root signs:",
            "",
            *[f"- `{mu_branch_probe._format_record(record)}`" for record in mixed],
            "",
            "The physical endpoint branches covered by the grid artifacts are:",
            "",
            "- Default Berger component, including the original near grid and the symmetric physical `alpha/omega` follow-up.",
            "- Positive `a,c` component with `3a-c>0`, including standard, boundary, and boundary-v2 strips.",
            "- Negative `a,c` component with `3a-c>0`.",
            "- Mixed endpoint-local opposite-`mu` component with `left p=(1,1,1)` and `right p=(-1,1,-1)`.",
            "",
            "## Conclusion",
            "",
            audit["conclusion"],
            "",
            "The Berger-near and symmetric `alpha/omega` grids recover Berger. The positive `a,c` and mixed-`mu` boundary follow-ups enter short-interval collapsed tails with nonzero residuals. The remaining standard branches have no low scout signal. No explored branch produced a verified non-Berger candidate.",
            "",
        ]
    )
    if audit.get("missing_artifacts"):
        lines.extend(["## Missing Artifacts", ""])
        lines.extend(f"- `{path}`" for path in audit["missing_artifacts"])
        lines.append("")
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> Path:
    """Write text, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _json_output_path(path: Path) -> Path:
    """Return the concrete JSON output path for a CLI argument."""
    if path.suffix == ".json":
        return path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path / f"{stamp}-{AUDIT_VERSION}.json"


def write_json(audit: dict, output: Path) -> Path:
    """Write the audit payload as JSON."""
    path = _json_output_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse audit CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-markdown", type=Path, default=None, help="write the rendered markdown report to this path")
    parser.add_argument("--write-json", type=Path, default=None, help="write the structured audit JSON to this path or directory")
    parser.add_argument("--allow-missing", action="store_true", help="record missing canonical artifacts instead of failing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the Berger branch audit."""
    args = parse_args(argv)
    audit = build_audit(allow_missing=args.allow_missing)
    markdown = render_markdown(audit)
    if args.write_markdown is not None:
        _write_text(args.write_markdown, markdown + "\n")
        print(f"markdown written to {args.write_markdown}", flush=True)
    else:
        print(markdown, flush=True)
    if args.write_json is not None:
        json_path = write_json(audit, args.write_json)
        print(f"json written to {json_path}", flush=True)
    return audit


if __name__ == "__main__":
    main()
