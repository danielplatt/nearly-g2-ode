"""Census and triage of cohomogeneity-one actions on S7.

The goal is not to classify every action from first principles.  It is a
practical search-planning layer: record the S7 actions most relevant to the
nearly-G2 exploration, remove already-tested or non-G2-visible cases, compute
the invariant principal-orbit 2- and 3-form dimensions, and rank the remaining
actions by expected scouting value.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


ACTION_CENSUS_VERSION = "s7-action-census-v1"
DEFAULT_REPORT_PATH = Path("docs/s7-action-census.md")


@dataclass(frozen=True)
class InvariantDimensions:
    """Invariant principal-orbit form dimensions for a cohomogeneity-one action."""

    two_forms: int
    three_forms: int

    @property
    def total_g2_functions(self) -> int:
        """Return the number of scalar functions in ``dt^omega + gamma`` form."""
        return self.two_forms + self.three_forms

    @property
    def has_stable_room(self) -> bool:
        """Return whether an invariant G2 ansatz is not ruled out dimensionally."""
        return self.two_forms > 0 and self.three_forms > 0


@dataclass(frozen=True)
class S7ActionCandidate:
    """One S7 cohomogeneity-one action or action family in the census."""

    key: str
    group: str
    principal_orbit: str
    singular_orbits: str
    isotropy_model: str
    source_family: str
    status: str
    duplicate_status: str
    invariant_dimensions: InvariantDimensions | None
    implemented_functions: int | None
    round_visible: str
    squashed_visible: str
    endpoint_profile: str
    endpoint_cost: int
    novelty_score: int
    rationale: str
    next_step: str

    @property
    def stable_room(self) -> bool:
        """Return whether invariant form dimensions leave room for a G2 form."""
        if self.invariant_dimensions is None:
            return self.implemented_functions is not None
        return self.invariant_dimensions.has_stable_room

    @property
    def total_functions(self) -> int | None:
        """Return the effective number of functions if known."""
        if self.invariant_dimensions is not None:
            return self.invariant_dimensions.total_g2_functions
        return self.implemented_functions

    @property
    def known_solution_bonus(self) -> int:
        """Reward actions that visibly contain known calibration solutions."""
        bonus = 0
        if self.round_visible == "yes":
            bonus += 8
        if self.squashed_visible == "yes":
            bonus += 12
        return bonus

    @property
    def rank_score(self) -> float | None:
        """Return a simple score for new viable actions, larger is better."""
        total = self.total_functions
        if self.status != "new-candidate" or not self.stable_room or total is None:
            return None
        return 100 - 2 * total - self.endpoint_cost + self.known_solution_bonus + self.novelty_score


def _so_standard_invariant_degrees(dimension: int, max_degree: int = 3) -> set[int]:
    """Return small-degree invariant exterior degrees for the SO(n) standard rep."""
    degrees = {0}
    if 0 < dimension <= max_degree:
        degrees.add(dimension)
    return degrees


def real_sum_invariant_dimensions(left_sphere_dim: int, right_sphere_dim: int) -> InvariantDimensions:
    """Compute form dimensions for SO(p+1)xSO(q+1) on S^p x S^q.

    The principal isotropy is SO(p)xSO(q), and the tangent representation is
    the direct sum of the two standard representations.  In degrees <= 3 the
    only SO(n)-standard invariants are constants and, when n <= 3, the volume
    form.
    """
    left_degrees = _so_standard_invariant_degrees(left_sphere_dim)
    right_degrees = _so_standard_invariant_degrees(right_sphere_dim)

    def count(total_degree: int) -> int:
        return sum(
            1
            for degree in range(total_degree + 1)
            if degree in left_degrees and total_degree - degree in right_degrees
        )

    return InvariantDimensions(two_forms=count(2), three_forms=count(3))


def s3x_s3_ladder_dimensions(isotropy: str) -> InvariantDimensions:
    """Return invariant dimensions for the S3 x S3 symmetry ladder on S7."""
    if isotropy == "sp1_diag":
        return InvariantDimensions(two_forms=1, three_forms=4)
    if isotropy == "u1_diag":
        return InvariantDimensions(two_forms=5, three_forms=8)
    if isotropy == "finite":
        return InvariantDimensions(two_forms=math.comb(6, 2), three_forms=math.comb(6, 3))
    raise ValueError(f"unknown S3xS3 isotropy model: {isotropy}")


def g2_principal_s6_dimensions() -> InvariantDimensions:
    """Return SU(3)-invariant dimensions on the G2/SU(3) principal S6."""
    return InvariantDimensions(two_forms=1, three_forms=2)


def su3_u1_s5_s1_dimensions() -> InvariantDimensions:
    """Return SU(2)-invariant dimensions for the S5 x S1 principal orbit."""
    # At a regular point in C^3+C, the principal isotropy is SU(2).  The
    # principal tangent representation is C^2 plus two trivial real lines.
    # SU(2) fixes the hyperkahler triple on C^2, giving 3 invariant 2-forms.
    # Add the area form of the two trivial lines.  Wedge each invariant 2-form
    # on C^2 with either trivial line for the invariant 3-forms.
    return InvariantDimensions(two_forms=4, three_forms=6)


def action_candidates() -> tuple[S7ActionCandidate, ...]:
    """Return the curated S7 action census."""
    candidates: list[S7ActionCandidate] = [
        S7ActionCandidate(
            key="so4_z2_q_system",
            group="SO(4)",
            principal_orbit="SO(4)/Z_2^2",
            singular_orbits="two lower-dimensional SO(4)-orbits in the q_i chart",
            isotropy_model="finite principal isotropy, specialized q_i invariant chart",
            source_family="existing q-system action",
            status="already-tested",
            duplicate_status="implemented Berger/S7 q_i action",
            invariant_dimensions=None,
            implemented_functions=8,
            round_visible="yes",
            squashed_visible="yes",
            endpoint_profile="implemented p2/p3 and Berger endpoint charts",
            endpoint_cost=0,
            novelty_score=0,
            rationale="This is the S7 action already used by the fixed-chart and full-moduli q_i searches.",
            next_step="Do not restart here unless changing the search objective or endpoint charts.",
        ),
        S7ActionCandidate(
            key="sp1_3_diag_podesta",
            group="Sp(1)^3",
            principal_orbit="S3 x S3",
            singular_orbits="S3 and S3",
            isotropy_model="diagonal Sp(1) acting on two standard R3 summands",
            source_family="Podesta SU(2)^3 action",
            status="already-tested",
            duplicate_status="larger normal extension of the S3 x S3 orbit foliation",
            invariant_dimensions=s3x_s3_ladder_dimensions("sp1_diag"),
            implemented_functions=None,
            round_visible="yes",
            squashed_visible="yes",
            endpoint_profile="codimension-4 endpoints; one-parameter smooth left germ after normalization",
            endpoint_cost=0,
            novelty_score=0,
            rationale="This is the recent five-function Podesta system; it recovered only round and squashed compact closures.",
            next_step="Keep its code as a calibration subfamily for weaker-invariance S3 x S3 actions.",
        ),
        S7ActionCandidate(
            key="sp1_2_u1_intermediate",
            group="Sp(1) x Sp(1) x U(1)",
            principal_orbit="S3 x S3",
            singular_orbits="S3 and S3",
            isotropy_model="diagonal U(1); two trivial lines plus two equal oriented 2-planes",
            source_family="intermediate S3 x S3 symmetry ladder",
            status="new-candidate",
            duplicate_status="same orbit foliation as Podesta, but strictly weaker invariance",
            invariant_dimensions=s3x_s3_ladder_dimensions("u1_diag"),
            implemented_functions=None,
            round_visible="yes",
            squashed_visible="yes",
            endpoint_profile="codimension-4 endpoints on both sides; broader than Podesta but still symmetry-reduced",
            endpoint_cost=12,
            novelty_score=16,
            rationale=(
                "Best balance of novelty and tractability: it contains the Podesta five-function chart as a "
                "subfamily but allows U(1)-invariant deformations on the same S3 x S3 orbit foliation."
            ),
            next_step="Derive the U(1)-invariant SU(3)-structure algebra on S3 x S3 and verify round/squashed restrictions.",
        ),
        S7ActionCandidate(
            key="su3_u1_complex_sum",
            group="S(U(3) x U(1))",
            principal_orbit="S5 x S1",
            singular_orbits="S5 and S1",
            isotropy_model="SU(2) acting on C2 plus two trivial real lines",
            source_family="complex linear sum action on C3 + C",
            status="new-candidate",
            duplicate_status="not equivalent to the tested S3 x S3 or q_i actions",
            invariant_dimensions=su3_u1_s5_s1_dimensions(),
            implemented_functions=None,
            round_visible="yes",
            squashed_visible="not-known",
            endpoint_profile="asymmetric codimension-2 and codimension-6 singular endpoints",
            endpoint_cost=16,
            novelty_score=8,
            rationale=(
                "A moderate 10-function ansatz with a different principal orbit.  Round S7 should calibrate it; "
                "squashed visibility is unclear, so validation is weaker than for the S3 x S3 ladder."
            ),
            next_step="Audit the SU(2)-invariant form basis on S5 x S1 and check whether squashed S7 is invariant.",
        ),
        S7ActionCandidate(
            key="sp1_2_left_sum",
            group="Sp(1) x Sp(1)",
            principal_orbit="S3 x S3",
            singular_orbits="S3 and S3",
            isotropy_model="finite principal isotropy; all left-invariant forms on S3 x S3",
            source_family="minimal S3 x S3 sum action on H + H",
            status="new-candidate",
            duplicate_status="same orbit foliation as Podesta, weakest invariance in this ladder",
            invariant_dimensions=s3x_s3_ladder_dimensions("finite"),
            implemented_functions=None,
            round_visible="yes",
            squashed_visible="yes",
            endpoint_profile="codimension-4 endpoints with large endpoint smoothness representation",
            endpoint_cost=24,
            novelty_score=25,
            rationale=(
                "Maximal search space on the S3 x S3 foliation, but the 35-function raw ansatz is probably too "
                "large until the U(1)-intermediate case has taught us the algebra."
            ),
            next_step="Park until the U(1)-intermediate action is understood, then use it as the full ladder endpoint.",
        ),
        S7ActionCandidate(
            key="g2_principal_s6",
            group="G2",
            principal_orbit="S6 = G2/SU(3)",
            singular_orbits="point and point",
            isotropy_model="SU(3)-structure on S6",
            source_family="simple-group action fixing the real octonion coordinate",
            status="new-candidate",
            duplicate_status="not equivalent, but expected to be uniqueness-rigid",
            invariant_dimensions=g2_principal_s6_dimensions(),
            implemented_functions=None,
            round_visible="yes",
            squashed_visible="no",
            endpoint_profile="two point singular orbits; very small sine-cone style endpoint problem",
            endpoint_cost=4,
            novelty_score=-45,
            rationale=(
                "Tiny and excellent as a sanity check, but Cleyton-Swann's simple-group picture makes it a poor "
                "place to expect a new compact nearly-parallel G2 structure."
            ),
            next_step="Use only as a calibration exercise if a very small new ODE is desired.",
        ),
    ]

    for p, q in ((1, 5), (2, 4), (3, 3)):
        dims = real_sum_invariant_dimensions(p, q)
        candidates.append(
            S7ActionCandidate(
                key=f"real_sum_s{p}_s{q}",
                group=f"SO({p + 1}) x SO({q + 1})",
                principal_orbit=f"S{p} x S{q}",
                singular_orbits=f"S{p} and S{q}",
                isotropy_model=f"SO({p}) x SO({q}) standard product",
                source_family="real linear sum action on R^{p+1}+R^{q+1}",
                status="discard",
                duplicate_status="symmetric-space linear action with no invariant G2-form room",
                invariant_dimensions=dims,
                implemented_functions=None,
                round_visible="no",
                squashed_visible="no",
                endpoint_profile="linear sphere endpoints; dimensionally fails invariant G2 ansatz",
                endpoint_cost=0,
                novelty_score=0,
                rationale=(
                    "The invariant principal 2- and 3-form dimensions do not both survive, so a stable "
                    "invariant G2 form cannot be built in the dt^omega+gamma ansatz."
                ),
                next_step="Discard for nearly-G2 scouting.",
            )
        )
    return tuple(candidates)


def ranked_new_actions() -> tuple[S7ActionCandidate, ...]:
    """Return viable new actions sorted by the census score."""
    viable = [
        candidate
        for candidate in action_candidates()
        if candidate.status == "new-candidate" and candidate.rank_score is not None
    ]
    return tuple(sorted(viable, key=lambda candidate: (-candidate.rank_score, candidate.key)))


def build_summary() -> dict:
    """Return a JSON-ready summary of steps 1-5."""
    candidates = action_candidates()
    ranked = ranked_new_actions()
    return {
        "version": ACTION_CENSUS_VERSION,
        "steps": {
            "1_list_candidates": [candidate.key for candidate in candidates],
            "2_remove_duplicates": {
                "already_tested": [candidate.key for candidate in candidates if candidate.status == "already-tested"],
                "discarded": [candidate.key for candidate in candidates if candidate.status == "discard"],
                "new_viable": [candidate.key for candidate in ranked],
            },
            "3_invariant_dimensions": {
                candidate.key: (
                    None
                    if candidate.invariant_dimensions is None
                    else asdict(candidate.invariant_dimensions)
                    | {"total_g2_functions": candidate.invariant_dimensions.total_g2_functions}
                )
                for candidate in candidates
            },
            "4_stable_ansatz_and_known_solutions": {
                candidate.key: {
                    "stable_room": candidate.stable_room,
                    "round_visible": candidate.round_visible,
                    "squashed_visible": candidate.squashed_visible,
                }
                for candidate in candidates
            },
            "5_ranked_actions": [
                {
                    "rank": index,
                    "key": candidate.key,
                    "rank_score": candidate.rank_score,
                    "total_functions": candidate.total_functions,
                    "endpoint_profile": candidate.endpoint_profile,
                    "next_step": candidate.next_step,
                }
                for index, candidate in enumerate(ranked, start=1)
            ],
        },
        "recommendation": {
            "top_choice": ranked[0].key if ranked else None,
            "why": ranked[0].rationale if ranked else None,
            "next_action": ranked[0].next_step if ranked else None,
        },
        "candidates": [
            {
                **asdict(candidate),
                "stable_room": candidate.stable_room,
                "total_functions": candidate.total_functions,
                "rank_score": candidate.rank_score,
            }
            for candidate in candidates
        ],
    }


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def render_markdown(summary: dict) -> str:
    """Render the census as a durable planning note."""
    lines: list[str] = [
        "# S7 Cohomogeneity-One Action Census",
        "",
        "Reproducibility command:",
        "",
        "```zsh",
        ".venv/bin/python -m experiments.s7_action_census --write-markdown docs/s7-action-census.md",
        "```",
        "",
        "## Executive recommendation",
        "",
        f"Top next action: `{summary['recommendation']['top_choice']}`.",
        "",
        summary["recommendation"]["why"] or "",
        "",
        "Immediate task:",
        "",
        f"> {summary['recommendation']['next_action']}",
        "",
        "The point is to leave the large-|a| proof route alone and instead test a new S7 symmetry reduction.  The best first target is not a wholly unrelated topology, but a weaker-invariance S3 x S3 action that contains the already-tested Podesta chart as a calibration subfamily.",
        "",
        "## Step 1: Candidate actions",
        "",
        "| key | group | principal orbit | singular orbits | source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for candidate in summary["candidates"]:
        lines.append(
            "| {key} | {group} | {principal_orbit} | {singular_orbits} | {source_family} |".format(
                **candidate
            )
        )

    lines += [
        "",
        "## Step 2: Duplicate and viability filter",
        "",
        "| key | status | duplicate/equivalence note | rationale |",
        "| --- | --- | --- | --- |",
    ]
    for candidate in summary["candidates"]:
        lines.append(
            "| {key} | {status} | {duplicate_status} | {rationale} |".format(**candidate)
        )

    lines += [
        "",
        "## Step 3: Invariant form dimensions",
        "",
        "| key | invariant 2-forms | invariant 3-forms | total functions | stable room? |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for candidate in summary["candidates"]:
        dims = candidate["invariant_dimensions"]
        two_forms = "n/a" if dims is None else dims["two_forms"]
        three_forms = "n/a" if dims is None else dims["three_forms"]
        lines.append(
            f"| `{candidate['key']}` | {two_forms} | {three_forms} | {_fmt(candidate['total_functions'])} | {candidate['stable_room']} |"
        )

    lines += [
        "",
        "## Step 4: Known-solution visibility",
        "",
        "| key | round visible | squashed visible | endpoint profile |",
        "| --- | --- | --- | --- |",
    ]
    for candidate in summary["candidates"]:
        lines.append(
            "| {key} | {round_visible} | {squashed_visible} | {endpoint_profile} |".format(
                **candidate
            )
        )

    lines += [
        "",
        "## Step 5: Ranking",
        "",
        "| rank | key | score | total functions | reason to do it next |",
        "| ---: | --- | ---: | ---: | --- |",
    ]
    ranked_by_key = {
        item["key"]: item for item in summary["steps"]["5_ranked_actions"]
    }
    for item in summary["steps"]["5_ranked_actions"]:
        candidate = next(candidate for candidate in summary["candidates"] if candidate["key"] == item["key"])
        lines.append(
            f"| {item['rank']} | `{item['key']}` | {_fmt(item['rank_score'])} | {item['total_functions']} | {candidate['rationale']} |"
        )

    lines += [
        "",
        "Actions not in the ranking are either already tested or dimensionally unsuitable for an invariant G2 ansatz.",
        "",
        "## Practical next sprint",
        "",
        "1. Work on `sp1_2_u1_intermediate` first.",
        "2. Build the U(1)-invariant 2-form and 3-form basis on the S3 x S3 principal orbit.",
        "3. Restrict that basis to the diagonal Sp(1)-invariant subspace and verify that it reproduces the existing Podesta five-function chart.",
        "4. Express the round and squashed S7 homogeneous solutions in the new coordinates.",
        "5. Only after those two calibrations pass, derive endpoint smoothness and a cheap scout.",
        "",
        "## References Used",
        "",
        "- Hoelscher, `Classification of Cohomogeneity One Manifolds in Low Dimensions`, arXiv:0712.1327.",
        "- Cleyton-Swann, `Cohomogeneity-one G2-structures`, arXiv:math/0111056.",
        "- Podesta, `Nearly parallel G2-structures with large symmetry group`, arXiv:1905.03077.",
        "- Existing local audits: `docs/s7-su2-cubed-action-audit.md`, `docs/s7-su2-cubed-podesta-scout.md`, and `docs/2026-07-07-handover.md`.",
        "",
    ]
    # Keep the local variable used so lint-like checks do not mistake it for a typo in the table construction.
    assert ranked_by_key or not summary["steps"]["5_ranked_actions"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Run the S7 action census."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--write-markdown", type=Path, help="write a Markdown census report")
    args = parser.parse_args(argv)
    summary = build_summary()
    if args.write_markdown:
        args.write_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.write_markdown.write_text(render_markdown(summary), encoding="utf-8")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif not args.write_markdown:
        print(render_markdown(summary))


if __name__ == "__main__":
    main()
