"""Seeded basin search for mirror-complete Berger-space candidates."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from mpmath import mp

from problem import DEFAULT_CONFIG, SolverConfig
from solver.mirror_refinement import CandidateTrack, NewtonSettings, newton_refine
from solver.mirror_shooting import BASE_POINT, MirrorResidualResult, MirrorSearchPoint, mirror_residual, params_from_scaled


SCOUT_CONFIG = SolverConfig(4, 30, 15, mp.mpf("0.95"), 0, DEFAULT_CONFIG.match_t)
ORDER6_CONFIG = SolverConfig(6, 40, 20, mp.mpf("0.9"), 0, DEFAULT_CONFIG.match_t)
ORDER10_CONFIG = SolverConfig(10, 70, 30, mp.mpf("0.7"), 1, DEFAULT_CONFIG.match_t)
VERIFY14_CONFIG = SolverConfig(14, 90, 35, mp.mpf("0.6"), 2, DEFAULT_CONFIG.match_t)
VERIFY18_CONFIG = SolverConfig(18, 110, 45, mp.mpf("0.55"), 2, DEFAULT_CONFIG.match_t)
REFINE_CONFIG = ORDER6_CONFIG

ORDER6_SETTINGS = NewtonSettings("order-6", ORDER6_CONFIG, mp.mpf("1e-3"), mp.mpf("1e-8"), 3)
ORDER10_SETTINGS = NewtonSettings("order-10", ORDER10_CONFIG, mp.mpf("3e-4"), mp.mpf("1e-10"), 3)
VERIFY_CONFIGS = (VERIFY14_CONFIG, VERIFY18_CONFIG)

RANDOM_SEED = 1729
NEAR_RANDOM_SAMPLES = 160
MIDDLE_RANDOM_SAMPLES = 120
FAR_RANDOM_SAMPLES = 80
KEEP_OVERALL = 30
KEEP_PER_REGION = 10
REFINE_SEEDS = 6
PROMOTE_SEEDS = 2


@dataclass(frozen=True)
class SearchSeed:
    """One deterministic scout point and its search-region label."""

    region: str
    point: MirrorSearchPoint


@dataclass(frozen=True)
class SearchCandidate:
    """One evaluated scout seed."""

    seed: SearchSeed
    result: MirrorResidualResult


def _point_distance(point: MirrorSearchPoint) -> mp.mpf:
    """Return max-distance from Berger in scaled coordinates."""
    return max(abs(point.u), abs(point.v), abs(point.r), abs(point.s))


def _point_from_values(values) -> MirrorSearchPoint:
    """Build one search point from four values."""
    return MirrorSearchPoint(*(mp.mpf(value) for value in values))


def _box_point(rng: Random) -> MirrorSearchPoint:
    """Sample one point from the fixed scout box."""
    return _point_from_values(
        (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-3, 3), rng.uniform(-1, 1))
    )


def _corner_seeds() -> list[SearchSeed]:
    """Return the fixed corner seeds from the original scout search."""
    seeds = []
    for u in (-0.7, 0.7):
        for v in (-0.7, 0.7):
            for r in (-2.0, 2.0):
                for s in (-0.7, 0.7):
                    seeds.append(SearchSeed("corners", _point_from_values((u, v, r, s))))
    return seeds


def _annular_seeds(region: str, lower: mp.mpf, upper: mp.mpf, count: int, rng: Random) -> list[SearchSeed]:
    """Return reproducible random seeds whose max-distance lies in one annulus."""
    seeds = []
    attempts = 0
    while len(seeds) < count and attempts < count * 1000:
        attempts += 1
        point = _box_point(rng)
        distance = _point_distance(point)
        if lower <= distance <= upper:
            seeds.append(SearchSeed(region, point))
    if len(seeds) != count:
        raise RuntimeError(f"Could not sample enough points for region {region!r}.")
    return seeds


def _search_seeds(seed: int = RANDOM_SEED) -> list[SearchSeed]:
    """Return the full deterministic scout seed list."""
    rng = Random(seed)
    seeds = _corner_seeds()
    seeds += _annular_seeds("near", mp.mpf("0.25"), mp.one, NEAR_RANDOM_SAMPLES, rng)
    seeds += _annular_seeds("middle", mp.one, mp.mpf("2.0"), MIDDLE_RANDOM_SAMPLES, rng)
    seeds += _annular_seeds("far", mp.mpf("2.0"), mp.mpf("3.0"), FAR_RANDOM_SAMPLES, rng)
    return seeds


def _evaluate_seed(seed: SearchSeed, config: SolverConfig = SCOUT_CONFIG) -> SearchCandidate:
    """Evaluate one seed with the requested residual config."""
    with mp.workdps(config.working_dps):
        return SearchCandidate(seed, mirror_residual(seed.point, config))


def _sort_key(candidate: SearchCandidate) -> tuple[bool, mp.mpf]:
    """Sort successful residuals before branch failures."""
    return candidate.result.failure is not None, candidate.result.residual_norm


def _successful(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """Return branch-valid candidates sorted by residual norm."""
    return [candidate for candidate in sorted(candidates, key=_sort_key) if candidate.result.failure is None]


def _format_point(point: MirrorSearchPoint) -> str:
    """Format one scaled point."""
    return f"u={mp.nstr(point.u, 8)}, v={mp.nstr(point.v, 8)}, r={mp.nstr(point.r, 8)}, s={mp.nstr(point.s, 8)}"


def _parameter_ratios(point: MirrorSearchPoint) -> str:
    """Format the physical parameter ratios relative to Berger."""
    return (
        f"a/aB={mp.nstr(mp.e**point.u, 8)}, c/cB={mp.nstr(mp.e**point.v, 8)}, "
        f"alpha/alphaB={mp.nstr(1 + point.r, 8)}, m/mB={mp.nstr(mp.e**point.s, 8)}"
    )


def _verify_point(point: MirrorSearchPoint) -> tuple[MirrorResidualResult, ...]:
    """Evaluate one point at the high-order verification configs."""
    results = []
    for config in VERIFY_CONFIGS:
        with mp.workdps(config.working_dps):
            results.append(mirror_residual(point, config))
    return tuple(results)


def _track_final_result(track: CandidateTrack) -> MirrorResidualResult:
    """Return the latest residual result in one refinement track."""
    return track.stages[-1].final if track.stages else track.scout_result


def _track_sort_key(track: CandidateTrack) -> tuple[bool, mp.mpf]:
    """Sort tracks by the final residual of their latest stage."""
    result = _track_final_result(track)
    return result.failure is not None, result.residual_norm


def _has_failure(track: CandidateTrack) -> bool:
    """Return whether any required stage or verification failed."""
    stage_failed = any(stage.final.failure for stage in track.stages)
    verification_failed = any(result.failure for result in track.verifications)
    return track.scout_result.failure is not None or stage_failed or verification_failed


def _verification_norms(track: CandidateTrack) -> tuple[mp.mpf, ...]:
    """Return finite verification norms for a completed track."""
    return tuple(result.residual_norm for result in track.verifications if result.failure is None)


def _stable_within_factor(norms: tuple[mp.mpf, ...], factor: mp.mpf) -> bool:
    """Return whether nonzero norms are stable within a multiplicative factor."""
    if len(norms) < 2:
        return False
    positive = [norm for norm in norms if norm != 0]
    return not positive or max(positive) <= factor * min(positive)


def _comparable_to_berger(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...]) -> bool:
    """Return whether verification residuals are comparable to Berger errors."""
    if len(track.verifications) != len(berger_refs):
        return False
    for result, berger in zip(track.verifications, berger_refs):
        threshold = max(mp.mpf("1e-8"), mp.mpf("100") * berger.residual_norm)
        if result.failure or result.residual_norm > threshold:
            return False
    return True


def _classify_track(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...]) -> str:
    """Classify one refinement track by distance and verification behavior."""
    if _has_failure(track):
        return "branch_failure"
    norms = _verification_norms(track)
    final = _track_final_result(track)
    distance = _point_distance(final.point)
    if distance >= mp.mpf("0.05") and max(norms or (mp.inf,)) < mp.mpf("1e-6") and _stable_within_factor(norms, mp.mpf("10")):
        return "possible_candidate"
    if final.residual_norm < mp.mpf("1e-8") and max(norms or (mp.zero,)) > mp.mpf("1e-4"):
        return "finite_order_artifact"
    if distance < mp.mpf("0.02") and _comparable_to_berger(track, berger_refs):
        return "flows_to_berger"
    return "inconclusive"


def _replace_track(
    track: CandidateTrack,
    stages,
    verifications,
    classification: str,
) -> CandidateTrack:
    """Return one track with updated refinement data."""
    return CandidateTrack(
        track.seed_rank,
        track.seed_region,
        track.seed_point,
        track.scout_result,
        tuple(stages),
        tuple(verifications),
        classification,
    )


def _initial_track(rank: int, candidate: SearchCandidate) -> CandidateTrack:
    """Run the order-6 refinement stage for one scout seed."""
    stage = newton_refine(candidate.seed.point, ORDER6_SETTINGS)
    return CandidateTrack(rank, candidate.seed.region, candidate.seed.point, candidate.result, (stage,), (), "inconclusive")


def _promote_track(track: CandidateTrack, berger_refs: tuple[MirrorResidualResult, ...]) -> CandidateTrack:
    """Run order-10 refinement and high-order verification for one track."""
    stage = newton_refine(track.stages[-1].final.point, ORDER10_SETTINGS)
    verifications = _verify_point(stage.final.point)
    promoted = _replace_track(track, track.stages + (stage,), verifications, "inconclusive")
    return _replace_track(promoted, promoted.stages, promoted.verifications, _classify_track(promoted, berger_refs))


def _print_candidate(prefix: str, candidate: SearchCandidate) -> None:
    """Print one scout candidate row."""
    result = candidate.result
    status = result.failure or f"norm={mp.nstr(result.residual_norm, 12)}, l={mp.nstr(result.l_value, 12)}"
    distance = mp.nstr(_point_distance(candidate.seed.point), 8)
    print(f"{prefix}: region={candidate.seed.region}, {_format_point(candidate.seed.point)}, distance={distance}, {status}")


def _print_region_summary(candidates: list[SearchCandidate]) -> None:
    """Print success and failure counts by scout region."""
    regions = sorted({candidate.seed.region for candidate in candidates})
    for region in regions:
        subset = [candidate for candidate in candidates if candidate.seed.region == region]
        successes = [candidate for candidate in subset if candidate.result.failure is None]
        print(f"{region}: total={len(subset)}, successes={len(successes)}, failures={len(subset) - len(successes)}")


def _print_best_by_region(candidates: list[SearchCandidate]) -> None:
    """Print the best scout candidates within each region."""
    print("\nbest residuals by region:")
    for region in sorted({candidate.seed.region for candidate in candidates}):
        successes = _successful([candidate for candidate in candidates if candidate.seed.region == region])
        for index, candidate in enumerate(successes[:KEEP_PER_REGION], start=1):
            _print_candidate(f"{region} {index:02d}", candidate)


def _print_stage(stage) -> None:
    """Print one refinement stage summary."""
    final = stage.final
    print(
        f"  {stage.settings.name}: status={stage.status}, steps={len(stage.steps)}, "
        f"norm={mp.nstr(final.residual_norm, 12)}, {_format_point(final.point)}"
    )


def _print_track(track: CandidateTrack) -> None:
    """Print one complete refinement track."""
    print(f"\nseed {track.seed_rank:02d}: region={track.seed_region}, scout_norm={mp.nstr(track.scout_result.residual_norm, 12)}")
    print(f"  scout point: {_format_point(track.seed_point)}")
    for stage in track.stages:
        _print_stage(stage)
    for result in track.verifications:
        print(f"  verify order={result.config.series_order}: norm={mp.nstr(result.residual_norm, 12)}, failure={result.failure}")
    final = _track_final_result(track)
    print(f"  distance={mp.nstr(_point_distance(final.point), 12)}, {_parameter_ratios(final.point)}")
    print(f"  classification={track.classification}")


def _print_berger_references() -> tuple[MirrorResidualResult, ...]:
    """Print Berger residuals at the verification configs."""
    references = _verify_point(BASE_POINT)
    print("\nBerger verification references:")
    for result in references:
        print(f"  order={result.config.series_order}: norm={mp.nstr(result.residual_norm, 12)}, l={mp.nstr(result.l_value, 12)}")
    return references


def _refine_tracks(candidates: list[SearchCandidate], berger_refs: tuple[MirrorResidualResult, ...]) -> list[CandidateTrack]:
    """Run staged refinement on the selected scout candidates."""
    tracks = [_initial_track(index, candidate) for index, candidate in enumerate(candidates[:REFINE_SEEDS], start=1)]
    promoted = {track.seed_rank for track in sorted(tracks, key=_track_sort_key)[:PROMOTE_SEEDS]}
    refined = []
    for track in tracks:
        if track.seed_rank in promoted:
            refined.append(_promote_track(track, berger_refs))
        else:
            refined.append(_replace_track(track, track.stages, (), _classify_track(track, berger_refs)))
    return refined


def main() -> None:
    """Run the seeded mirror search and staged basin classification."""
    print("search box: u,v,s in [-1,1], r in [-3,3]", flush=True)
    print(
        f"samples: corners=16, near={NEAR_RANDOM_SAMPLES}, middle={MIDDLE_RANDOM_SAMPLES}, far={FAR_RANDOM_SAMPLES}, seed={RANDOM_SEED}",
        flush=True,
    )
    print(f"scout config: order={SCOUT_CONFIG.series_order}, dps={SCOUT_CONFIG.working_dps}", flush=True)
    print(f"refinement: order-6 seeds={REFINE_SEEDS}, order-10 promotions={PROMOTE_SEEDS}", flush=True)
    seeds = _search_seeds()
    candidates = []
    for index, seed in enumerate(seeds, start=1):
        candidates.append(_evaluate_seed(seed))
        if index % 25 == 0 or index == len(seeds):
            print(f"evaluated {index}/{len(seeds)} scout seeds", flush=True)
    print("\nscout-region counts:")
    _print_region_summary(candidates)
    successes = _successful(candidates)
    print(f"\nscout totals: successes={len(successes)}, failures={len(candidates) - len(successes)}")
    print("\ntop scout residuals:")
    for index, candidate in enumerate(successes[:KEEP_OVERALL], start=1):
        _print_candidate(f"{index:02d}", candidate)
    _print_best_by_region(candidates)
    berger_refs = _print_berger_references()
    print("\nrefinement tracks:")
    for track in _refine_tracks(successes, berger_refs):
        _print_track(track)


if __name__ == "__main__":
    main()
