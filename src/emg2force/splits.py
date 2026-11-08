"""Train/test splits over whole recordings, one builder and one set of leakage checks per protocol.

Development subjects (4 drawn at random once, seed DEV_SEED) choose every setting. The other 16 are scored.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from emg2force.hyser import N_RANDOM_TRIALS, N_SESSIONS, N_SUBJECTS, parse_key, recording_key

SUBJECTS = tuple(range(1, N_SUBJECTS + 1))
SESSIONS = tuple(range(1, N_SESSIONS + 1))
TRIALS = tuple(range(1, N_RANDOM_TRIALS + 1))
DEV_SEED = 20261011
DEV_SUBJECTS = (1, 8, 10, 12)  # sorted(np.random.default_rng(DEV_SEED).choice(SUBJECTS, 4, replace=False))
EVAL_SUBJECTS = tuple(s for s in SUBJECTS if s not in DEV_SUBJECTS)
CAL_TRIAL = 1  # session-2 trial used for labelled recalibration
DAY1, DAY2 = 1, 2


# Protocols that train on other people plus the scored person's day 1
POPULATION_PROTOCOLS = frozenset({"cross_day_pop", "cross_day_pop_cal"})


@dataclass(frozen=True)
class Split:
    """One fold: whole train and test recordings, and the person being scored."""

    protocol: str
    name: str
    train: tuple[str, ...]
    test: tuple[str, ...]
    held_out_subject: int

    @property
    def focus_subject(self) -> int | None:
        """The scored person when training pools other people (their crops can be up-weighted). Otherwise None."""
        return self.held_out_subject if self.protocol in POPULATION_PROTOCOLS else None


def own_day1(sp: Split) -> list[str]:
    """The scored person's session-1 recordings in the training set, in training order."""
    return [k for k in sp.train if subject_of(k) == sp.held_out_subject and session_of(k) == DAY1]


def subject_of(key: str) -> int:
    return parse_key(key)[0]


def session_of(key: str) -> int:
    return parse_key(key)[1]


def trial_of(key: str) -> int:
    return parse_key(key)[2]


def _keys(subjects: Iterable[int], sessions: Iterable[int] = SESSIONS, trials: Iterable[int] = TRIALS) -> tuple[str, ...]:
    return tuple(recording_key(s, d, t) for s in subjects for d in sessions for t in trials)


def _others(s: int, pool: Iterable[int] = SUBJECTS) -> list[int]:
    return [o for o in pool if o != s]


def _cross_day(protocol: str, s: int) -> list[Split]:
    return [Split(protocol, f"s{s:02d}", _keys([s], [DAY1]), _keys([s], [DAY2]), s)]


def _cross_day_pop(protocol: str, s: int) -> list[Split]:
    return [Split(protocol, f"s{s:02d}", _keys([s], [DAY1]) + _keys(_others(s)), _keys([s], [DAY2]), s)]


def calibration_key(s: int) -> str:
    """The session-2 recording whose first seconds are the labelled calibration data."""
    return recording_key(s, DAY2, CAL_TRIAL)


def _day2_after_cal(s: int) -> tuple[str, ...]:
    return _keys([s], [DAY2], [t for t in TRIALS if t != CAL_TRIAL])


def _cross_day_pop_cal(protocol: str, s: int) -> list[Split]:
    (sp,) = _cross_day_pop(protocol, s)
    return [Split(protocol, sp.name, sp.train, _day2_after_cal(s), s)]


def _loso(protocol: str, s: int) -> list[Split]:
    return [Split(protocol, f"s{s:02d}", _keys(_others(s)), _keys([s]), s)]


_BUILDERS: dict[str, Callable[[str, int], list[Split]]] = {
    "cross_day": _cross_day,
    "cross_day_pop": _cross_day_pop,
    "cross_day_pop_cal": _cross_day_pop_cal,
    "loso": _loso,
}
PROTOCOLS = tuple(_BUILDERS)


def make_splits(protocol: str, subjects: Iterable[int]) -> list[Split]:
    """All splits of `protocol` whose test subject is in `subjects`, each checked for leakage."""
    if protocol not in _BUILDERS:
        raise ValueError(f"unknown protocol {protocol!r}, use one of {PROTOCOLS}")
    splits = [sp for s in subjects for sp in _BUILDERS[protocol](protocol, s)]
    for sp in splits:
        check_split(sp)
    return splits


# Leakage rules
def _fail(sp: Split, why: str) -> None:
    raise ValueError(f"{sp.name}: {why}")


def _check_common(sp: Split) -> None:
    if not sp.train or not sp.test:
        _fail(sp, "empty train or test set")
    if set(sp.train) & set(sp.test):
        _fail(sp, "recordings in both train and test")


def _check_one_subject(sp: Split) -> None:
    if {subject_of(k) for k in sp.train + sp.test} != {sp.held_out_subject}:
        _fail(sp, "personalised split mixes subjects")


def _check_test_is_subject(sp: Split) -> None:
    if {subject_of(k) for k in sp.test} != {sp.held_out_subject}:
        _fail(sp, "new-person split tests on someone other than the held-out subject")


def _check_test_is_day2(sp: Split) -> None:
    if {session_of(k) for k in sp.test} != {DAY2} or {subject_of(k) for k in sp.test} != {sp.held_out_subject}:
        _fail(sp, "cross-day must test on the subject's session 2")


def _check_train_day1(sp: Split) -> None:
    if {session_of(k) for k in sp.train} != {DAY1}:
        _fail(sp, "cross-day must train on session 1")


def _check_own_day1_only(sp: Split) -> None:
    if any(subject_of(k) == sp.held_out_subject and session_of(k) != DAY1 for k in sp.train):
        _fail(sp, "population cross-day may use only the subject's session 1")


def _check_calibration_held_out(sp: Split) -> None:
    if calibration_key(sp.held_out_subject) in sp.train + sp.test or CAL_TRIAL in {trial_of(k) for k in sp.test}:
        _fail(sp, f"session-2 trial {CAL_TRIAL} is calibration data, so it may be in neither train nor test")


def _check_subject_absent(sp: Split) -> None:
    if any(subject_of(k) == sp.held_out_subject for k in sp.train):
        _fail(sp, "held-out subject in training")


_RULES: dict[str, tuple[Callable[[Split], None], ...]] = {
    "cross_day": (_check_one_subject, _check_train_day1, _check_test_is_day2),
    "cross_day_pop": (_check_own_day1_only, _check_test_is_day2),
    "cross_day_pop_cal": (_check_own_day1_only, _check_test_is_day2, _check_calibration_held_out),
    "loso": (_check_subject_absent, _check_test_is_subject),
}


def check_split(sp: Split) -> None:
    """Raise ValueError if a split leaks test information into training for its protocol."""
    _check_common(sp)
    for rule in _RULES[sp.protocol]:
        rule(sp)


def check_calibration(sp: Split, cal: Iterable[str]) -> None:
    """Raise unless calibration recordings are the scored person's own, from the test session, in neither train nor test."""
    check_split(sp)
    cal = set(cal)
    if cal & (set(sp.train) | set(sp.test)):
        _fail(sp, "calibration recordings are also training or test recordings")
    sessions = {session_of(k) for k in sp.test}
    if any(subject_of(k) != sp.held_out_subject or session_of(k) not in sessions for k in cal):
        _fail(sp, "calibration recordings must be the scored person's own, from the test session")
