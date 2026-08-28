# serve-lifecycle-races

## 1. The mechanism of the "hang" (found, reproduced, proven)

**The stream suites were never hanging on anything the lifecycle fix did. They were hanging on
`~/.cache/shogym/sessions`, which has grown to 79,053 files / 311 MB, and which every single
episode construction reads end to end.**

`ServedEpisode.__init__` calls `FinalizationStore.recover()` (episode.py:392). `recover()` calls
`load_all()` (lifecycle.py:397 -> :376), which does
`sorted(self._dir.glob("finalization-*.json"))` and then `read_text()` + `json.loads()` on every
match. When no trace path is given, `FinalizationStore.resolve_dir()` returns
`_sessions_cache_root()` = `~/.cache/shogym/sessions`: one machine-global directory, shared by
every session, deliberately never keyed per run, and never pruned.

So the cost of opening one episode is O(every finalization record ever written on this machine),
and every suite run adds ~88 more of them. Measured here:

| where | count | `load_all()` |
|---|---|---|
| `~/.cache/shogym/sessions` (Andrew's machine, records dating from Aug 7) | 79,053 | **2.62 s**, warm |

An episode costs 2.6 s before it does anything. `tests/test_serve_stream_feedback.py` opens ~80
episodes, so the module costs ~3.5 minutes of pure directory scanning, and grows every time it is
run. That is the "89 seconds to not finishing".

### The faulthandler stack (the exact wait)

Three dumps, three different tests, the same frame every time:

```
Thread 0x00000001f096a180 (most recent call first):
  File ".../pathlib.py", line 1013 in open
  File ".../pathlib.py", line 1027 in read_text
  File ".../src/shogym/serve/lifecycle.py", line 379 in load_all
  File ".../src/shogym/serve/lifecycle.py", line 397 in recover
  File ".../src/shogym/serve/episode.py", line 392 in __init__
  File ".../src/shogym/serve/episode.py", line 485 in open_env
  File ".../src/shogym/serve/stream.py", line 3735 in get_task
  File ".../tests/test_serve_stream_feedback.py", line 382 in test_immediate_never_crosses_envs
```

The only other thread is an idle pool worker parked in `concurrent.futures.thread._worker`
(`work_queue.get`). Nothing is deadlocked, nothing is waiting on a future, nothing is waiting on
a loop. The main thread is reading JSON files, one at a time, in a loop.

### The control that proves it

The same reverted single-owner patch, unchanged, with `HOME` pointed at an empty directory:

| tree | store | `feedback` + `channels` |
|---|---|---|
| head `1643675` | 79k records | channels alone **47.3 s**; feedback does not finish |
| head + reverted single-owner patch | 79k records | channels alone **38.8 s**; feedback does not finish |
| head `1643675` | empty | **72 passed in 5.89 s** |
| head + reverted single-owner patch | empty | **72 passed in 7.27 s** |

The patch is not slower than head. Under the real store *head is slower than the patch*. The
bisection to `serve/episode.py` was a coincidence of timing: every run of the suite adds records,
so each attempt was measured against a bigger store than the one before it, and the last thing
touched took the blame.

### Why it looked like the fix

`_SESSION_THREADS` is a module-global `ThreadPoolExecutor`; the fork test in
`test_serve_stream_feedback.py` forks a process that has its threads. That is a real fork hazard
(and it is why the lazy + `register_at_fork` commit exists), but it is not what stopped the suite:
with an empty store the pooled patch passes the fork test in 7 seconds.

## 2. What is being changed

- `lifecycle.py`: recovery is a **startup** concern, run once per store directory per process,
  not once per episode; and a terminal record whose owner process is gone is retired, so the
  directory is bounded by live work rather than by all history.
- `core.py`: `Env.end_session()` claims the session id **before** entering `_end_session()`, so a
  release is single-entry and idempotent for every env. This is what makes a second caller (a
  completion callback, `env.close()` after a timed-out teardown) a no-op instead of a second
  concurrent hook.
- `episode.py` (A): setup rollback is owned by a per-session single-thread executor whose
  completion callback runs *in the worker thread*, so it finishes independently of the caller task
  and of the event loop. The outer failure handler waits on that one owner and does not call
  `env.close()` beside it.
- `episode.py` (B): one owned release future per session, created once, observed by `_teardown`
  and by `close()`. `close()` never re-enters the hook.
- `episode.py` / `stream.py` (C): env construction is offloaded (`ServedEpisode.start`'s `make`,
  `TaskStream.get_task`'s `self._env_for`).

## 3. Result

PR: https://github.com/shojin-lab/shogym/pull/141 (base `appworld-env`, head `serve-lifecycle-races`).

Timings on this machine, real 79,053-record store:

| suite | before | after |
|---|---|---|
| `test_serve_stream_channels.py` | 47.3 s | 14.1 s |
| `test_serve_stream_feedback.py` | did not finish | 18.4 s |
| both in one process | did not finish | 18.7 s |
| `tests/` minus `tests/envs`, one process | n/a | 588 passed, 1 skipped, 42.8 s |
| `test_appworld_{ledger,payload,runtime,table}.py` | n/a | 76 passed, 1 skipped, 26.1 s |

`tests/test_serve_session_lifecycle.py`: 11 tests, 9 of which fail on `origin/appworld-env`.

Left undone and written into the PR body: the store is still unbounded (about 88 records added
per suite run, nothing ever retired), so the one-time scan will keep growing. Retiring a terminal
record whose owner process is gone would bound it, but that deletes 311 MB on the maintainer's
machine and is not one of the three findings.

## 4. Status

- [x] mechanism reproduced and proven
- [x] fix A/B/C
- [x] regressions
- [x] full stream suites in one process
- [x] PR

## 5. Cold review round 2 (five findings, all upheld)

All five reproduced on `dd17d055` before any change:

| # | reproduction on `dd17d055` | after |
|---|---|---|
| 1 factory affinity | dispense raised `RuntimeError: no running event loop` | dispense OK; offload behind `off_loop_factory` |
| 2 env never closed | `sessions_open=0 closed=False` | `closed=True` |
| 3 `_close` beside release | close 1 ms, `close_during_release=True` | close 1 ms, `close_during_release=False` |
| 4 release on the loop | `release_thread=MainThread`, 305 ms, 1 tick | `shogym-session_0`, 51 ticks |
| 5a transient write | record stays `PENDING` forever | second pass resolves it |
| 5b fork | child `resolved=0 status=PENDING` | child `resolved=1 status=FAILED` |

Head `8c05fff5c18dcbd5ce8397586953abf782cdd766`. Suites against the real store: 595 passed /
1 skipped / 34.2 s (tests minus tests/envs, one process); channels 3.7 s; feedback 18.7 s;
appworld pure 76 passed / 21.1 s. Lifecycle module 18 tests, 13 failing on `dd17d055`.

## 6. Cold review pass 2 (five findings, all upheld)

Findings 1 to 3 were one bug: the env close was decided by a coroutine after a wait.
Reproduced at `8c05fff5` before any change.

| # | at `8c05fff5` | after |
|---|---|---|
| 1 loop-loss leaks the env | `release_entries=1 env_closed=False` | `env_closed=True` |
| 2 foreign-loop close | `foreign_close=True closed_on_owner=None` | `foreign_close=False closed_on_owner=True` |
| 3 deferred close not single-owner | `close_entries=2` | `close_entries=1 peak_close=1` |
| 4 contextvars dropped | `begin_ctx='unset' end_ctx='unset'` | both `'tenant-a'` |
| 5 `[]` / `{}` / list verdict | `AttributeError` / `TypeError` / `AttributeError` | all skipped, scan left incomplete |

Head `ecc7f19b12fe4aa1d607957a9b376c791a2df0e9`. Lifecycle module 25 tests, 8 failing on
`8c05fff5`. Suites: 602 passed / 1 skipped / 49.8 s (tests minus tests/envs, one process);
terminal 60 passed; appworld pure 76 passed.

One deliberate divergence from the brief: when the owning loop is genuinely gone the close still
runs on a temporary loop and *warns* on failure, rather than not running at all. Refusing to try
would satisfy "never asyncio.run" and leak the env in exactly the shape finding 1 is about.

## 7. Cold review pass 3 (seven findings plus test hygiene, all upheld)

| # | at `ecc7f19b` | after |
|---|---|---|
| 1 slot freed early | task 2 dispensed, env 1 `closed=False` | `closed=True` before dispense |
| 2 close swallows | `close()` returned over a raising `_close` | raises `RuntimeError: close boom` |
| 3 accepted-but-unrun close | `taken=True done=False`, `_run` never awaited | closed by the thread that posted it |
| 4a rollback cancel | `env_closed=False` | `env_closed=True` |
| 4b ordinary cancel | env's `CancelledError` escaped the terminal call | call returns terminated; env closed |
| 5 field validation | string pid raised; boolean pid = live; `[]` cost 3 scans | quarantined; 1 scan |
| 6 hook thread leak | `shogym-session_0` alive | none alive |
| 7 documented example | no `off_loop_factory`, built on the loop | opts in, built off the loop |

Head `e8e0867dd5b104178cc55701f0ca9ae4799024bd` (rebased onto `95e7e85`, one signature conflict
resolved by keeping both `adopt_unidentified` and `off_loop_factory`). Lifecycle module 36 tests,
15 failing on `ecc7f19b`. Suites: 617 passed / 1 skipped / 57.1 s (tests minus tests/envs, one
process); lifecycle + terminal 96 passed; appworld pure 85 passed.

`tests/conftest.py` now redirects the no-trace finalization store per test, so the suite stops
growing `~/.cache/shogym/sessions`.

## 8. Cold review pass 4: the design changed

Finding 3 was not fixable as designed. No elapsed time proves an open loop is dead, so any
takeover-by-timeout either reclaims a live owner or abandons a dead one. The takeover machinery
(tickets, generations, quiet periods) is gone.

**Every episode now owns one thread running one event loop.** The env is built on it (when
`off_loop_factory` is set), the session hooks run on it, the env is closed on it. Callers await
futures with bounds they choose. Loop affinity holds by construction; nothing is ever posted to a
loop this layer does not own. Exit hook is `threading._register_atexit`, which runs before the
non-daemon join (a plain `atexit` hook runs after it and would hang the interpreter).

Feasibility checked before adopting: no env calls `asyncio.run` in a constructor or a session
hook, and `_sync_run_async` already handles a running loop.

All nine findings reproduced at `e8e0867` and fixed. Head
`46b391933034588722f76afa253f478ebc47d5f6` (rebased onto `9390375`). Lifecycle module 49 tests,
16 failing on `e8e0867`. Suites: 630 passed / 1 skipped / 53.9 s (tests minus tests/envs, one
process); lifecycle + terminal 109 passed; appworld pure 93 passed.
