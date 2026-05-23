# Tatsu Parser Parallel Race Condition Fix

## Root Cause

TextWorld 1.7.0 uses 4 module/class-level **singleton** TatSu parser instances:

| # | Location | Parser Class | Used For |
|---|----------|-------------|----------|
| 1 | `textworld.logic.__init__._PARSER` | GameLogicParser | Game logic parsing |
| 2 | `textworld.textgen.__init__.TextGrammar._PARSER` | TextGrammarParser | Text grammar |
| 3 | `textworld.envs.pddl.logic.__init__._PARSER` | PddlLogicParser | PDDL domain loading |
| 4 | `textworld.envs.pddl.textgen.__init__._PARSER` | CSGParser | **Context-sensitive grammar derive()** |

TatSu `ParseContext.parse()` mutates instance state on every call:
- `self._tokenizer` (input buffer)
- `self._statestack` (parser state stack)
- `self._rule_stack`, `self._cut_stack`
- `self._last_node`, `self._furthest_exception`

When multiple threads call `.parse()` on the same singleton, they corrupt each other's state.

**Critical path**: Parser #4 (CSGParser) is called on **every env.step()** through:
```
pddl.step() → grammar.derive(feedback_rule) → _parse_and_convert() → _PARSER.parse()
pddl._gather_infos() → grammar.derive(command_template) → _parse_and_convert() → _PARSER.parse()
```

This means even with the existing `_TW_LOAD_LOCK` (which serializes env creation), parallel `step()` calls still race.

## Error Manifestation

```
tatsu.exceptions.FailedToken: expecting 'action', got template :: "take {o} from {r}"
IndexError: pop from empty list  (corrupted _statestack)
TypeError: 'NoneType' object is not iterable  (tokenizer replaced by None)
```

## Fix: Thread-Local Parser Instances

`patches/tatsu_parallel_fix.py` replaces each singleton with `threading.local()`-backed per-thread instances.

- Zero lock contention (true parallelism)
- ~4 KB memory per parser × N threads (negligible)
- Applied via monkey-patch at import time

## Verification

| Condition | Result |
|-----------|--------|
| Without patch, 4 workers parallel step | 7/8 crashed |
| With patch, 4 workers parallel step | 8/8 passed |

## Usage

Add to the entry point (before textworld import):
```python
import sys
sys.path.insert(0, './RAGEN')
import patches.tatsu_parallel_fix
```

Or in `ragen/env/alfworld/env.py`:
```python
import patches.tatsu_parallel_fix  # noqa: F401 (at top of file)
```

Then set `parallel_friendly: True` and `max_workers: 4` in the ALFWorld env config.

## Expected Speedup

With `max_workers=4` for val (32 games):
- Serial: ~10 min (32 games × ~20s each, sequential)
- Parallel: ~2.5 min (8 batches × ~20s)
- **~4x speedup on val evaluation**

Note: env creation (register_game + make + fast_downward planning) still needs
serialization via `_TW_LOAD_LOCK`. The parallelism benefit comes from concurrent
step() calls across already-initialized envs.
