"""
Monkey-patch to fix tatsu parser race condition in textworld.

Root cause: textworld uses module-level singleton tatsu Parser instances
(PddlLogicParser, CSGParser, TextGrammarParser, GameLogicParser).
TatSu's ParseContext.parse() mutates instance state (_tokenizer, _statestack,
_last_node, etc.), so concurrent .parse() calls on the same instance corrupt
each other's state, producing FailedToken errors like:
  "expecting 'action', got template :: 'take {o} from {r}'"

Fix: Replace each singleton with a threading.local()-backed factory that
gives each thread its own parser instance. Zero lock contention, ~4 KB
per thread overhead.

Usage:
    import patches.tatsu_parallel_fix  # apply once at process startup
    # OR
    from patches.tatsu_parallel_fix import apply_patch
    apply_patch()
"""

import threading

_applied = False


def apply_patch():
    global _applied
    if _applied:
        return
    _applied = True

    # --- Patch 1: textworld.envs.pddl.logic ---
    import textworld.envs.pddl.logic as pddl_logic_mod
    from textworld.envs.pddl.logic.parser import PddlLogicParser
    from textworld.envs.pddl.logic.model import PddlLogicModelBuilderSemantics

    _pddl_logic_local = threading.local()

    def _get_pddl_logic_parser():
        p = getattr(_pddl_logic_local, 'parser', None)
        if p is None:
            p = PddlLogicParser(semantics=PddlLogicModelBuilderSemantics(), parseinfo=True)
            _pddl_logic_local.parser = p
        return p

    original_pddl_logic_parse = pddl_logic_mod._parse_and_convert

    def _patched_pddl_logic_parse_and_convert(*args, **kwargs):
        from textworld.envs.pddl.logic import _ModelConverter
        parser = _get_pddl_logic_parser()
        model = parser.parse(*args, **kwargs)
        return _ModelConverter().walk(model)

    pddl_logic_mod._parse_and_convert = _patched_pddl_logic_parse_and_convert

    # Also patch direct _PARSER usage in GameLogic.__init__
    pddl_logic_mod._PARSER = None  # prevent accidental direct use

    # --- Patch 2: textworld.envs.pddl.textgen (CSGParser - HOT PATH) ---
    import textworld.envs.pddl.textgen as pddl_textgen_mod
    from textworld.envs.pddl.textgen.parser import CSGParser
    from textworld.envs.pddl.textgen.model import CSGModelBuilderSemantics

    _csg_local = threading.local()

    def _get_csg_parser():
        p = getattr(_csg_local, 'parser', None)
        if p is None:
            p = CSGParser(semantics=CSGModelBuilderSemantics(), parseinfo=True)
            _csg_local.parser = p
        return p

    def _patched_csg_parse_and_convert(*args, **kwargs):
        from textworld.envs.pddl.textgen import _Converter
        parser = _get_csg_parser()
        model = parser.parse(*args, **kwargs)
        return _Converter().walk(model)

    pddl_textgen_mod._parse_and_convert = _patched_csg_parse_and_convert
    pddl_textgen_mod._PARSER = None

    # --- Patch 3: textworld.logic (GameLogicParser) ---
    import textworld.logic as tw_logic_mod
    from textworld.logic.parser import GameLogicParser
    from textworld.logic.model import GameLogicModelBuilderSemantics

    _game_logic_local = threading.local()

    def _get_game_logic_parser():
        p = getattr(_game_logic_local, 'parser', None)
        if p is None:
            p = GameLogicParser(semantics=GameLogicModelBuilderSemantics(), parseinfo=True)
            _game_logic_local.parser = p
        return p

    def _patched_game_logic_parse_and_convert(*args, **kwargs):
        parser = _get_game_logic_parser()
        model = parser.parse(*args, **kwargs)
        return tw_logic_mod._ModelConverter().walk(model)

    tw_logic_mod._parse_and_convert = _patched_game_logic_parse_and_convert
    tw_logic_mod._PARSER = None

    # --- Patch 4: textworld.textgen (TextGrammarParser) ---
    import textworld.textgen as tw_textgen_mod
    from textworld.textgen.parser import TextGrammarParser
    from textworld.textgen.model import TextGrammarModelBuilderSemantics

    _text_grammar_local = threading.local()

    def _get_text_grammar_parser():
        p = getattr(_text_grammar_local, 'parser', None)
        if p is None:
            p = TextGrammarParser(semantics=TextGrammarModelBuilderSemantics(), parseinfo=True)
            _text_grammar_local.parser = p
        return p

    @classmethod
    def _patched_textgrammar_parse(cls, grammar, filename=None):
        parser = _get_text_grammar_parser()
        model = parser.parse(grammar, filename=filename)
        return cls._CONVERTER.walk(model)

    tw_textgen_mod.TextGrammar.parse = _patched_textgrammar_parse
    tw_textgen_mod.TextGrammar._PARSER = None

    print("[tatsu_parallel_fix] Patched 4 singleton parsers with thread-local instances")


# Auto-apply on import
apply_patch()
