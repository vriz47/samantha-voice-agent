from samantha.duplex.relay import RelayTurn, _first_clause, _trim_context


def _history(turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "You are Samantha."}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"Q{i}"})
        msgs.append({"role": "assistant", "content": f"A{i}"})
    msgs.append({"role": "user", "content": "FINAL?"})
    return msgs


def _recap(out: list[dict]) -> str:
    head = out[0]["content"]
    return head.split("\n\n")[-1]


def test_trim_context_always_system_plus_one_user():
    for turns in (0, 1, 2, 3, 5):
        out = _trim_context(_history(turns))
        assert [m["role"] for m in out] == ["system", "user"]
        assert out[1]["content"] == "FINAL?"
        assert out[0]["content"].startswith("You are Samantha.")


def test_trim_context_keeps_only_latest_turns_in_order():
    body = _recap(_trim_context(_history(3)))
    assert body.count("They asked") == 2
    assert body.index('"Q1"') < body.index('"Q2"')


def test_trim_context_keeps_recap_within_word_budget():
    long_turn = [{"role": "system", "content": "S."}]
    for i in range(2):
        long_turn.append({"role": "user", "content": " ".join(f"u{i}w{n}" for n in range(60))})
        long_turn.append({"role": "assistant", "content": " ".join(f"a{i}w{n}" for n in range(60))})
    long_turn.append({"role": "user", "content": "now"})
    body = _recap(_trim_context(long_turn))
    assert "w24" not in body
    assert "u0w0" in body and "u0w23" in body


def test_first_clause_prefers_comma():
    cut = _first_clause("The sea is deep and cold, and very quiet at night indeed", 3)
    assert cut[0] == "The sea is deep and cold"
    assert cut[1].startswith("and very quiet")


def test_first_clause_breaks_before_conjunction_without_comma():
    cut = _first_clause("the sea is deep and very cold at night and it is still", 3)
    assert cut[0].endswith("deep")
    assert cut[1].lower().startswith("and very cold")


def test_first_clause_never_ends_on_a_dangling_auxiliary():
    for text in ("the sea can be so powerful and mysterious indeed",
                 "the water is deep and cold and quiet at night"):
        head, rest = _first_clause(text, 3)
        assert head.split()[-1].lower() not in {"is", "are", "be", "can", "will", "was"}


def test_first_clause_falls_back_to_word_count():
    cut = _first_clause("one two three four five six seven", 3)
    assert cut == ("one two three", "four five six seven")


def test_split_ready_never_cuts_after_first_fragment():
    text = "The sea can be so powerful and mysterious. It shapes our world!"
    ready, rest = RelayTurn._split_ready(text, head_words=3, allow_short=True)
    assert ready == "The sea can be so powerful"
    assert RelayTurn._split_ready(rest, head_words=3, allow_short=False) == (
        "and mysterious.",
        "It shapes our world!",
    )


def test_split_ready_waits_for_boundary_without_allow_short():
    assert RelayTurn._split_ready("be so powerful and mysterious", head_words=3, allow_short=False) is None