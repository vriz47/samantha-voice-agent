from brain.chunker import ClauseStreamer, clean, clauses, split_prefix


def test_clean_strips_markdown_and_emoji():
    raw = (
        "Halo **Samantha**, ini `code` dan emoji \U0001f60a "
        "link [sini](https://x.io).\n- poin satu\n- poin dua"
    )
    out = clean(raw)
    assert "**" not in out
    assert "\U0001f60a" not in out
    assert "poin satu" in out
    assert "sini" in out and "https://x.io" not in out


def test_clean_strips_closed_fence_only():
    out = clean("sebelum ```python\nprint(1)\n``` setelah")
    assert "print" not in out
    assert "sebelum" in out and "setelah" in out


def test_clean_swallows_unterminated_fence():
    out = clean("sebelum ```python\nprint(1)\nsetelah")
    assert "print" not in out
    assert "sebelum" in out


def test_clean_removes_list_markers():
    assert not clean("- alpha\n- beta").startswith("-")


def test_split_prefix_waits_for_min_len():
    head, rest = split_prefix("Halo Samantha, apa kabar?", 18, 140)
    assert head
    assert rest == ""


def test_split_prefix_cuts_at_max_len():
    text = "a" * 200
    head, rest = split_prefix(text, 18, 140)
    assert 18 <= len(head) <= 140
    assert rest


def test_clauses_merges_short_tail():
    parts = clauses("Kabar baik, terima kasih sudah bertanya. Lalu apa?", 18, 140)
    assert parts
    assert parts[0].endswith(" Lalu apa?")


def test_streamer_emits_before_stream_ends():
    s = ClauseStreamer(min_len=12, max_len=90)
    emitted = []
    for delta in ["Kabar saya baik, terima kasih", " sudah bertanya. Semoga", " harimuunque"]:
        emitted += s.push(delta)
    assert emitted, "clause harus keluar sebelum stream selesai"
    assert emitted[0].startswith("Kabar saya baik")
    assert "".join(emitted + s.flush())


def test_streamer_never_loses_text():
    text = (
        "Halo, saya Samantha. Aku tinggal di HP kamu, jadi semua jalan lokal. "
        "Mau ngobrol apa hari ini?"
    )
    s = ClauseStreamer(min_len=14, max_len=60)
    got = []
    for ch in text:
        got += s.push(ch)
    got += s.flush()
    assert "".join(got).replace(" ", "") == text.replace(" ", "")


def test_flush_empty_when_only_noise():
    s = ClauseStreamer()
    s.push("\U0001f600 ```x```")
    assert s.flush() == []