from brain.chunker import ClauseStreamer
from brain.persona import PersonaStreamer, clamp_sentences, sanitize


def test_strips_identity_leak():
    assert sanitize("Sebagai model bahasa, saya tidak punya perasaan.") == "Saya tidak punya perasaan."
    assert "model" not in sanitize("Aku adalah model bahasa yang membantu.")


def test_strips_speaker_mark():
    assert sanitize("Samantha: Kabar baik!") == "Kabar baik!"


def test_strips_apology_prefix():
    out = sanitize("Maaf ya, saya hanya bisa membantu dengan teks. Hutch.")
    assert out.startswith("Hutch")


def test_strips_list_lead():
    assert sanitize("Berikut: satu, dua.") == "Satu, dua."


def test_keeps_clean_text_intact():
    text = "Kabar baik, terima kasih sudah bertanya."
    assert sanitize(text) == text


def test_clamp_sentences():
    assert clamp_sentences("Satu. Dua. Tiga.") == "Satu. Dua."


def test_persona_streamer_filters_every_clause():
    ps = PersonaStreamer(ClauseStreamer(min_len=10, max_len=80))
    out = []
    for d in ["Sebagai model bahasa, saya akan ", "membantu denganPertanyaan kamu. ", "Ada lagi?"]:
        out += ps.push(d)
    out += ps.flush()
    joined = " ".join(out)
    assert "model bahasa" not in joined
    assert joined.strip()