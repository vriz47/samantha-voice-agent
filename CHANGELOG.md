# Changelog

Semua perubahan yang berarti untuk Samantha voice agent. Format mengikuti
[Keep a Changelog](https://keepachangelog.com/), versi mengikuti semver.

## [Unreleased] — 2026-10-02

### Changed

- **Kecepatan bicara `speed` 1.0 → 0.80.** Render A/B dengan tiga varian
  (`samantha-A-100.wav`, `B-088`, `C-080`) dibuat dari reply LLM identik;
  versi 0.80 dipilih karena 1.0 terdengar terlalu cepat.
- **`head_words` 5 → 3** pada `RelayTurn`, membuka frasa pertama lebih
  pendek tanpa harus memecah setiap fragmen.
- **System prompt** sekarang meminta kalimat pendek dengan koma sering,
  dan sampling dikencangkan ke `temperature=0.6`, `top_p=0.9`.
- **Target first audio <700 ms dibatalkan.** Alasan lengkap ada di
  `docs/UPDATE-2026-10-02-naturalness.md`.

### Added

- `_first_clause()` memilih titik potong klausa dengan urutan preferensi:
  koma pertama, lalu batas lemah sebelum konjungsi (`and`, `but`, `so`,
  …), baru potong kata ke-N sebagai fallback.
- `_AUX` mencegah pemotongan yang meninggalkan kata kerja bantu
  menggantung di akhir fragmen.
- `StreamPlayer` menyambung antar fragmen: crossfade 30 ms, jeda 0.12 s
  setelah koma dan 0.22 s setelah titik, normalisasi peak 0.70, dan
  flush tail saat `close()` agar tidak ada sampel terpotong.
- `brain/tests/test_relay_split.py` — 8 test untuk splitting klausa dan
  `_trim_context()`.
- `~/bin/gallery-watchdog` — reviving server AI Edge Gallery saat mati.

### Fixed

- **`_trim_context()` salah pair history.** Slicing lama hanya benar saat
  history persis dua turn; dengan tiga turn atau lebih, iterasi mulai dari
  pesan `assistant` sehingga pasangan user/assistant bergeser dan baris
  recap salah kutipan. Sekarang berjalan dari belakang mengambil pasangan
  `user`/`assistant` yang valid, maksimal dua turn terakhir.
- **Pemotongan cepat hanya untuk fragmen pertama.** Sebelumnya
  `_split_ready()` mengecek tanda baca kalimat pada kata pertama sisa
  teks, yang hampir tidak pernah cocok, sehingga *setiap* fragmen dipotong
  pada jumlah kata yang sama dan seluruh jawaban tersendat.
- **Terminator kosong ikut mendapat jeda.** `"" in ",;:"` bernilai
  `True` di Python, jadi fragmen tanpa tanda baca tak sengaja diberi jeda
  koma.
- **Self-healing model 404** di `BrainClient`: bila model yang diminta
  tidak tersedia, client mengambil model aktif dari `/v1/models`
  alih-alih gagal.
- `tools/bench_relay.py` sekarang mendeteksi model aktif dan tahan
  terhadap giliran tanpa audio.

### Measured

- `pytest -q`: 26 passed.
- First audio median: 0.868 s → **1.059 s** (regresi yang disengaja demi
  naturalness).
- Relay benchmark 15 giliran: 0 error, wall median 6.23 s.

---

Catatan versi: entri ini belum dipublikasikan sebagai release GitHub.

## [Unreleased] — 2026-10-03

### Changed

- `StreamPlayer` mencatat onset sink dan mengombinasikannya dengan `latency_msec`
  untuk memperkirakan onset speaker, jadi laporan benchmark tidak lagi berhenti di
  antrean.
- `tools/bench_relay.py` melaporkan `speaker onset` terpisah dari `first audio`.

### Added

- `docs/UPDATE-2026-10-03-playback-measurement.md`.
- `~/bin/gallery-watchdog` sekarang drove UI Gallery (tab Server lalu toggle) alih-alih
  hanya `am start`, dan memverifikasi model benar-benar terlayani.

### Fixed

- **`first_drain_at` dicatat setelah `write()` selesai.** Untuk chunk besar, `write()`
  tertahan menunggu paplay mengonsumsi audio secara realtime, jadi onset yang terukur
  meleset sekitar 1829 ms dan terlihat seperti backlog padahal tidak ada. Dicatat
  sebelum `write()`, selisih antre-ke-sink jadi konsisten ~1 ms.
- Watchdog menganggap server sehat bila port listening, padahal bisa mengembalikan
  `{"data":[]}` — port hidup tanpa model termuat.

### Measured

- Relay benchmark 15 giliran: first audio median 1.036 s, speaker onset median
  **1.157 s**, wall median 6.33 s, 0 error.

### Known limitation

- Onset speaker masih estimasi dari `latency_msec`, bukan pengukuran akustik.
- `Gemma3-1B-IT` dipasang langsung ke filesystem sehingga tidak ada di registry model
  Gallery; kalau proses Gallery mati, model harus dimuat ulang manual lewat UI.