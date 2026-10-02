# Update 2026-10-02 — Naturalness pass

Dokumen ini adalah catatan perubahan versi ini. `docs/LATENCY.md` dan
`docs/DEVLOG.md` sengaja **tidak** diubah supaya riwayat pengukuran sebelumnya
tetap utuh; semua angka dan keputusan baru ada di sini.

## Ringkasan

Relay race semula dioptimumkan murni demi latensi, dan hasilnya suara Samantha
terdengar terpotong. Update ini memperbaikinya dengan bantuan saran luar
(Gemini), lalu mengorbankan sebagian latensi. Namun target `<700 ms`
dibatalkan secara sadar.

## Apa yang berubah

### 1. Pemotongan klausa lebih cerdas

`_first_clause()` di `samantha/duplex/relay.py` memilih titik potong dengan
urutan preferensi:

1. koma pertama yang tersedia,
2. batas lemah sebelum konjungsi (`and`, `but`, `or`, `so`, `because`, …),
3. potong pada kata ke-N sebagai fallback.

`_AUX` menutup jalur yang meninggalkan kata kerja bantu menggantung di akhir
fragmen, misalnya `"The sea can be"` digeser agar tidak terpotong di situ.

`_split_ready(allow_short=...)` sekarang hanya mengizinkan pemotongan cepat
pada fragmen **pertama**. Sebelumnya pemeriksaan tanda baca dilakukan pada kata
pertama sisa teks, yang hampir tidak pernah cocok, sehingga setiap fragmen
dipotong pada jumlah kata yang sama dan seluruh jawaban tersendat.

### 2. Menyambung audio antar fragmen

`samantha/duplex/player.py` sekarang memperlakukan setiap fragmen sebagai potongan
dari satu tarikan napas:

- crossfade 30 ms antar fragmen,
- jeda 0.12 s setelah koma, 0.22 s setelah titik,
- normalisasi peak ke 0.70 supaya volume tidak melonjak-naik,
- tail yang ditahan di-flush saat `close()` supaya tidak ada sampel terpotong.

### 3. Prompt dan laju bicara

System prompt meminta kalimat pendek dengan koma sering, memberi batas alami
bagi splitter. Sampling dikencangkan ke `temperature=0.6`, `top_p=0.9`. Laju
bicara `speed` diturunkan dari `1.0` ke `0.80` setelah render A/B dari reply
LLM identik.

## Bug yang ditemukan sekalian

- **`_trim_context()` salah pair history.** Slicing lama hanya benar saat
  history persis dua turn; dengan tiga turn atau lebih, iterasi dimulai dari
  pesan `assistant` sehingga pasangan bergeser dan baris recap salah kutipan.
- **Terminator kosong ikut mendapat jeda.** `"" in ",;:"` bernilai `True`
  di Python.
- **Self-healing model 404** di `BrainClient`: bila model yang diminta tidak
  tersedia, client mengambil model aktif dari `/v1/models`.

## Angka

Semua diukur di **POCO F6, Snapdragon 8s Gen 3, Android 16, Termux**, dengan
reply dan fragmen identik agar perbandingan adil.

| Konfigurasi | First audio (median) |
|---|---|
| `head_words=5`, potong butir kata, tanpa crossfade | 1.258 s |
| `head_words=3`, potong butir kata | 0.868 s |
| `head_words=2`, potong butir kata (terdengar terpotong) | 0.683 s |
| `head_words=3`, koma-dulu + crossfade + jeda, `speed=0.80` | **1.059 s** |

Benchmark relay 15 giliran pada konfigurasi terakhir: 0 error, wall median
6.23 s. `pytest -q`: 26 passed.

## Tradeoff yang diterima

Setiap perbaikan yang membuat suara lebih manusia menaikkan latensi: menunggu
koma butuh lebih banyak kata, dan `speed` yang lebih lambat memperlambat
sintesis fragmen pertama. Versi `head_words=2` memang mencapai 0.683 s, tapi
didengar terpotong dan terlalu cepat, sehingga tidak dipakai.

Nomor 0.683 s sengaja tidak dikejar lagi. Fokus berikutnya adalah menjaga
naturalitas, bukan memangkas milidetik.

## Yang belum selesai

- First audio terukur masih memakai proxy waktu antrean, bukan timestamp PCM
  pertama yang benar-benar diputar di speaker.
- `_trim_context()` belum diuji langsung terhadap Gallery untuk history
  panjang lebih dari dua turn.
- Pipeline audio di belakang TTS (VAD, ASR streaming, barge-in) belum
  diintegrasikan ke relay race ini.