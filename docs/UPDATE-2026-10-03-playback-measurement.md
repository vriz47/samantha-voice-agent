# Update 2026-10-03 — Mengukur playback secara jujur

Dokumen ini melanjutkan `docs/UPDATE-2026-10-02-naturalness.md`, yang tidak diubah.
Fokusnya: angka first audio yang kita laporkan sebelumnya ternyata mengukur hal yang
berbeda dari yang dikira.

## Temuan utama

`StreamPlayer` mencatat `first_audio_at` ketika PCM **diantrekan**, bukan ketika
suara keluar dari speaker. Selama ini angka itu dilaporkan sebagai "first audio",
padahal masih ada dua tahap tambahan:

```
antre di queue  ->  paplay (stdin)  ->  speaker
                 +1 ms              +120 ms (latency_msec)
```

Pengukuran menunjukkan antre-ke-paplay hanya **~1 ms**, jadi tidak ada backlog.
Selisih sink ke speaker adalah `latency_msec` yang sengaja diset 120 ms.

Artinya semua angka first audio sebelumnya **kurang optimistis sekitar 120 ms**.

## Bug measurement yang ditemukan

Versi pertama dari instrumentation ini mencatat `first_drain_at` **setelah**
`proc.stdin.write()` selesai. Untuk chunk besar, `write()` tertahanwhile paplay
mencerna audio pada kecepatan realtime, sehingga angka pertama keluar sebagai
**1829 ms** — tampak seperti backlog besar, padahal tidak ada.

Setelah dipindahkan ke sebelum `write()`, selisih antre-ke-sink konsisten ~1 ms.
Ini penting dicatat: angka 1829 ms itu artefak pengukuran, bukan performa sistem.

## Angka final

Benchmark relay, 4 pertanyaan × 3 ronde, Gallery + `Gemma3-1B-IT`:

| Metrik | Antre | Speaker |
|---|---|---|
| min | 0.646 s | 0.767 s |
| median | 1.036 s | 1.157 s |
| mean | 1.033 s | — |
| max | 1.370 s | 1.495 s |

Wall median 6.33 s, 0 error dari 15 giliran, `<1 s` 5/15.

## Watchdog Gallery akhirnya berfungsi

Temuan penting soal pemulihan server: **`am start` saja tidak cukup**.

Urutannya yang benar, terverifikasi lewat `uiautomator dump`:

1. `am start -n com.server.edge.gallery/.MainActivity` — membuka tab **Chats**
2. Ketuk tab **Server** di `(765, 2592)`
3. Ketuk toggle di `(1100, 600)`
4. Tunggu ±12 s

`~/bin/gallery-watchdog` sekarang menjalankan urutan itu dan, yang lebih penting,
**memverifikasi model benar-benar terlayani** lewat `/v1/models` — bukan sekadar
port yang listening. Dulu itu keliru: server pernah hidup dengan `{"data":[]}`,
yaitu port aktif tapi tidak ada model termuat, sehingga tetap tidak berguna.

### Model harus sudah termuat di RAM

Aplikasi*{\jiwax}*''*~ forget model saat prosesnya mati, dan `Gemma3-1B-IT` yang kita
pasang langsung ke filesystem **tidak terdaftar** di registry aplikasi. Daftar model
yang dikenali aplikasi hanya:

| Model | Status |
|---|---|
| Gemma-4-E2B-it | Downloaded, 2.6 GB |
| Gemma-4-E4B-it | belum diunduh |
| Gemma-3n-E2B-it | belum diunduh |
| gemma-4-E2B-it-gpu.litertlm | Imported |
| DeepSeek-R1-Distill-Qwen-1.5B | Downloaded |
| MobileActions-270M | belum diunduh |

Konsekuensinya: kalau proses Gallery mati, model harus **dimuat ulang lewat UI**
secara manual. Watchdog tidak bisa melakukannya sendiri dan akan mencatat
`no model served: load a model on the Models tab by hand`.

## Yang masih terbuka

- Onset speaker masih **estimasi** (`latency_msec`), bukan hasil pengukuran
  akustik nyata. Mengukurnya butuh loopback capture, belum dikerjakan.
- Model belum bisa dimuat ulang tanpa campur tangan manusia.
- Pipeline VAD, ASR streaming, dan barge-in belum masuk ke relay race.