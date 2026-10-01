#!/data/data/com.termux/files/usr/bin/bash
# Download + extract one sherpa-onnx model archive.
#   ./fetch_model.sh tts <asset-name>   e.g. tts sherpa-onnx-pocket-tts-int8-2026-01-26
#   ./fetch_model.sh asr <asset-name>   e.g. asr sherpa-onnx-whisper-base.en
#   ./fetch_model.sh asr silero_vad.onnx
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
MODELS="$ROOT/models"
LOG="${TMPDIR:-/data/data/com.termux/files/usr/tmp}/samantha-fetch.log"
CATEGORY="${1:?usage: fetch_model.sh <tts|asr> <asset-name>}"
NAME="${2:?usage: fetch_model.sh <tts|asr> <asset-name>}"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/${CATEGORY}-models/${NAME}.tar.bz2"

mkdir -p "$MODELS" || exit 1
echo "[$(date +%H:%M:%S)] download $NAME"
curl -sL --progress-bar -o "$MODELS/${NAME}.tar.bz2" "$URL" || exit 1
echo "[$(date +%H:%M:%S)] $(stat -c%s "$MODELS/${NAME}.tar.bz2") bytes, extracting..."

tar xjf "$MODELS/${NAME}.tar.bz2" -C "$MODELS" >>"$LOG" 2>&1
rc=$?
rm -f "$MODELS/${NAME}.tar.bz2"
if [ $rc -eq 0 ]; then
  echo "[$(date +%H:%M:%S)] done: $NAME (log: $LOG)"
else
  echo "[$(date +%H:%M:%S)] extract failed rc=$rc, manual check: $LOG"
fi
exit $rc