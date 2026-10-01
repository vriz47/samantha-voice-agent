set -euo pipefail
cd "$(dirname "$0")"

SHERPA_LIB=$(python -c "import os, sherpa_onnx; print(os.path.join(os.path.dirname(sherpa_onnx.__file__), 'lib'))")

mkdir -p build
clang -O2 -fPIC -shared -o build/libsamantha_ort.so ort_predict.c -I. -Wall \
  -L"$SHERPA_LIB" -lonnxruntime -Wl,-rpath,"$SHERPA_LIB"

echo "built build/libsamantha_ort.so ($(stat -c %s build/libsamantha_ort.so) bytes)"