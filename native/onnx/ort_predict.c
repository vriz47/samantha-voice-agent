#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vendor/onnxruntime_c_api.h"

#define N_MELS 80
#define N_FRAMES 800
#define FEAT_LEN (N_MELS * N_FRAMES)

static const OrtApi *g_api = NULL;
static OrtEnv *g_env = NULL;
static OrtSessionOptions *g_opts = NULL;
static OrtSession *g_sess = NULL;
static OrtMemoryInfo *g_mem = NULL;
static int g_ready = 0;

static int chk(OrtStatus *st, const char *what) {
  if (st == NULL) return 0;
  const char *msg = g_api->GetErrorMessage(st);
  fprintf(stderr, "[ort] %s failed: %s\n", what, msg ? msg : "?");
  return -1;
}

int samantha_ort_open(const char *model_path, int n_threads) {
  if (g_ready) return 0;

  g_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!g_api) {
    fprintf(stderr, "[ort] GetApi(%d) returned NULL\n", ORT_API_VERSION);
    return -1;
  }

  if (chk(g_api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "samantha", &g_env), "CreateEnv")) return -1;
  if (chk(g_api->CreateSessionOptions(&g_opts), "CreateSessionOptions")) return -1;

  int thr = n_threads > 0 ? n_threads : 1;
  (void)g_api->SetIntraOpNumThreads(g_opts, thr);
  (void)g_api->SetInterOpNumThreads(g_opts, 1);
  (void)g_api->SetSessionExecutionMode(g_opts, ORT_SEQUENTIAL);
  (void)g_api->SetSessionGraphOptimizationLevel(g_opts, ORT_ENABLE_ALL);

  if (chk(g_api->CreateSession(g_env, model_path, g_opts, &g_sess), "CreateSession")) return -1;
  if (chk(g_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &g_mem), "CreateCpuMemoryInfo"))
    return -1;

  g_ready = 1;
  return 0;
}

static float run_one(const float *feat) {
  int64_t shape[3] = {1, N_MELS, N_FRAMES};
  OrtValue *in = NULL;
  if (chk(g_api->CreateTensorWithDataAsOrtValue(g_mem, (void *)feat, sizeof(float) * FEAT_LEN, shape, 3,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in),
          "CreateTensorWithDataAsOrtValue"))
    return -1.0f;

  const char *in_names[1] = {"input_features"};
  const char *out_names[1] = {"logits"};
  OrtValue *out[1] = {NULL};

  OrtStatus *st = g_api->Run(g_sess, NULL, in_names, (const OrtValue *const *)&in, 1, out_names, 1, out);
  if (chk(st, "Run")) {
    g_api->ReleaseValue(in);
    return -1.0f;
  }

  float *data = NULL;
  if (chk(g_api->GetTensorMutableData(out[0], (void **)&data), "GetTensorMutableData")) {
    g_api->ReleaseValue(in);
    g_api->ReleaseValue(out[0]);
    return -1.0f;
  }

  float prob = data[0];
  g_api->ReleaseValue(in);
  g_api->ReleaseValue(out[0]);
  return prob;
}

float samantha_ort_predict(const float *feat) {
  if (!g_ready) return -1.0f;
  return run_one(feat);
}

int samantha_ort_predict_batch(const float *feats, int n, float *out) {
  if (!g_ready) return -1;
  for (int i = 0; i < n; i++) out[i] = run_one(feats + (size_t)i * FEAT_LEN);
  return 0;
}

int samantha_ort_ready(void) { return g_ready; }

void samantha_ort_close(void) {
  if (g_mem) {
    g_api->ReleaseMemoryInfo(g_mem);
    g_mem = NULL;
  }
  if (g_sess) {
    g_api->ReleaseSession(g_sess);
    g_sess = NULL;
  }
  if (g_opts) {
    g_api->ReleaseSessionOptions(g_opts);
    g_opts = NULL;
  }
  if (g_env) {
    g_api->ReleaseEnv(g_env);
    g_env = NULL;
  }
  g_ready = 0;
}