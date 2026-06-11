#ifndef READ_JSON_H
#define READ_JSON_H

/*
 * CLoRA per-step JSON schema (emitted by the Python strategy plugin,
 * consumed by main.c).
 *
 * One JSON file == one decoder pass (all n_layers, all n_matrices fused).
 * The C simulator translates each adapter entry into READ / READ_COMPUTE
 * requests on the CXL link + an attention RC distributed across n_cxl, plus
 * a single NPU_COMPUTE for the base-model GPU time.
 *
 *   {
 *     "kind": "decode" | "prefill",
 *     "model": { "d", "n_layers", "n_matrices", "s_dtype" },
 *     "hw":    { "n_cxl" },
 *     "input_tokens_per_request": int,
 *     "base_model_compute_ns":    float,
 *     "kv": { "kv_tokens_total", "kv_in_gpu_fraction" },
 *     "adapters": [
 *       { "id", "rank", "batch", "strategy" (1..4), "kv_tokens" }, ...
 *     ]
 *   }
 *
 * Sizes computed in the C side are in BYTES. Times in ns. cxl_bandwidth and
 * dram_bandwidth read from the .conf file are in B/ns (== GB/s numerically).
 */

typedef struct {
    unsigned int id;
    unsigned int rank;
    unsigned int batch;
    unsigned int strategy;   /* 1=E1, 2=E2, 3=E3, 4=E4 */
    unsigned int kv_tokens;
} AdapterDecision;

typedef struct {
    int          kind;                       /* 0 = decode, 1 = prefill */
    unsigned int model_d;
    unsigned int n_layers;
    unsigned int n_matrices;
    unsigned int s_dtype;                    /* bytes per FP element */
    unsigned int n_cxl;
    unsigned int input_tokens_per_request;
    double       base_model_compute_ns;
    unsigned int kv_tokens_total;
    double       kv_in_gpu_fraction;

    unsigned int      n_adapters;
    AdapterDecision  *adapters;              /* heap-allocated, length n_adapters */
} JsonData;

JsonData *readJsonData(const char *filename);
void      freeJsonData(JsonData *d);

#endif /* READ_JSON_H */
