#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/readJson.h"
#include "../lib/cJSON/cJSON.h"

/* ----- small helpers that return defaults if a field is missing ----- */

static double j_dbl(cJSON *parent, const char *key, double def) {
    if (!parent) return def;
    cJSON *n = cJSON_GetObjectItemCaseSensitive(parent, key);
    return cJSON_IsNumber(n) ? n->valuedouble : def;
}

static unsigned int j_uint(cJSON *parent, const char *key, unsigned int def) {
    if (!parent) return def;
    cJSON *n = cJSON_GetObjectItemCaseSensitive(parent, key);
    if (!cJSON_IsNumber(n)) return def;
    if (n->valuedouble < 0) return 0;
    return (unsigned int)n->valuedouble;
}

static const char *j_str(cJSON *parent, const char *key, const char *def) {
    if (!parent) return def;
    cJSON *n = cJSON_GetObjectItemCaseSensitive(parent, key);
    return cJSON_IsString(n) ? n->valuestring : def;
}

static char *slurp_file(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "readJson: cannot open '%s'\n", filename);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long sz = ftell(fp);
    if (sz < 0) { fclose(fp); return NULL; }
    rewind(fp);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(fp); return NULL; }
    size_t r = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[r] = '\0';
    return buf;
}

JsonData *readJsonData(const char *filename) {
    char *text = slurp_file(filename);
    if (!text) return NULL;

    cJSON *root = cJSON_Parse(text);
    if (!root) {
        const char *err = cJSON_GetErrorPtr();
        fprintf(stderr, "readJson: parse error near '%s'\n", err ? err : "?");
        free(text);
        return NULL;
    }

    JsonData *d = (JsonData *)calloc(1, sizeof(JsonData));
    if (!d) { cJSON_Delete(root); free(text); return NULL; }

    /* kind */
    const char *kind = j_str(root, "kind", "decode");
    d->kind = (strcmp(kind, "prefill") == 0) ? 1 : 0;

    /* model.* */
    cJSON *model = cJSON_GetObjectItemCaseSensitive(root, "model");
    d->model_d    = j_uint(model, "d",          4096);
    d->n_layers   = j_uint(model, "n_layers",   32);
    d->n_matrices = j_uint(model, "n_matrices", 7);
    d->s_dtype    = j_uint(model, "s_dtype",    2);

    /* hw.* */
    cJSON *hw = cJSON_GetObjectItemCaseSensitive(root, "hw");
    d->n_cxl = j_uint(hw, "n_cxl", 4);
    if (d->n_cxl == 0) d->n_cxl = 1;

    /* per-step scalars */
    d->input_tokens_per_request = j_uint(root, "input_tokens_per_request", 1);
    d->base_model_compute_ns    = j_dbl(root, "base_model_compute_ns", 0.0);

    /* kv.* */
    cJSON *kv = cJSON_GetObjectItemCaseSensitive(root, "kv");
    d->kv_tokens_total    = j_uint(kv, "kv_tokens_total", 0);
    d->kv_in_gpu_fraction = j_dbl(kv, "kv_in_gpu_fraction", 0.0);

    /* adapters[] */
    cJSON *adapters = cJSON_GetObjectItemCaseSensitive(root, "adapters");
    if (!cJSON_IsArray(adapters)) {
        fprintf(stderr, "readJson: 'adapters' must be an array\n");
        cJSON_Delete(root);
        free(text);
        free(d);
        return NULL;
    }

    int n = cJSON_GetArraySize(adapters);
    d->n_adapters = (unsigned int)(n > 0 ? n : 0);
    d->adapters = (AdapterDecision *)calloc((size_t)(n > 0 ? n : 1),
                                            sizeof(AdapterDecision));
    if (!d->adapters) {
        cJSON_Delete(root);
        free(text);
        free(d);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        cJSON *a = cJSON_GetArrayItem(adapters, i);
        d->adapters[i].id        = j_uint(a, "id",        (unsigned)i);
        d->adapters[i].rank      = j_uint(a, "rank",      16);
        d->adapters[i].batch     = j_uint(a, "batch",     0);
        d->adapters[i].strategy  = j_uint(a, "strategy",  3);
        d->adapters[i].kv_tokens = j_uint(a, "kv_tokens", 0);

        if (d->adapters[i].strategy < 1 || d->adapters[i].strategy > 4) {
            fprintf(stderr,
                    "readJson: adapter id=%u has invalid strategy=%u, defaulting to 3 (E3)\n",
                    d->adapters[i].id, d->adapters[i].strategy);
            d->adapters[i].strategy = 3;
        }
    }

    cJSON_Delete(root);
    free(text);
    return d;
}

void freeJsonData(JsonData *d) {
    if (!d) return;
    if (d->adapters) free(d->adapters);
    free(d);
}
