#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/ssd.h"

/* #define DEBUG */

/* ============================================================================
 * CLoRA driver: consume a per-step JSON (one decoder pass) and emit all the
 * CXL-link / NDP traffic + base-model GPU compute for the strategies chosen
 * upstream by the Python plugin (E1..E4).
 *
 * One JSON == one decoder pass.  Per adapter:
 *   E1  -> no CXL traffic (LoRA matrices cached in GPU HBM)
 *   E2  -> one READ            (loads A,B over CXL)
 *   E3  -> one READ_COMPUTE    (NDP computes the full LoRA path)
 *   E4  -> one READ_COMPUTE    (NDP computes m=xA, GPU finishes y=mB)
 * Plus one attention READ_COMPUTE per adapter, distributed across n_cxl
 * devices (token-level partitioning).  Plus one NPU_COMPUTE for the base
 * model.  All sizes are in BYTES; the simulator's bandwidths are in B/ns.
 * ============================================================================ */

struct npu_request *create_npu_request(struct cxl_switch_device_info *ssd,
                                       unsigned int req_type, int64_t duration) {
    struct npu_request *r = (struct npu_request *)malloc(sizeof(struct npu_request));
    alloc_assert(r, "npu_request");
    memset(r, 0, sizeof(*r));

    r->type       = req_type;
    r->begin_time = ssd->current_time;
    r->end_time   = ssd->current_time + duration;
    r->idx        = ssd->npu->req_idx++;

    if (ssd->npu->npu_reqs_head == NULL) {
        ssd->npu->npu_reqs_head = r;
        ssd->npu->npu_reqs_tail = r;
    } else {
        ssd->npu->npu_reqs_tail->next_node = r;
        ssd->npu->npu_reqs_tail = r;
    }
    ssd->npu->npu_reqs_queue_length++;
    return r;
}

struct request *create_ssd_request(struct cxl_switch_device_info *ssd,
                                   unsigned int ope,
                                   struct addr_info *addr, unsigned int addr_num) {
    struct request *req = (struct request *)malloc(sizeof(struct request));
    alloc_assert(req, "ssd_request");
    memset(req, 0, sizeof(*req));

    req->time         = ssd->current_time;
    req->idx          = ssd->npu->req_idx++;
    req->size         = addr_num;
    req->operation    = ope;
    req->begin_time   = ssd->current_time;
    req->next_node    = NULL;
    req->distri_flag  = 0;
    req->subs         = NULL;
    req->need_distr_flag = NULL;
    req->addr         = addr;
    req->addr_num     = addr_num;

    /* statistics */
    if (ope == READ) {
        ssd->ave_read_size = (ssd->ave_read_size * ssd->read_request_count + req->size)
                             / (ssd->read_request_count + 1);
        ssd->read_request_size  += req->size * ssd->parameter->subpage_capacity;
        ssd->read_request_c_a_size += 7 * req->size / ssd->parameter->subpage_page;
    } else if (ope == READ_COMPUTE) {
        unsigned int in_bytes = 0, out_bytes = 0;
        for (unsigned int i = 0; i < addr_num; i++) {
            in_bytes  += addr[i].input_size;
            out_bytes += addr[i].output_size;
        }
        ssd->rc_request_input_size  += in_bytes;
        ssd->rc_request_result_size += out_bytes;
        ssd->rc_request_c_a_size    += 7 * req->size / ssd->parameter->subpage_page;
    } else if (ope == WRITE) {
        ssd->ave_write_size = (ssd->ave_write_size * ssd->write_request_count + req->size)
                              / (ssd->write_request_count + 1);
        ssd->write_request_size += req->size * ssd->parameter->subpage_capacity;
    }

    if (ssd->request_queue == NULL) {
        ssd->request_queue = req;
        ssd->request_tail  = req;
    } else {
        ssd->request_tail->next_node = req;
        ssd->request_tail = req;
    }
    ssd->request_queue_length++;
    return req;
}

static int is_req_finished(struct cxl_switch_device_info *ssd, unsigned int idx) {
    struct Set *s = ssd->npu->finished_reqs;
    for (size_t i = 0; i < s->used; i++) {
        if (s->array[i] == idx) return 1;
    }
    return 0;
}

static void npu_go_one_step(struct cxl_switch_device_info *ssd) {
    if (ssd->npu->step_issue[ssd->npu->step] == 0) return;

    unsigned int next = ssd->npu->step + 1;
    struct Set *waits = ssd->npu->step_wait_reqs[next];
    for (size_t i = 0; i < waits->used; i++) {
        if (!is_req_finished(ssd, waits->array[i])) return;
    }

    printf("step%u finished, time=%lld ns = %.6f ms\n",
           ssd->npu->step, (long long)ssd->current_time, ssd->current_time / 1e6);
    ssd->npu->step++;
}

/* ----- request emitters ------------------------------------------------- */

static struct addr_info *make_addr_array(unsigned int n) {
    struct addr_info *a = (struct addr_info *)malloc(n * sizeof(struct addr_info));
    alloc_assert(a, "addr_info[]");
    memset(a, 0, n * sizeof(struct addr_info));
    return a;
}

/* Pick the CXL device this adapter's matrices live on.  Simple round-robin
 * by adapter id.  Production code would track actual placement. */
static unsigned int adapter_home_device(unsigned int adapter_id,
                                        unsigned int n_cxl) {
    return adapter_id % n_cxl;
}

/* Emit the LoRA path for one adapter.  Returns the request idx to wait on, or
 * (unsigned)-1 if no request was created (E1, batch=0). */
static unsigned int emit_lora_for_adapter(struct cxl_switch_device_info *ssd,
                                          const AdapterDecision *ad,
                                          const JsonData *j) {
    if (ad->batch == 0) return (unsigned)-1;

    /* 64-bit so prefill-scale products (huge effective batch) don't overflow. */
    uint64_t D = j->model_d;
    uint64_t S = j->s_dtype;
    uint64_t per_layer_mat = (uint64_t)j->n_layers * j->n_matrices;
    uint64_t batch = ad->batch;
    unsigned int home = adapter_home_device(ad->id, j->n_cxl);

    switch (ad->strategy) {
    case 1: /* E1 -- cached on GPU */
        return (unsigned)-1;

    case 2: { /* E2: pull A and B over the CXL link once per batch */
        struct addr_info *addr = make_addr_array(1);
        addr[0].gpu_id      = 0;
        addr[0].cxl_id      = home;
        addr[0].size        = 2u * ad->rank * D * S * per_layer_mat;
        addr[0].input_size  = 0;
        addr[0].output_size = 0;
        struct request *r = create_ssd_request(ssd, READ, addr, 1);
        slice_request(ssd);
        return r->idx;
    }

    case 3: { /* E3: NDP runs full LoRA path; only x in, y back */
        struct addr_info *addr = make_addr_array(1);
        addr[0].gpu_id      = 0;
        addr[0].cxl_id      = home;
        addr[0].size        = 2u * ad->rank * D * S * per_layer_mat;
        addr[0].input_size  = batch * D * S * per_layer_mat;
        addr[0].output_size = batch * D * S * per_layer_mat;
        struct request *r = create_ssd_request(ssd, READ_COMPUTE, addr, 1);
        slice_request(ssd);
        return r->idx;
    }

    case 4: { /* E4: NDP computes m=xA only; GPU handles y=mB */
        struct addr_info *addr = make_addr_array(1);
        addr[0].gpu_id      = 0;
        addr[0].cxl_id      = home;
        addr[0].size        = ad->rank * D * S * per_layer_mat;
        addr[0].input_size  = batch * D * S * per_layer_mat;
        addr[0].output_size = batch * ad->rank * S * per_layer_mat;
        struct request *r = create_ssd_request(ssd, READ_COMPUTE, addr, 1);
        slice_request(ssd);
        return r->idx;
    }

    default:
        fprintf(stderr, "emit_lora: unknown strategy %u for adapter %u\n",
                ad->strategy, ad->id);
        return (unsigned)-1;
    }
}

/* Emit the attention path for one adapter, distributed across all CXL devices
 * at the token level (paper §5.2).
 *
 * In decode, Q has shape (B, D). Every device needs the *full* Q to compute
 * attention on its KV slice, and every device returns a *full* partial output
 * O(i) of shape (B, D). KV is the only quantity sharded across devices.
 * Per-device transfer = 2*B*D*S bytes per layer (Q in, partial O back); the
 * devices' links operate in parallel, matching paper Eq (8).
 *
 * Optionally, a fraction kv_in_gpu_fraction of the KV cache is duplicated in
 * GPU HBM. Each CXL device only needs to read (1 - kv_in_gpu_fraction) of its
 * KV slice from device DRAM.
 */
static unsigned int emit_attention_for_adapter(struct cxl_switch_device_info *ssd,
                                               const AdapterDecision *ad,
                                               const JsonData *j) {
    if (ad->batch == 0 || ad->kv_tokens == 0) return (unsigned)-1;

    /* 64-bit so prefill-scale products don't overflow. */
    uint64_t D = j->model_d;
    uint64_t S = j->s_dtype;
    uint64_t NL = j->n_layers;
    uint64_t batch = ad->batch;
    unsigned int n_cxl = j->n_cxl;
    if (n_cxl == 0) n_cxl = 1;

    /* Prefill (kind==1, paper §5.2): the GPU computes the whole prompt's KV
     * during the forward pass (attention runs on GPU as FlashAttention; that
     * compute is charged in the GPU NPU_COMPUTE term).  The CXL-side cost is
     * therefore *writing* the freshly built KV cache out to the devices,
     * distributed at the token level.  Per device per layer we write
     * 2*D*S*(kv_tokens/n_cxl) bytes; the n_cxl device links operate in
     * parallel, so we issue one WRITE whose sub-requests fan across devices. */
    if (j->kind == 1) {
        uint64_t kv_per_device = ad->kv_tokens / n_cxl;
        if (kv_per_device == 0) kv_per_device = 1;
        uint64_t per_dev_write = 2u * D * S * kv_per_device * NL;
        if (per_dev_write == 0) per_dev_write = 1;
        struct addr_info *addr = make_addr_array(n_cxl);
        for (unsigned int c = 0; c < n_cxl; c++) {
            addr[c].gpu_id      = 0;
            addr[c].cxl_id      = c;
            addr[c].size        = per_dev_write;
            addr[c].input_size  = 0;
            addr[c].output_size = 0;
        }
        struct request *r = create_ssd_request(ssd, WRITE, addr, n_cxl);
        slice_request(ssd);
        return r->idx;
    }

    /* KV cache slice per device, reduced by the GPU-resident fraction. */
    uint64_t kv_per_device = ad->kv_tokens / n_cxl;
    if (kv_per_device == 0) kv_per_device = 1;
    double p_cxl = 1.0 - j->kv_in_gpu_fraction;
    if (p_cxl < 0.0) p_cxl = 0.0;
    if (p_cxl > 1.0) p_cxl = 1.0;

    uint64_t per_dev_dram = (uint64_t)(
        2.0 * D * S * (double)kv_per_device * (double)NL * p_cxl);
    if (per_dev_dram == 0) per_dev_dram = 1;
    /* Full Q replicated to each device per layer; full partial O returned. */
    uint64_t per_dev_input  = batch * D * S * NL;
    uint64_t per_dev_output = batch * D * S * NL;

    struct addr_info *addr = make_addr_array(n_cxl);
    for (unsigned int c = 0; c < n_cxl; c++) {
        addr[c].gpu_id      = 0;
        addr[c].cxl_id      = c;
        addr[c].size        = per_dev_dram;
        addr[c].input_size  = per_dev_input;
        addr[c].output_size = per_dev_output;
    }
    struct request *r = create_ssd_request(ssd, READ_COMPUTE, addr, n_cxl);
    slice_request(ssd);
    return r->idx;
}

/* ----- per-step issuer -------------------------------------------------- */

int npu_process(struct cxl_switch_device_info *ssd, JsonData *j) {
    if (!j) return 1;

    /* We use a 2-step state machine: step 0 issues every request; once they
     * all finish, step advances to 1 and the simulator drains. */
    if (ssd->npu->step != 0) {
        return 0;
    }

    if (ssd->npu->step_issue[0] == 0) {
#ifdef DEBUG
        printf("npu_process: issuing for %u adapter(s) at t=%lld\n",
               j->n_adapters, (long long)ssd->current_time);
#endif
        unsigned int n_cxl = j->n_cxl ? j->n_cxl : 1;
        if (n_cxl > ssd->parameter->cxl_channel_number) {
            n_cxl = ssd->parameter->cxl_channel_number;
        }
        /* mutate (safe -- single owner) so downstream uses clamped value */
        j->n_cxl = n_cxl;

        for (unsigned int i = 0; i < j->n_adapters; i++) {
            unsigned int idx;
            idx = emit_lora_for_adapter(ssd, &j->adapters[i], j);
            if (idx != (unsigned)-1) addSet(ssd->npu->step_wait_reqs[1], idx);

            idx = emit_attention_for_adapter(ssd, &j->adapters[i], j);
            if (idx != (unsigned)-1) addSet(ssd->npu->step_wait_reqs[1], idx);
        }

        if (j->base_model_compute_ns > 0.0) {
            double total = j->base_model_compute_ns * (double)j->n_layers;
            if (total < 1.0) total = 1.0;
            struct npu_request *r =
                create_npu_request(ssd, NPU_COMPUTE, (int64_t)total);
            addSet(ssd->npu->step_wait_reqs[1], r->idx);
        }

        ssd->npu->step_issue[0] = 1;
        ssd->npu->npu_finish    = 1;   /* nothing further to issue */
    }

    npu_go_one_step(ssd);
    return 0;
}

/* ----- request bookkeeping (npu side) ----------------------------------- */

void npu_trace_output(struct cxl_switch_device_info *ssd) {
    struct npu_request *prev = NULL, *cur = ssd->npu->npu_reqs_head;
    while (cur != NULL) {
        if (ssd->current_time >= cur->end_time) {
            addSet(ssd->npu->finished_reqs, cur->idx);
            struct npu_request *gone = cur;
            if (prev == NULL) {
                ssd->npu->npu_reqs_head = cur->next_node;
                if (ssd->npu->npu_reqs_head == NULL) ssd->npu->npu_reqs_tail = NULL;
            } else {
                prev->next_node = cur->next_node;
                if (prev->next_node == NULL) ssd->npu->npu_reqs_tail = prev;
            }
            cur = cur->next_node;
            free(gone);
            ssd->npu->npu_reqs_queue_length--;
        } else {
            prev = cur;
            cur  = cur->next_node;
        }
    }
}

/* ----- top-level simulate ----------------------------------------------- */

struct cxl_switch_device_info *simulate(struct cxl_switch_device_info *ssd,
                                        char *trace_filename) {
    fprintf(ssd->outputfile,
            "      arrive           lsn     size ope     begin time"
            "    response time    process time\n");
    fflush(ssd->outputfile);

    JsonData *jsonData = readJsonData(trace_filename);
    if (jsonData == NULL) {
        fprintf(stderr, "simulate: failed to read trace JSON: %s\n",
                trace_filename);
        return NULL;
    }

    ssd->simulation_start_time = ssd->current_time;
    int64_t safety_iters = 0;
    const int64_t safety_cap = (int64_t)2e8;  /* hard cap so we never wedge */

    while (1) {
        find_nearest_event_sys(ssd);
        npu_process(ssd, jsonData);
        process(ssd);
        npu_trace_output(ssd);
        trace_output(ssd);

        if (ssd->npu->npu_finish == 1 &&
            ssd->request_queue == NULL &&
            ssd->npu->npu_reqs_head == NULL) {
            break;
        }

        if (++safety_iters > safety_cap) {
            fprintf(stderr,
                    "simulate: safety cap of %lld iterations hit -- aborting.\n",
                    (long long)safety_cap);
            break;
        }
    }

    ssd->simulation_end_time = ssd->current_time;

    /* Final, machine-parseable line for the Python driver. */
    int64_t duration = ssd->simulation_end_time - ssd->simulation_start_time;
    printf("\nCLORA_RESULT duration_ns=%lld start_ns=%lld end_ns=%lld\n",
           (long long)duration,
           (long long)ssd->simulation_start_time,
           (long long)ssd->simulation_end_time);

    freeJsonData(jsonData);
    return ssd;
}

int main(int argc, char *argv[]) {
    struct user_args *uargs = (struct user_args *)malloc(sizeof(struct user_args));
    alloc_assert(uargs, "user args");
    memset(uargs, 0, sizeof(*uargs));

    if (parse_user_args(argc, argv, uargs) == -1) {
        display_help();
        free(uargs);
        return 0;
    }

    display_title();

    struct cxl_switch_device_info *ssd =
        (struct cxl_switch_device_info *)malloc(sizeof(struct cxl_switch_device_info));
    alloc_assert(ssd, "ssd");
    memset(ssd, 0, sizeof(*ssd));

    ssd = initialize_ssd(ssd, uargs);
    printf("finish initialize ssd\n");

    ssd = initiation(ssd);
    printf("finish ssd initiation (prepare inner structure)\n");

    display_simulation_intro(ssd);
    ssd = simulate(ssd, uargs->trace_filename);

    if (ssd != NULL) {
        statistic_output(ssd);
        close_file(ssd);
        free(ssd);
    }

    free(uargs);
    printf("\nThe simulation is completed!\n");
    return 0;
}
