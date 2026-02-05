#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/ssd.h"

#define DEBUG

struct npu_request *create_npu_request (struct cxl_switch_device_info * ssd, unsigned int req_type, int64_t duration){
    struct npu_request* request1;
    request1 = (struct npu_request*)malloc(sizeof(struct npu_request));
    alloc_assert(request1,"npu_request");
    memset(request1,0, sizeof(struct npu_request));

    request1->type = req_type;
    request1->begin_time = ssd->current_time;
    request1->end_time = ssd->current_time + duration;
    request1->idx = ssd->npu->req_idx;

    ssd->npu->req_idx += 1;

    if(ssd->npu->npu_reqs_head == NULL)          //The queue is empty
    {
        ssd->npu->npu_reqs_head = request1;
        ssd->npu->npu_reqs_tail = request1;
        ssd->npu->npu_reqs_queue_length += 1;
    }
    else
    {
        (ssd->npu->npu_reqs_tail)->next_node = request1;
        ssd->npu->npu_reqs_tail = request1;
        ssd->npu->npu_reqs_queue_length += 1;
    }

    return request1;
}

struct request *create_ssd_request (struct cxl_switch_device_info * ssd, unsigned int ope, struct addr_info * addr, unsigned int addr_num){
    struct request *request1;

    request1 = (struct request*)malloc(sizeof(struct request));
    alloc_assert(request1,"ssd_request");
    memset(request1,0, sizeof(struct request));

    request1->time = ssd->current_time;
    request1->idx = ssd->npu->req_idx;
    request1->size = addr_num;
    request1->operation = ope;
    request1->begin_time = ssd->current_time;
    request1->response_time = 0;
    request1->energy_consumption = 0;
    request1->next_node = NULL;
    request1->distri_flag = 0;              // indicate whether this request has been distributed already
    request1->subs = NULL;
    request1->need_distr_flag = NULL;
    request1->addr = addr;
    request1->addr_num = addr_num;

    ssd->npu->req_idx += 1;

    if (request1->operation==READ)             // 统计相关参数
    {
        ssd->ave_read_size=(ssd->ave_read_size*ssd->read_request_count+request1->size)/(ssd->read_request_count+1);
        ssd->read_request_size += request1->size * ssd->parameter->subpage_capacity;
        ssd->read_request_c_a_size += 7 * request1->size / ssd->parameter->subpage_page;
    }
    else if (request1->operation == READ_COMPUTE)
    {
        ssd->rc_request_input_size += 64;
        ssd->rc_request_result_size += ssd->parameter->cxl_channel_number * 64;
        ssd->rc_request_c_a_size += 7 * request1->size / ssd->parameter->subpage_page;
        ssd->rc_request_compute_ops += 64 * 64 * 64 * 2;
    }
    else if (request1->operation == WRITE)
    {
        ssd->ave_write_size=(ssd->ave_write_size*ssd->write_request_count+request1->size)/(ssd->write_request_count+1);
        ssd->write_request_size += request1->size * ssd->parameter->subpage_capacity;
    }

    if(ssd->request_queue == NULL)          //The queue is empty
    {
        ssd->request_queue = request1;
        ssd->request_tail = request1;
        ssd->request_queue_length++;
    }
    else
    {
        (ssd->request_tail)->next_node = request1;
        ssd->request_tail = request1;
        ssd->request_queue_length++;
    }

    return request1;
}

int is_req_finished(struct cxl_switch_device_info* ssd, unsigned int req_idx){
    struct Set *finished_reqs = ssd->npu->finished_reqs;
    for (int i = 0; i < finished_reqs->used; i++){
        if (finished_reqs->array[i] == req_idx){
            return 1;
        }
    }
    return 0;
}

int npu_go_one_step(struct cxl_switch_device_info * ssd){
    unsigned int req_idx;
    int req_finished;
    int go_one_step_flag = 1;

    if (ssd->npu->step_issue[ssd->npu->step] == 0){
        return 0;
    }

    // 决定npu是否可以进入下一个step
    unsigned int next_step = ssd->npu->step + 1;
    struct Set *next_step_wait_reqs = ssd->npu->step_wait_reqs[next_step];
    for (int i = 0; i < next_step_wait_reqs->used; i++){
        req_idx = next_step_wait_reqs->array[i];
        req_finished = is_req_finished(ssd, req_idx);
        if (req_finished == 0){     // 还有请求没完成，不能进入下一step
            go_one_step_flag = 0;
            break;
        }
    }

    // npu 进入下一个step
    if (go_one_step_flag == 1){
        printf("step%d finished, time=%lldns = %.6fms\n", ssd->npu->step, ssd->current_time, ssd->current_time/1e6);
        ssd->npu->step += 1;
    }


    return 0;
}

int npu_process (struct cxl_switch_device_info *ssd, JsonData *jsonData){
    if (jsonData == NULL) {
        fprintf(stderr, "Error reading JSON data\n");
        return 1;
    }
    struct request *request;
    struct npu_request *npu_request;
    struct npu_info *npu = ssd->npu;
#ifdef DEBUG
    printf("enter npu_process, current time:%lld\n", ssd->current_time);
    printf("npu step: %d\n", ssd->npu->step);
#endif
    int addr_num = 1;
    int64_t t;
    struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info)); 
    unsigned int input_size = jsonData->input_token_num;//采用 FP16

    double plan_1=jsonData->b1,plan_2=jsonData->b2,plan_3=jsonData->b3;

    if (ssd->npu->step == 0) {          // step 0 --------------------------------------------
        if (npu->step_issue[0] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        // Q_(𝑛+1)=X 𝑊_𝑄
        // for(int i = 0; i<jsonData->q_rc_req_arrays; i++) {
        //     addr_num = jsonData->q_rc_req[i][2];
        //     struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
        //     for(unsigned int j = 0;j < addr_num; j++){
        //         addr[j].gpu_id = jsonData->q_rc_req[i][0];
        //         addr[j].cxl_id = jsonData->q_rc_req[i][1];
        //         addr[j].size = jsonData->q_rc_req[i][3];
        //         addr[j].input_size = input_size;
        //     }
        //     request = create_ssd_request(ssd, READ_COMPUTE, addr, addr_num);
        //     slice_request(ssd);
        //     addSet(ssd->npu->step_wait_reqs[1], request->idx);
        //     addr_num = 1;
        // }

        // for(int i = 0; i<jsonData->q_r_req_arrays; i++) {
        //     addr_num = jsonData->q_r_req[i][2];
        //     struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
        //     for(unsigned int j = 0;j < addr_num; j++){
        //         addr[j].gpu_id = jsonData->q_r_req[i][0];
        //         addr[j].cxl_id = jsonData->q_r_req[i][1];
        //         addr[j].size   = jsonData->q_r_req[i][3];
        //     }
        //     request = create_ssd_request(ssd, READ, addr, addr_num);
        //     slice_request(ssd);
        //     addSet(ssd->npu->step_wait_reqs[1], request->idx);
        //     addr_num = 1;
        // }

        for(int i = 0; i<jsonData->q_r_req_arrays; i++) {
            addr_num = jsonData->q_r_req[i][2];
            addr_num = 10;
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                // addr[j].gpu_id = jsonData->q_r_req[i][0];
                // addr[j].cxl_id = jsonData->q_r_req[i][1];
                // addr[j].size   = jsonData->q_r_req[i][3];
                addr[j].gpu_id = 0;
                addr[j].cxl_id = 0;
                addr[j].size   = 100000;
                addr[j].input_size = 256;
                addr[j].output_size = 256;
            }
            for (int i = 0; i < 30; i++){
                request = create_ssd_request(ssd, READ_COMPUTE, addr, addr_num);
                slice_request(ssd);
                request = create_ssd_request(ssd, READ, addr, addr_num);
                slice_request(ssd);
            }
            addSet(ssd->npu->step_wait_reqs[1], request->idx);
            addr_num = 1;
        }


        npu->step_issue[0] = 1;    // 该step的请求已经下发过一次
        
        ssd->npu->npu_finish = 1;   // 单步测试用
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 1){    // step 1 --------------------------------------------
        if (npu->step_issue[1] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //𝐾_(𝑛+1)=X 𝑊_K
        for(int i = 0; i<jsonData->k_rc_req_arrays; i++) {
            addr_num = jsonData->k_rc_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->k_rc_req[i][0];
                addr[j].cxl_id = jsonData->k_rc_req[i][1];
                addr[j].size = 4*jsonData->k_rc_req[i][3];
                addr[j].input_size = input_size;
            }
            request = create_ssd_request(ssd, READ_COMPUTE, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[2], request->idx);
            addr_num = 1;
        }

        for(int i = 0; i<jsonData->k_r_req_arrays; i++) {
            addr_num = jsonData->k_r_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->k_r_req[i][0];
                addr[j].cxl_id = jsonData->k_r_req[i][1];
                addr[j].size = 4*jsonData->k_r_req[i][3];
            }
            request = create_ssd_request(ssd, READ, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[2], request->idx);
            addr_num = 1;
        }

        //Load K1-Kn
        t=jsonData->load_k_latency;
        npu_request = create_npu_request(ssd, NPU_READ_DRAM, t);
        addSet(ssd->npu->step_wait_reqs[5], npu_request->idx);

        //Q_(𝑛+1)=X 𝑊_𝑄(Core)
        t=jsonData->q_0_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[2], npu_request->idx);
        
        npu->step_issue[1] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 2){    // step 2 --------------------------------------------
        if (npu->step_issue[2] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //𝐾_(𝑛+1)=X 𝑊_𝐾(Core)
        t=jsonData->k_0_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[3], npu_request->idx);

        //∆Q_(𝑛+1)=X 𝐵_𝑄 𝐴_𝑄(RC)
        //新函数，用来确定三种方案 比重并模拟，目前仅做依赖参考
        t=jsonData->tmp_lora_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[3], npu_request->idx);


        //𝑉_(𝑛+1)=X 𝑊_𝑉
        for(int i = 0; i<jsonData->v_rc_req_arrays; i++) {
            addr_num = jsonData->v_rc_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->v_rc_req[i][0];
                addr[j].cxl_id = jsonData->v_rc_req[i][1];
                addr[j].size = 4*jsonData->v_rc_req[i][3];
                addr[j].input_size = input_size;
            }
            request = create_ssd_request(ssd, READ_COMPUTE, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[7], request->idx);
            addr_num = 1;
        }

        for(int i = 0; i<jsonData->v_r_req_arrays; i++) {
            addr_num = jsonData->v_r_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->v_r_req[i][0];
                addr[j].cxl_id = jsonData->v_r_req[i][1];
                addr[j].size = 4*jsonData->v_r_req[i][3];
            }
            request = create_ssd_request(ssd, READ, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[7], request->idx);
            addr_num = 1;
        }

        npu->step_issue[2] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 3){    // step 3 --------------------------------------------
        if (npu->step_issue[3] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //∆𝐾_(𝑛+1)=X 𝐵_𝐾 𝐴_𝐾(C-RC)
        //新函数，用来确定三种方案 比重并模拟，目前仅做依赖参考
        t=jsonData->tmp_lora_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[4], npu_request->idx);

        //Q_(𝑛+1)’= Q_(𝑛+1)+ ∆Q_(𝑛+1)(Core)
        t=jsonData->q_1_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[5], npu_request->idx);

        npu->step_issue[3] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 4){    // step 4 --------------------------------------------
        if (npu->step_issue[4] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //𝐾_(𝑛+1)’= 𝐾_(𝑛+1)+ ∆𝐾_(𝑛+1)(Core)
        t=jsonData->k_1_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[5], npu_request->idx);

        //Load V1-Vn
        t=jsonData->load_v_latency;
        npu_request = create_npu_request(ssd, NPU_READ_DRAM, t);
        addSet(ssd->npu->step_wait_reqs[10], npu_request->idx);

        npu->step_issue[4] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 5){    // step 5 --------------------------------------------
        if (npu->step_issue[5] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //𝑃_{𝑛+1} = 𝑄′_{𝑛+1} [𝐾_1^′..𝐾′ _{𝑛+1}] (Core)
        t=jsonData->p_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[6], npu_request->idx);

        npu->step_issue[5] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 6){    // step 6 --------------------------------------------
        if (npu->step_issue[6] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        // 𝑆_(𝑛+1)= Softmax𝑃_(𝑛+1) (Core)
        t=jsonData->s_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[10], npu_request->idx);

        npu->step_issue[6] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 7){    // step 7 --------------------------------------------
        if (npu->step_issue[7] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //𝑉_(𝑛+1)=X 𝑊_𝑉(Core)
        t=jsonData->v_0_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[8], npu_request->idx);

        npu->step_issue[7] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 8){    // step 8 --------------------------------------------
        if (npu->step_issue[8] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //∆𝑉_(𝑛+1)=X 𝐵_𝑉 𝐴_𝑉(C-RC)
        //新函数，用来确定三种方案 比重并模拟，目前仅做依赖参考
        t=jsonData->tmp_lora_latency;
        npu_request = create_npu_request(ssd, NPU_SF_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[9], npu_request->idx);

        npu->step_issue[8] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 9){    // step 9 --------------------------------------------
        if (npu->step_issue[9] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //V_(𝑛+1)’= 𝑉_(𝑛+1)+ ∆𝑉_(𝑛+1)(Core)
        t=jsonData->v_1_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[10], npu_request->idx);

        npu->step_issue[9] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 10){    // step 10 --------------------------------------------
        if (npu->step_issue[10] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //𝐴_{𝑛+1} = 𝑆_{𝑛+1}  [𝑉′_{𝑛+1}..]
        t=jsonData->a_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[11], npu_request->idx);

        npu->step_issue[10] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 11){    // step 11 --------------------------------------------
        if (npu->step_issue[11] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //O_(𝑛+1)=𝑊_O  A_(n+1)
        for(int i = 0; i<jsonData->o_rc_req_arrays; i++) {
            addr_num = jsonData->o_rc_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->o_rc_req[i][0];
                addr[j].cxl_id = jsonData->o_rc_req[i][1];
                addr[j].size = 4*jsonData->o_rc_req[i][3];
                addr[j].input_size = input_size;
            }
            request = create_ssd_request(ssd, READ_COMPUTE, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[12], request->idx);
            addr_num = 1;
        }

        for(int i = 0; i<jsonData->o_r_req_arrays; i++) {
            addr_num = jsonData->o_r_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->o_r_req[i][0];
                addr[j].cxl_id = jsonData->o_r_req[i][1];
                addr[j].size = 4*jsonData->o_r_req[i][3];
            }
            request = create_ssd_request(ssd, READ, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[12], request->idx);
            addr_num = 1;
        }
        npu->step_issue[11] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 12){    // step 12 --------------------------------------------
        if (npu->step_issue[12] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //O_(𝑛+1)= 𝑊_O  A_(n+1)(Core)
        t=jsonData->o_0_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[13], npu_request->idx);

        npu->step_issue[12] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 13){    // step 13 --------------------------------------------
        if (npu->step_issue[13] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //∆O_(𝑛+1)=𝐵_O 𝐴_O  A_(n+1)(C-RC)
        //新函数，用来确定三种方案 比重并模拟，目前仅做依赖参考
        t=jsonData->tmp_lora_latency;
        npu_request = create_npu_request(ssd, NPU_SF_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[14], npu_request->idx);

        npu->step_issue[13] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 14){    // step 14 --------------------------------------------
        if (npu->step_issue[14] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //O_(𝑛+1)’= O_(𝑛+1)+ ∆O_(𝑛+1)(Core)
        t=jsonData->o_1_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[15], npu_request->idx);

        npu->step_issue[14] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 15){    // step 15 --------------------------------------------
        if (npu->step_issue[15] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //G_(𝑛+1)=𝑊_G  〖O’〗_(n+1)
        for(int i = 0; i<jsonData->g_rc_req_arrays; i++) {
            addr_num = jsonData->g_rc_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->g_rc_req[i][0];
                addr[j].cxl_id = jsonData->g_rc_req[i][1];
                addr[j].size = 4*jsonData->g_rc_req[i][3];
                addr[j].input_size = input_size;
            }
            request = create_ssd_request(ssd, READ_COMPUTE, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[16], request->idx);
            addr_num = 1;
        }

        for(int i = 0; i<jsonData->g_r_req_arrays; i++) {
            addr_num = jsonData->g_r_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->g_r_req[i][0];
                addr[j].cxl_id = jsonData->g_r_req[i][1];
                addr[j].size = 4*jsonData->g_r_req[i][3];
            }
            request = create_ssd_request(ssd, READ, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[16], request->idx);
            addr_num = 1;
        }
        
        npu->step_issue[15] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 16){    // step 16 --------------------------------------------
        if (npu->step_issue[16] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //𝐺_(𝑛+1)= 𝑊_𝐺  〖𝑂′〗_(n+1)(Core)
        t=jsonData->g_0_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[17], npu_request->idx);

        //𝑈_(𝑛+1)=𝑊_𝑈  〖O’〗_(n+1)
        for(int i = 0; i<jsonData->u_rc_req_arrays; i++) {
            addr_num = jsonData->u_rc_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->u_rc_req[i][0];
                addr[j].cxl_id = jsonData->u_rc_req[i][1];
                addr[j].size = 4*jsonData->u_rc_req[i][3];
                addr[j].input_size = input_size;
            }
            request = create_ssd_request(ssd, READ_COMPUTE, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[17], request->idx);
            addr_num = 1;
        }

        for(int i = 0; i<jsonData->u_r_req_arrays; i++) {
            addr_num = jsonData->u_r_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->u_r_req[i][0];
                addr[j].cxl_id = jsonData->u_r_req[i][1];
                addr[j].size = 4*jsonData->u_r_req[i][3];
            }
            request = create_ssd_request(ssd, READ, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[17], request->idx);
            addr_num = 1;
        }
        
        npu->step_issue[16] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 17){    // step 17 --------------------------------------------
        if (npu->step_issue[17] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //∆𝐺_(𝑛+1)=𝐵_𝐺 𝐴_𝐺  〖𝑂′〗_(n+1)(C-RC)
        //新函数，用来确定三种方案 比重并模拟，目前仅做依赖参考
        t=jsonData->tmp_lora_latency;
        npu_request = create_npu_request(ssd, NPU_SF_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[18], npu_request->idx);

        //𝑈_(𝑛+1)= 𝑊_𝑈  〖𝑂′〗_(n+1)(Core)
        t=jsonData->u_0_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[18], npu_request->idx);

        npu->step_issue[17] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 18){    // step 18 --------------------------------------------
        if (npu->step_issue[18] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //〖𝐺′〗_(𝑛+1)’= 𝐺_(𝑛+1)+ ∆𝐺_(𝑛+1)(Core)
        t=jsonData->g_1_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[20], npu_request->idx);

        //∆𝑈_(𝑛+1)=𝐵_𝑈 𝐴_𝑈  〖𝑂′〗_(n+1)(C-RC)
        //新函数，用来确定三种方案 比重并模拟，目前仅做依赖参考
        t=jsonData->tmp_lora_latency;
        npu_request = create_npu_request(ssd, NPU_SF_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[19], npu_request->idx);

        npu->step_issue[18] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 19){    // step 19 --------------------------------------------
        if (npu->step_issue[19] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //〖𝑈′〗_(𝑛+1)’= 𝑈_(𝑛+1)+ ∆𝑈_(𝑛+1)(Core)
        t=jsonData->u_1_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[20], npu_request->idx);

        npu->step_issue[19] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 20){    // step 20 --------------------------------------------
        if (npu->step_issue[20] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //Output_{n+1} = W_1 (G'_{n+1} + U'_{n+1})
        for(int i = 0; i<jsonData->output_rc_req_arrays; i++) {
            addr_num = jsonData->output_rc_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->output_rc_req[i][0];
                addr[j].cxl_id = jsonData->output_rc_req[i][1];
                addr[j].size = 4*jsonData->output_rc_req[i][3];
                addr[j].input_size = input_size;
            }
            request = create_ssd_request(ssd, READ_COMPUTE, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[21], request->idx);
            addr_num = 1;
        }

        for(int i = 0; i<jsonData->output_r_req_arrays; i++) {
            addr_num = jsonData->output_r_req[i][2];
            struct addr_info *addr = (struct addr_info *)malloc(addr_num*sizeof(struct addr_info));
            for(unsigned int j = 0;j < addr_num; j++){
                addr[j].gpu_id = jsonData->output_r_req[i][0];
                addr[j].cxl_id = jsonData->output_r_req[i][1];
                addr[j].size = 4*jsonData->output_r_req[i][3];
            }
            request = create_ssd_request(ssd, READ, addr, addr_num);
            slice_request(ssd);
            addSet(ssd->npu->step_wait_reqs[21], request->idx);
            addr_num = 1;
        }
         
        npu->step_issue[20] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 21){    // step 21 --------------------------------------------
        if (npu->step_issue[21] == 1){
            npu_go_one_step(ssd);
            return 0;
        }

        //Output_{n+1} = W_1 (G'_{n+1} + U'_{n+1})(Core)
        t=jsonData->output_latency;
        npu_request = create_npu_request(ssd, NPU_COMPUTE, t);
        addSet(ssd->npu->step_wait_reqs[22], npu_request->idx);

        npu->step_issue[21] = 1;    // 该step的请求已经下发过一次
        npu_go_one_step(ssd);
    } else if (ssd->npu->step == 22){    // step 22 --------------------------------------------
        ssd->npu->npu_finish = 1;//结束
    }


    return 0;
}

void npu_trace_output(struct cxl_switch_device_info* ssd) {
#ifdef DEBUG
    printf("enter trace_output, ssd's current time:%lld\n", ssd->current_time);
    printf("ssd->npu->npu_reqs_queue_length:%d\n", ssd->npu->npu_reqs_queue_length);
#endif
    struct npu_request *pre_node = NULL, *req = ssd->npu->npu_reqs_head;
    while(req != NULL) {
        if (ssd->current_time >= req->end_time) { // 当前req已经结束
            addSet(ssd->npu->finished_reqs, req->idx); // 记录执行完成的requests
            if(pre_node == NULL) { // 当前req是整个ssd中第一个
                if(req->next_node == NULL) { //当前request是queue中最后一个
                    req = NULL;
                    ssd->npu->npu_reqs_head = NULL;
                    ssd->npu->npu_reqs_tail = NULL;
                    ssd->npu->npu_reqs_queue_length --;
                }
                else { // 当前req不是queue中最后一个
                    ssd->npu->npu_reqs_head = req->next_node;
                    pre_node = req;
                    req = req->next_node;
                    free(pre_node);
                    pre_node = NULL;
                    ssd->npu->npu_reqs_queue_length--;
                }
            }
            else { // 当前req不是整个ssd中第一个
                if(req->next_node == NULL) { // 当前req是queue中最后一个
                    pre_node->next_node = NULL;
                    free(req);
                    req = NULL;
                    ssd->npu->npu_reqs_tail = pre_node;
                    ssd->npu->npu_reqs_queue_length--;
                }
                else { // 当前req不是queue中最后一个
                    pre_node->next_node = req->next_node;
                    free(req);
                    req = pre_node->next_node;
                    ssd->npu->npu_reqs_queue_length--;
                }

            }
        }
        else { // 当前req还没有结束
            pre_node = req;
            req = req->next_node;
        }

    }
}


/******************simulate() *********************************************************************
 *simulate()是核心处理函数，主要实现的功能包括
 *1,从trace文件中获取一条请求，挂到ssd->request
 *2，根据ssd是否有dram分别处理读出来的请求，把这些请求处理成为读写子请求，挂到ssd->channel或者ssd上
 *3，按照事件的先后来处理这些读写子请求。
 *4，输出每条请求的子请求都处理完后的相关信息到outputfile文件中
 **************************************************************************************************/
struct cxl_switch_device_info *simulate(struct cxl_switch_device_info *ssd, char *trace_filename) {

    fprintf(ssd->outputfile, "      arrive           lsn     size ope     begin time    response time    process time\n");
    fflush(ssd->outputfile);

    JsonData *jsonData = readJsonData(trace_filename);
    if (jsonData == NULL) {
        fprintf(stderr, "Error reading NULL JSON data: %s\n", trace_filename);
        return NULL;
    }

    while (1) {
#ifdef DEBUG
    int sum_busy_channel = 0;
    printf("\n$$$$ time: %lld\n", ssd->current_time);
    for(int i=0; i<ssd->parameter->cxl_channel_number; i++) {
        if (ssd->bottom_channel_head[i].subs_r_head != NULL || ssd->bottom_channel_head[i].subs_rc_head != NULL || ssd->bottom_channel_head[i].subs_w_head != NULL) {
            sum_busy_channel++;
            printf("  -------------- cxl %d --------------\n", i);
            printf("cxlctrl%d/%d/%d & cxl_dram%d/%d/%d & cxl_pe%d/%d/%d\n", 
            ssd->bottom_channel_head[i].cxl_device->chip_head[0].cxlctrl->current_state,
            ssd->bottom_channel_head[i].cxl_device->chip_head[0].cxlctrl->next_state,
            ssd->bottom_channel_head[i].cxl_device->chip_head[0].cxlctrl->next_state_predict_time,
            ssd->bottom_channel_head[i].cxl_device->chip_head[0].cxldram->current_state,
            ssd->bottom_channel_head[i].cxl_device->chip_head[0].cxldram->next_state,
            ssd->bottom_channel_head[i].cxl_device->chip_head[0].cxldram->next_state_predict_time,
            ssd->bottom_channel_head[i].cxl_device->chip_head[0].pe->current_state,
            ssd->bottom_channel_head[i].cxl_device->chip_head[0].pe->next_state,
            ssd->bottom_channel_head[i].cxl_device->chip_head[0].pe->next_state_predict_time      
            );
            struct sub_request *sub = NULL;

            // 打印read sub
            sub = ssd->bottom_channel_head[i].subs_r_head;
            if(sub != NULL) printf("  read@@ ");
            while (sub != NULL){
                printf(" sub %d: state %d,from cxl%d & gpu%d & size%d & num%d  | ",sub->idx, sub->current_state, sub->p_addr[0]->cxl_id, sub->p_addr[0]->gpu_id, sub->p_addr[0]->size, sub->addr_num);
                sub = sub->next_node;
            }
            if(ssd->bottom_channel_head[i].subs_r_head != NULL) printf("\n");

            // 打印read compute sub
            sub = ssd->bottom_channel_head[i].subs_rc_head;
            if(sub != NULL) printf("  read_compute@@ ");
            while (sub != NULL){
                printf(" sub %d: state %d,from cxl%d & gpu%d & size%d & num%d & next_stage_time%d | ",sub->idx, sub->current_state, sub->p_addr[0]->cxl_id, sub->p_addr[0]->gpu_id, sub->p_addr[0]->size, sub->addr_num, sub->next_state_predict_time);
                sub = sub->next_node;
            }
            if(ssd->bottom_channel_head[i].subs_rc_head != NULL) printf("\n");

            // 打印write sub
            sub = ssd->bottom_channel_head[i].subs_w_head;
            if(sub != NULL) printf("  write@@ ");
            while (sub != NULL){
                printf(" sub %d: state %d,from cxl%d & gpu%d & size%d & num%d  | ",sub->idx, sub->current_state, sub->p_addr[0]->cxl_id, sub->p_addr[0]->gpu_id, sub->p_addr[0]->size, sub->addr_num);
                sub = sub->next_node;
            }
            if(ssd->bottom_channel_head[i].subs_w_head != NULL) printf("\n");
        }
    }
    printf("** cxl non empty channels: %d\n", sum_busy_channel);
    // 刷新输出，使得输出能够正确的输入到ouput文件中
    fflush(stdout);
#endif
        // update system time
        find_nearest_event_sys(ssd);

        // npu step
        npu_process(ssd, jsonData);

        // ssd step
        process(ssd); // 处理ssd侧的状态变化

        // trace all finished npu request
        npu_trace_output(ssd);

        // trace all finished ssd request
        trace_output(ssd);

        // end of simulation
        if((ssd->npu->npu_finish == 1) && (ssd->request_queue == NULL) && (ssd->npu->npu_reqs_head == NULL)) {
            break;
        }
    }

    return ssd;
}


int main(int argc, char *argv[]) {
    struct user_args *uargs = (struct user_args*) malloc(sizeof(struct user_args));
    alloc_assert(uargs, "user args");
    memset(uargs, 0, sizeof(struct user_args));

    if (parse_user_args(argc, argv, uargs) == -1) {
        display_help();
        return 0;
    }

    display_title();

    // simulate_ssd is the main function to initialize and simulate a single ssd device
    struct cxl_switch_device_info *ssd = (struct cxl_switch_device_info*) malloc(sizeof(struct cxl_switch_device_info));
    alloc_assert(ssd,"ssd");
    memset(ssd, 0, sizeof(struct cxl_switch_device_info));

    ssd = initialize_ssd(ssd, uargs);
    printf("finish initialize ssd\n");

    ssd = initiation(ssd);
    printf("finish ssd initiation (prepare inner structure)\n");

    display_simulation_intro(ssd);
    ssd = simulate(ssd, uargs->trace_filename);

    statistic_output(ssd);
    close_file(ssd);

    free(ssd);

    free(uargs);
    printf("\nThe simulation is completed! \n");

    return 0;
}

