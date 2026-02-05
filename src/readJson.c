#include <stdio.h>
#include <stdlib.h>
#include "../include/readJson.h"
#include "../lib/cJSON/cJSON.h"

// 函数定义用于读取和解析 JSON 数据
JsonData* readJsonData(const char *filename) {
    // 分配内存来存储解析后的 JSON 数据
    JsonData *jsonData = (JsonData *)malloc(sizeof(JsonData));
    if (jsonData == NULL) {
        perror("Memory allocation failed");
        return NULL;
    }

    // 初始化字段
    jsonData->q_0_latency = 10;
    jsonData->q_1_latency = 10;
    jsonData->k_0_latency = 10;
    jsonData->k_1_latency = 10;
    jsonData->p_latency = 10;
    jsonData->s_latency = 10;
    jsonData->v_0_latency = 10;
    jsonData->v_1_latency = 10;
    jsonData->a_latency = 10;
    jsonData->o_0_latency = 10;
    jsonData->o_1_latency = 10;
    jsonData->g_0_latency = 10;
    jsonData->g_1_latency = 10;
    jsonData->u_0_latency = 10;
    jsonData->u_1_latency = 10;
    jsonData->output_latency = 10;
    jsonData->load_k_latency = 10;
    jsonData->load_v_latency = 10;
    jsonData->tmp_lora_latency = 10;

    jsonData->input_token_num = 400;//假设输入了三个token
    

    jsonData->b1 = 0.9990;
    jsonData->b2 = 0.0000075;
    jsonData->b3 = 0.00098;

    /*jsonData->ndp_emb_lookup_rc_req = NULL;
    jsonData->num_ndp_emb_lookup_rc_req_arrays = 0;*/
    
    jsonData->q_rc_req[0][0] = 1;           //cxlid
    jsonData->q_rc_req[0][1] = 1;           //gpuid
    jsonData->q_rc_req[0][2] = 32;      //addr_num-heads
    jsonData->q_rc_req[0][3] = 4096*4096/32;          //单位B_单笔size

    //outputsize,inputsize

    jsonData->q_r_req[0][0] = 1;
    jsonData->q_r_req[0][1] = 1;
    jsonData->q_r_req[0][2] = 1;
    jsonData->q_r_req[0][3] = 4096*4096;    //Wq 4096*4096

    jsonData->k_rc_req[0][0] = 1;
    jsonData->k_rc_req[0][1] = 1;
    jsonData->k_rc_req[0][2] = 32;
    jsonData->k_rc_req[0][3] = 4096*4096/32;

    jsonData->k_r_req[0][0] = 1;
    jsonData->k_r_req[0][1] = 1;
    jsonData->k_r_req[0][2] = 1;
    jsonData->k_r_req[0][3] = 4096*4096;    //Wk 4096*4096

    jsonData->v_rc_req[0][0] = 1;
    jsonData->v_rc_req[0][1] = 1;
    jsonData->v_rc_req[0][2] = 32;
    jsonData->v_rc_req[0][3] = 4096*4096/32;

    jsonData->v_r_req[0][0] = 1;
    jsonData->v_r_req[0][1] = 1;
    jsonData->v_r_req[0][2] = 1;
    jsonData->v_r_req[0][3] = 4096*4096;

    jsonData->o_rc_req[0][0] = 1;
    jsonData->o_rc_req[0][1] = 1;
    jsonData->o_rc_req[0][2] = 32;
    jsonData->o_rc_req[0][3] = 4096*4096/32;

    jsonData->o_r_req[0][0] = 1;
    jsonData->o_r_req[0][1] = 1;
    jsonData->o_r_req[0][2] = 1;
    jsonData->o_r_req[0][3] = 4096*4096;

    jsonData->g_rc_req[0][0] = 1;
    jsonData->g_rc_req[0][1] = 1;
    jsonData->g_rc_req[0][2] = 1;
    jsonData->g_rc_req[0][3] = 4096*16384;

    jsonData->g_r_req[0][0] = 1;
    jsonData->g_r_req[0][1] = 1;
    jsonData->g_r_req[0][2] = 1;
    jsonData->g_r_req[0][3] = 4096*16384;

    jsonData->u_rc_req[0][0] = 1;
    jsonData->u_rc_req[0][1] = 1;
    jsonData->u_rc_req[0][2] = 1;
    jsonData->u_rc_req[0][3] = 4096*16384;

    jsonData->u_r_req[0][0] = 1;
    jsonData->u_r_req[0][1] = 1;
    jsonData->u_r_req[0][2] = 1;
    jsonData->u_r_req[0][3] = 4096*16384;

    jsonData->output_rc_req[0][0] = 1;
    jsonData->output_rc_req[0][1] = 1;
    jsonData->output_rc_req[0][2] = 1;
    jsonData->output_rc_req[0][3] = 4096*4096;

    jsonData->output_r_req[0][0] = 1;
    jsonData->output_r_req[0][1] = 1;
    jsonData->output_r_req[0][2] = 1;
    jsonData->output_r_req[0][3] = 4096*4096;


    jsonData->q_rc_req_arrays = 1;
    jsonData->q_r_req_arrays = 1;
    jsonData->k_rc_req_arrays = 1;
    jsonData->k_r_req_arrays = 1;
    jsonData->v_rc_req_arrays = 1;
    jsonData->v_r_req_arrays = 1;
    jsonData->o_rc_req_arrays = 1;
    jsonData->o_r_req_arrays = 1;
    jsonData->g_rc_req_arrays = 1;
    jsonData->g_r_req_arrays = 1;
    jsonData->u_rc_req_arrays = 1;
    jsonData->u_r_req_arrays = 1;
    jsonData->output_rc_req_arrays = 1;
    jsonData->output_r_req_arrays = 1;
    /*
    // 读取 JSON 文件
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error opening file");
        free(jsonData);
        return NULL;
    }
    
    // 获取文件大小
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // 分配足够大的缓冲区来存储文件内容
    char *json_buffer = (char *)malloc(file_size + 1);
    if (json_buffer == NULL) {
        perror("Memory allocation failed");
        fclose(fp);
        free(jsonData);
        return NULL;
    }

    // 读取文件内容到缓冲区
    fread(json_buffer, 1, file_size, fp);
    fclose(fp);

    // 添加字符串结束符
    json_buffer[file_size] = '\0';

    // 解析 JSON 数据
    cJSON *root = cJSON_Parse(json_buffer);
    if (root == NULL) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
        cJSON_Delete(root);
        free(json_buffer);
        free(jsonData);
        return NULL;
    }

    // 从 JSON 对象中提取需要的信息并存储在结构体中
      
    
    cJSON *q_0 = cJSON_GetObjectItemCaseSensitive(root, "q_0");
    if (cJSON_HasObjectItem(q_0, "latency")) {
        jsonData->q_0_latency = cJSON_GetObjectItemCaseSensitive(q_0, "latency")->valuedouble;
    }

    cJSON *q_1 = cJSON_GetObjectItemCaseSensitive(root, "q_1");
    if (cJSON_HasObjectItem(q_1, "latency")) {
        jsonData->q_1_latency = cJSON_GetObjectItemCaseSensitive(q_1, "latency")->valuedouble;
    }

    cJSON *k_0 = cJSON_GetObjectItemCaseSensitive(root, "k_0");
    if (cJSON_HasObjectItem(k_0, "latency")) {
        jsonData->k_0_latency = cJSON_GetObjectItemCaseSensitive(k_0, "latency")->valuedouble;
    }

    cJSON *k_1 = cJSON_GetObjectItemCaseSensitive(root, "k_1");
    if (cJSON_HasObjectItem(k_1, "latency")) {
        jsonData->k_1_latency = cJSON_GetObjectItemCaseSensitive(k_1, "latency")->valuedouble;
    }

    cJSON *p = cJSON_GetObjectItemCaseSensitive(root, "p");
    if (cJSON_HasObjectItem(p, "latency")) {
        jsonData->p_latency = cJSON_GetObjectItemCaseSensitive(p, "latency")->valuedouble;
    }

    cJSON *s = cJSON_GetObjectItemCaseSensitive(root, "s");
    if (cJSON_HasObjectItem(s, "latency")) {
        jsonData->s_latency = cJSON_GetObjectItemCaseSensitive(s, "latency")->valuedouble;
    }

    cJSON *v_0 = cJSON_GetObjectItemCaseSensitive(root, "v_0");
    if (cJSON_HasObjectItem(v_0, "latency")) {
        jsonData->v_0_latency = cJSON_GetObjectItemCaseSensitive(v_0, "latency")->valuedouble;
    }

    cJSON *v_1 = cJSON_GetObjectItemCaseSensitive(root, "v_1");
    if (cJSON_HasObjectItem(v_1, "latency")) {
        jsonData->v_1_latency = cJSON_GetObjectItemCaseSensitive(v_1, "latency")->valuedouble;
    }

    cJSON *a = cJSON_GetObjectItemCaseSensitive(root, "a");
    if (cJSON_HasObjectItem(a, "latency")) {
        jsonData->a_latency = cJSON_GetObjectItemCaseSensitive(a, "latency")->valuedouble;
    }

    cJSON *o_0 = cJSON_GetObjectItemCaseSensitive(root, "o_0");
    if (cJSON_HasObjectItem(o_0, "latency")) {
        jsonData->o_0_latency = cJSON_GetObjectItemCaseSensitive(o_0, "latency")->valuedouble;
    }

    cJSON *o_1 = cJSON_GetObjectItemCaseSensitive(root, "o_1");
    if (cJSON_HasObjectItem(o_1, "latency")) {
        jsonData->o_1_latency = cJSON_GetObjectItemCaseSensitive(o_1, "latency")->valuedouble;
    }

    cJSON *g_0 = cJSON_GetObjectItemCaseSensitive(root, "g_0");
    if (cJSON_HasObjectItem(g_0, "latency")) {
        jsonData->g_0_latency = cJSON_GetObjectItemCaseSensitive(g_0, "latency")->valuedouble;
    }

    cJSON *g_1 = cJSON_GetObjectItemCaseSensitive(root, "g_1");
    if (cJSON_HasObjectItem(g_1, "latency")) {
        jsonData->g_1_latency = cJSON_GetObjectItemCaseSensitive(g_1, "latency")->valuedouble;
    }

    cJSON *u_0 = cJSON_GetObjectItemCaseSensitive(root, "u_0");
    if (cJSON_HasObjectItem(u_0, "latency")) {
        jsonData->u_0_latency = cJSON_GetObjectItemCaseSensitive(u_0, "latency")->valuedouble;
    }

    cJSON *u_1 = cJSON_GetObjectItemCaseSensitive(root, "u_1");
    if (cJSON_HasObjectItem(u_1, "latency")) {
        jsonData->u_1_latency = cJSON_GetObjectItemCaseSensitive(u_1, "latency")->valuedouble;
    }

    cJSON *output_core = cJSON_GetObjectItemCaseSensitive(root, "output");
    if (cJSON_HasObjectItem(output_core, "latency")) {
        jsonData->output_latency = cJSON_GetObjectItemCaseSensitive(output_core, "latency")->valuedouble;
    }

    cJSON *q = cJSON_GetObjectItemCaseSensitive(root, "q");
    if (cJSON_IsObject(q)) {
        cJSON *q_rc_req = cJSON_GetObjectItemCaseSensitive(q, "rc_req");
        if (cJSON_IsArray(q_rc_req)) {
            int size = cJSON_GetArraySize(q_rc_req);
            jsonData->q_rc_req_arrays = size;
            jsonData->q_rc_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(q_rc_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->q_rc_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->q_rc_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }

        cJSON *q_r_req = cJSON_GetObjectItemCaseSensitive(q, "r_req");
        if (cJSON_IsArray(q_r_req)) {
            int size = cJSON_GetArraySize(q_r_req);
            jsonData->q_r_req_arrays = size;
            jsonData->q_r_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(q_r_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->q_r_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->q_r_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }
    }

    cJSON *k = cJSON_GetObjectItemCaseSensitive(root, "k");
    if (cJSON_IsObject(k)) {
        cJSON *k_rc_req = cJSON_GetObjectItemCaseSensitive(k, "rc_req");
        if (cJSON_IsArray(k_rc_req)) {
            int size = cJSON_GetArraySize(k_rc_req);
            jsonData->k_rc_req_arrays = size;
            jsonData->k_rc_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(k_rc_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->k_rc_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->k_rc_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }

        cJSON *k_r_req = cJSON_GetObjectItemCaseSensitive(k, "r_req");
        if (cJSON_IsArray(k_r_req)) {
            int size = cJSON_GetArraySize(k_r_req);
            jsonData->k_r_req_arrays = size;
            jsonData->k_r_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(k_r_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->k_r_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->k_r_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }
    }

    cJSON *v = cJSON_GetObjectItemCaseSensitive(root, "v");
    if (cJSON_IsObject(v)) {
        cJSON *v_rc_req = cJSON_GetObjectItemCaseSensitive(v, "rc_req");
        if (cJSON_IsArray(v_rc_req)) {
            int size = cJSON_GetArraySize(v_rc_req);
            jsonData->v_rc_req_arrays = size;
            jsonData->v_rc_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(v_rc_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->v_rc_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->v_rc_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }

        cJSON *v_r_req = cJSON_GetObjectItemCaseSensitive(v, "r_req");
        if (cJSON_IsArray(v_r_req)) {
            int size = cJSON_GetArraySize(v_r_req);
            jsonData->v_r_req_arrays = size;
            jsonData->v_r_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(v_r_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->v_r_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->v_r_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }
    }

    cJSON *o = cJSON_GetObjectItemCaseSensitive(root, "o");
    if (cJSON_IsObject(o)) {
        cJSON *o_rc_req = cJSON_GetObjectItemCaseSensitive(o, "rc_req");
        if (cJSON_IsArray(o_rc_req)) {
            int size = cJSON_GetArraySize(o_rc_req);
            jsonData->o_rc_req_arrays = size;
            jsonData->o_rc_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(o_rc_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->o_rc_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->o_rc_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }

        cJSON *o_r_req = cJSON_GetObjectItemCaseSensitive(o, "r_req");
        if (cJSON_IsArray(o_r_req)) {
            int size = cJSON_GetArraySize(o_r_req);
            jsonData->o_r_req_arrays = size;
            jsonData->o_r_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(o_r_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->o_r_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->o_r_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }
    }

    cJSON *g = cJSON_GetObjectItemCaseSensitive(root, "g");
    if (cJSON_IsObject(g)) {
        cJSON *g_rc_req = cJSON_GetObjectItemCaseSensitive(g, "rc_req");
        if (cJSON_IsArray(g_rc_req)) {
            int size = cJSON_GetArraySize(g_rc_req);
            jsonData->g_rc_req_arrays = size;
            jsonData->g_rc_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(g_rc_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->g_rc_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->g_rc_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }

        cJSON *g_r_req = cJSON_GetObjectItemCaseSensitive(g, "r_req");
        if (cJSON_IsArray(g_r_req)) {
            int size = cJSON_GetArraySize(g_r_req);
            jsonData->g_r_req_arrays = size;
            jsonData->g_r_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(g_r_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->g_r_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->g_r_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }
    }

    cJSON *u = cJSON_GetObjectItemCaseSensitive(root, "u");
    if (cJSON_IsObject(u)) {
        cJSON *u_rc_req = cJSON_GetObjectItemCaseSensitive(u, "rc_req");
        if (cJSON_IsArray(u_rc_req)) {
            int size = cJSON_GetArraySize(u_rc_req);
            jsonData->u_rc_req_arrays = size;
            jsonData->u_rc_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(u_rc_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->u_rc_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->u_rc_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }

        cJSON *u_r_req = cJSON_GetObjectItemCaseSensitive(u, "r_req");
        if (cJSON_IsArray(u_r_req)) {
            int size = cJSON_GetArraySize(u_r_req);
            jsonData->u_r_req_arrays = size;
            jsonData->u_r_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(u_r_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->u_r_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->u_r_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }
    }

    cJSON *output = cJSON_GetObjectItemCaseSensitive(root, "output");
    if (cJSON_IsObject(output)) {
        cJSON *output_rc_req = cJSON_GetObjectItemCaseSensitive(output, "rc_req");
        if (cJSON_IsArray(output_rc_req)) {
            int size = cJSON_GetArraySize(output_rc_req);
            jsonData->output_rc_req_arrays = size;
            jsonData->output_rc_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(output_rc_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->output_rc_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->output_rc_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }

        cJSON *output_r_req = cJSON_GetObjectItemCaseSensitive(output, "r_req");
        if (cJSON_IsArray(output_r_req)) {
            int size = cJSON_GetArraySize(output_r_req);
            jsonData->output_r_req_arrays = size;
            jsonData->output_r_req = (unsigned int **)malloc(size * sizeof(unsigned int *));
            for (int i = 0; i < size; ++i) {
                cJSON *item = cJSON_GetArrayItem(output_r_req, i);
                int subsize = cJSON_GetArraySize(item);
                jsonData->output_r_req[i] = (unsigned int *)malloc(subsize * sizeof(unsigned int));
                for (int j = 0; j < subsize; ++j) {
                    jsonData->output_r_req[i][j] = cJSON_GetArrayItem(item, j)->valueint;
                }
            }
        }
    }
*/    

}   