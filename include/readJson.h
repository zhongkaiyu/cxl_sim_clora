#ifndef READ_JSON_H
#define READ_JSON_H

// 定义 JsonData 结构体
typedef struct {
    
    //npu request latency (Core)
    //0为原本计算，1为带delta的lora计算
    double q_0_latency;
    double q_1_latency;
    double k_0_latency;
    double k_1_latency;
    double p_latency;
    double s_latency;
    double v_0_latency;
    double v_1_latency;
    double a_latency;
    double o_0_latency;
    double o_1_latency;
    double g_0_latency;
    double g_1_latency;
    double u_0_latency;
    double u_1_latency;
    double output_latency;

    double load_k_latency;
    double load_v_latency;

    double tmp_lora_latency;//在py未接入的情况先tmp

    double b1,b2,b3;//三种方案分别所占的比例

    unsigned int input_token_num;//输入词数量

    //ssd request 参数
    unsigned int q_rc_req[5][10];
    unsigned int q_r_req[5][10];
    unsigned int k_rc_req[5][10];
    unsigned int k_r_req[5][10];
    unsigned int v_rc_req[5][10];
    unsigned int v_r_req[5][10];
    unsigned int o_rc_req[5][10];
    unsigned int o_r_req[5][10];
    unsigned int g_rc_req[5][10];
    unsigned int g_r_req[5][10];
    unsigned int u_rc_req[5][10];
    unsigned int u_r_req[5][10];
    unsigned int output_rc_req[5][10];
    unsigned int output_r_req[5][10];

    unsigned int q_rc_req_arrays;
    unsigned int q_r_req_arrays;
    unsigned int k_rc_req_arrays;
    unsigned int k_r_req_arrays;
    unsigned int v_rc_req_arrays;
    unsigned int v_r_req_arrays;
    unsigned int o_rc_req_arrays;
    unsigned int o_r_req_arrays;
    unsigned int g_rc_req_arrays;
    unsigned int g_r_req_arrays;
    unsigned int u_rc_req_arrays;
    unsigned int u_r_req_arrays;
    unsigned int output_rc_req_arrays;
    unsigned int output_r_req_arrays;

} JsonData;

// 函数声明用于读取和解析 JSON 数据
JsonData* readJsonData(const char *filename);

#endif /* READ_JSON_H */
