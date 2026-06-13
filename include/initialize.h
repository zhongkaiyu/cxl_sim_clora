/*****************************************************************************************************************************
  This project was supported by the National Basic Research 973 Program of China under Grant No.2011CB302301
  Huazhong University of Science and Technology (HUST)   Wuhan National Laboratory for Optoelectronics

  FileName： initialize.h
Author: Hu Yang		Version: 2.1	Date:2011/12/02
Description: 

History:
<contributor>     <time>        <version>       <desc>                   <e-mail>
Yang Hu	        2009/09/25	      1.0		    Creat SSDsim       yanghu@foxmail.com
2010/05/01        2.x           Change 
Zhiming Zhu     2011/07/01        2.0           Change               812839842@qq.com
Shuangwu Zhang  2011/11/01        2.1           Change               820876427@qq.com
Chao Ren        2011/07/01        2.0           Change               529517386@qq.com
Hao Luo         2011/01/01        2.0           Change               luohao135680@gmail.com
 *****************************************************************************************************************************/
#ifndef INITIALIZE_H
#define INITIALIZE_H 10000

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include <sys/types.h>
#include "../lib/avlTree/avlTree.h"

#define SECTOR 512
#define BUFSIZE 200

#define READ 1
#define WRITE 0
#define READ_COMPUTE 2

#define NPU_COMPUTE 11
#define NPU_READ_DRAM 12
#define NPU_SF_COMPUTE 13

// "CHANNEL"指的是GPU和CXL之间的通道。
// 通道是一种用于传输数据的通信通道，用于连接不同的设备或系统。
#define CHANNEL_IDLE 000 // CHANNEL_IDLE 表示通道处于空闲状态，简单来说CHANNEL只有两种状态：空闲和忙碌
#define CHANNEL_CA_TRANSFER 3
#define CHANNEL_GC 4
#define CHANNEL_DATA_TRANSFER 5
#define CHANNEL_CA_AND_DATA_TRANSFER 8 // 只有WRITE操作会有这个状态，也可以写为CHANNEL_CA_TRANSFER(3)+CHANNEL_DATA_TRANSFER(5)=CHANNEL_DATA_AND_CA_TRANSFER(8)

// "CHIP"指的是"Flash Chip"（闪存芯片）。
// 闪存芯片是一种用于存储数据的半导体存储器，具有非易失性，可以在断电后保留数据。
#define CHIP_IDLE 100
#define CHIP_WRITE_BUSY 101
#define CHIP_READ_BUSY 102
#define CHIP_CA_TRANSFER 103
#define CHIP_DATA_TRANSFER 107
#define CHIP_WAIT 108

// “PE”是指“Processing Element”（处理单元）。
// 在硬件架构和计算系统中，PE（处理单元）是一种独立的计算资源，负责执行特定的计算任务或操作。不同的PE可能具有不同的功能，
#define PE_IDLE 120 // PE_IDLE 表示某个处理单元处于空闲状态，
#define PE_CA_TRANSFER 121
#define PE_READ_BUSY 122 // PE_READ_BUSY 表示该处理单元正在执行读取操作并处于忙碌状态，
#define PE_COMPUTE 123 // PE_COMPUTE 表示该处理单元正在进行计算操作。
#define PE_DATA_TRANSFER 124

// "CC"指的是"CXL Controller"（CXL控制器），是一种用于控制CXL MEM操作的硬件设备。
// CXL Controller负责调度Dram、DramChannel等硬件资源，以便执行读取、写入和计算操作。
#define CC_IDLE 300 // CC_R_IDLE 表示CXL Controller处于空闲状态
#define CC_ANALYZE 301 // CC_R_CA_TRANSFER 表示CXL Controller正在进行命令地址解析
#define CC_CA_TRANSFER 302 // CC_R_CA_TRANSFER 表示CXL Controller正在进行从CC到DRAM的命令地址传输
#define CC_DATA_TRANSFER 303 // CC_R_DATA_TRANSFER 表示CXL Controller正在进行从DRAM到CC的数据传输
#define CC_WRITE 304 // CC_W_TRANSFER 表示CXL Controller正在进行向DRAM写入数据
#define CC_CONFIRM 305 // CC_W_CONFIRM 表示CXL Controller最后确认写入操作

#define DRAM_IDLE 500
#define DRAM_CA_TRANSFER 501
#define DRAM_READ 502
#define DRAM_WRITE 503


// "SR"指的是"Sub Request"（子请求）。
// 子请求是一个更大的请求的一部分，包含读取或写入数据的指令。
#define SR_WAIT 200 // SR_WAIT 表示子请求正在等待
// ------ READ ------
#define SR_CHANNEL_R_CA_TRANSFER 201 // SR_CHANNEL_R_CA_TRANSFER 表示子请求正在进行channel中进行命令地址传输
#define SR_CXLCTRL_R_CA_ANALYZE 202 // SR_CXLCTRL_R_CA_TRANSFER 表示CXL Controller正在进行命令地址解析
#define SR_CXLCTRL_R_CA_TRANSFER 203 // SR_CXLCTRL_R_CA_TRANSFER 表示CXL Controller正在进行从CC到DRAM的命令地址传输
#define SR_CXLCTRL_R_READ 204 // SR_CXLCTRL_R_READ 表示CXL Controller正在进行从DRAM到CC的数据传输
#define SR_CHANNEL_R_DATA_TRANSFER 205 // SR_CHANNEL_R_DATA_TRANSFER 表示子请求正在进行channel中进行数据传输
// ------ WRITE ------
#define SR_CHANNEL_W_TRANSFER 210 // SR_CHANNEL_W_TRANSFER 表示子请求正在进行channel中进行命令地址和数据的打包传输
#define SR_CXLCTRL_W_CA_ANALYZE 211 // SR_CXLCTRL_W_CA_ANALYZE 表示CXL Controller正在进行命令地址解析
#define SR_CXLCTRL_W_WRITE 212 // SR_CXLCTRL_W_TRANSFER 表示CXL Controller正在进行向DRAM写入数据
#define SR_CXLCTRL_W_CONFIRM 213 // SR_CXLCTRL_W_CONFIRM 表示CXL Controller最后确认写入操作
// ------ READ_COMPUTE ------
#define SR_CHANNEL_RC_CA_TRANSFER 220 // SR_CHANNEL_RC_CA_TRANSFER 表示子请求正在进行channel中进行命令地址传输
#define SR_CXLCTRL_RC_CA_ANALYZE 221 // CC_RC_CA_ANALYZE 表示CXL Controller正在进行命令地址解析
#define SR_CXLCTRL_RC_CA_TRANSFER 222 // SR_CXLCTRL_RC_CA_TRANSFER 表示CXL Controller正在进行从CC到DRAM的命令地址传输
#define SR_CXLCTRL_RC_READ 223 // DRAM或连接的近存计算引擎在接收到计算指令后，首先根据地址信息读取需要的数据。
#define SR_CXLCTRL_RC_COMPUTE 224 // 接着，内存控制器或专用的计算引擎直接在DRAM中执行指定的计算操作。
#define SR_CXLCTRL_RC_DATA_TRANSFER 225 // SR_CXLCTRL_RC_DATA_TRANSFER 表示CXL Controller正在进行从DRAM到CC的数据传输
#define SR_CHANNEL_RC_DATA_TRANSFER 226 // SR_CHANNEL_RC_DATA_TRANSFER 表示子请求正在进行channel中进行数据传输
// ------ COMPLETE ------
#define SR_COMPLETE 299



// "GC" 指的是 "Garbage Collection"（垃圾回收）。
// 垃圾回收是管理内存的机制，用于自动回收不再使用的内存资源，以防止内存泄漏。
#define GC_WAIT 400
#define GC_UNINTERRUPT 1

#define PG_SUB 0xffffffff			

/*****************************************
 *函数结果状态代码
 *Status 是函数类型，其值是函数结果状态代码
 ******************************************/
#define SUCCESS		1
#define FAILURE		0
#define ERROR		(-1)
typedef int Status;

struct Set{
    unsigned int *array;        // 动态数组存储集合元素
    size_t used;                // 动态数组已使用的大小
    size_t capacity;            // 动态数组当前分配的总容量
};

struct npu_info{
    unsigned int req_idx;               // npu下一个应该发出的req的id
    unsigned int step;                  // npu当前处在的步骤
    unsigned int step_issue[100];       // npu每一个step的全部指令是否已经issue

    struct npu_request *npu_reqs_head;
    struct npu_request *npu_reqs_tail;
    unsigned int npu_reqs_queue_length;

    struct Set *step_wait_reqs[30];     //20炸于20，20+炸于18
    struct Set *finished_reqs;

    unsigned int npu_finish;
};

struct addr_info {
    unsigned int gpu_id;
    unsigned int cxl_id;
    unsigned int size;
    unsigned int input_size;    // B
    unsigned int output_size;   // B
};

struct user_args{
    char parameter_filename[80];
    char trace_filename[80];
    char simulation_timestamp[16];
    int num_disk;

    int is_gcsync;
    int is_gclock;
    int is_gcdefer;
    int diskid;
    int64_t gc_time_window;
};

struct ac_time_characteristics{
    int tPROG;     //program time
    int tDBSY;     //bummy busy time for two-plane program
    int tBERS;     //block erase time
    int tCLS;      //CLE setup time
    int tCLH;      //CLE hold time
    int tCS;       //CE setup time
    int tCH;       //CE hold time
    int tWP;       //WE pulse width
    int tALS;      //ALE setup time
    int tALH;      //ALE hold time
    int tDS;       //data setup time
    int tDH;       //data hold time
    int tWC;       //write cycle time
    int tWH;       //WE high hold time
    int tADL;      //address to data loading time
    int tR;        //data transfer from cell to register
    int tAR;       //ALE to RE delay
    int tCLR;      //CLE to RE delay
    int tRR;       //ready to RE low
    int tRP;       //RE pulse width
    int tWB;       //WE high to busy
    int tRC;       //read cycle time
    int tREA;      //RE access time
    int tCEA;      //CE access time
    int tRHZ;      //RE high to output hi-z
    int tCHZ;      //CE high to output hi-z
    int tRHOH;     //RE high to output hold
    int tRLOH;     //RE low to output hold
    int tCOH;      //CE high to output hold
    int tREH;      //RE high to output time
    int tIR;       //output hi-z to RE low
    int tRHW;      //RE high to WE low
    int tWHR;      //WE high to RE low
    int tRST;      //device resetting time

    int tCMDCXL;       //command input setup time
    int tCMDDRAM;       //command hold time
    int tANALYZE;       //data input setup time
    int tDRAMRL;         // dram read latency
    int tDRAMWL;         // dram write latency

    /* Reviewer-requested overhead terms. Defaults are 0 so existing runs are
     * unaffected; set in config/parameters.conf to enable. All values in ns. */
    int L_CXL_switch;        /* fixed latency per CXL switch traversal (one-way) */
    int L_read_compute_cmd;  /* fixed NDP-side command setup/dispatch overhead */
}ac_timing;

struct model_characteristics{
    int model_dim;
    int ffn_dim;
    int input_length;
    int data_type;
}model_par;

struct cxl_switch_device_info{
    int is_gcsync;
    int is_gclock;
    int is_gcdefer;
    int ndisk;
    int diskid;
    int64_t gc_time_window;
    struct gclock_raid_info *gclock_pointer;

    int64_t simulation_start_time;
    int64_t simulation_end_time;

    int64_t current_time;                //记录系统时间
    unsigned int page;

    unsigned int gc_request;             //记录在SSD中，当前时刻有多少gc操作的请求

    unsigned int write_request_count;    //记录写操作的次数
    unsigned int read_request_count;     //记录读操作的次数
    int64_t write_avg;                   //记录用于计算写请求平均响应时间的时间
    int64_t read_avg;                    //记录用于计算读请求平均响应时间的时间

    unsigned int write_request_size;    // total write size in B
    unsigned int read_request_size;     // total read size in B
    unsigned int read_request_c_a_size;
    unsigned int rc_request_result_size; // total result data of rc request transfered through channels
    unsigned int rc_request_input_size; // total input data of rc request transfered through channels 
    unsigned int rc_request_c_a_size;

    unsigned int rc_request_compute_ops;           // ops

    unsigned int in_program_size;       // total internal write (program) size in bytes
    unsigned int in_read_size;          // total internal read size in bytes
    unsigned int write_subreq_count;
    unsigned int read_subreq_count;

    unsigned int min_lsn;
    unsigned int max_lsn;
    unsigned long read_count;
    unsigned long program_count;

    unsigned long write_flash_count;     //实际产生的对flash的写操作 | The actual write to flash
    float ave_read_size;
    float ave_write_size;
    unsigned int request_queue_length;
    unsigned int update_read_count;      //记录因为更新操作导致的额外读出操作 | Record additional read operations due to update operations

    char parameterfilename[80];
    char tracefilename[80];
    char outputfilename[80];
    char statisticfilename[80];
    char statisticfilename2[80];
    char outfile_gc_name[80];
    char outfile_io_name[80];
    char outfile_io_write_name[80];
    char outfile_io_read_name[80];

    FILE * outputfile;
    FILE * tracefile;
    FILE * statisticfile;
    FILE * statisticfile2;
    FILE * outfile_gc;
    FILE * outfile_io;
    FILE * outfile_io_write;
    FILE * outfile_io_read;

    struct npu_info *npu;
    struct parameter_value *parameter;   //SSD参数因子
    struct dram_info *dram;                // 不用管
    struct request *request_queue;       //dynamic request queue
    struct request *request_tail;	     // the tail of the request queue
    struct sub_request *subs_w_head;     // 不用管
    struct sub_request *subs_w_tail;

    struct channel_info *top_channel_head;   //指向GPU侧channel结构体数组的首地址
    struct channel_info *bottom_channel_head;   //指向CXL侧channel结构体数组的首地址
};

struct channel_info{
    int type;                            //表示该通道是GPU侧还是CXL侧还是写，1表示GPU侧，0表示CXL侧
    unsigned long program_count;
    unsigned int subs_idx;             // 给每个sub reqeust赋予一个idx，逐个增加

    int current_state;                   //channel has serveral states, including idle, command/address transfer,data transfer,unknown
    int next_state;
    int64_t current_time;                //记录该通道的当前时间
    int64_t next_state_predict_time;     //the predict time of next state, used to decide the sate at the moment

    struct sub_request *subs_r_head;     //channel上的读请求队列头，先服务处于队列头的子请求
    struct sub_request *subs_r_tail;     //channel上的读请求队列尾，新加进来的子请求加到队尾
    struct sub_request *subs_w_head;     //channel上的写请求队列头，先服务处于队列头的子请求
    struct sub_request *subs_w_tail;     //channel上的写请求队列，新加进来的子请求加到队尾
    struct sub_request *subs_rc_head;    //channel上的读计算请求队列头，先服务处于队列头的子请求
    struct sub_request *subs_rc_tail;    //channel上的读计算队列，新加进来的子请求加到队尾
    struct gc_operation *gc_command;     //记录需要产生gc的位置

    struct cxl_device_info *cxl_device; //每个channel上连接有一个cxl device
};

struct cxl_device_info{
    int *current_states;
    int *next_states;
    int64_t current_time;
    int64_t next_state_predict_time;

    struct chip_info{
        unsigned int die_num;               //表示一个颗粒中有多少个die
        unsigned int plane_num_die;         //indicate how many planes in a die
        unsigned int block_num_plane;       //indicate how many blocks in a plane
        unsigned int page_num_block;        //indicate how many pages in a block
        unsigned int subpage_num_page;      //indicate how many subpage in a page
        unsigned int ers_limit;             //该chip中每块能够被擦除的次数
        unsigned int token;                 //在动态分配中，为防止每次分配在第一个die需要维持一个令牌，每次从令牌所指的位置开始分配
        unsigned int subs_idx;              //正在执行的sub request编号

        int current_state;                  //chip has serveral states, including idle, command/address transfer,data transfer,unknown
        int next_state;
        int64_t current_time;               //记录该通道的当前时间
        int64_t next_state_predict_time;    //the predict time of next state, used to decide the sate at the moment

        unsigned long read_count;           //how many read count in the process of workload
        unsigned long program_count;
        unsigned long erase_count;

        struct ac_time_characteristics ac_timing;

        struct die_info{
            unsigned int token;                 //在动态分配中，为防止每次分配在第一个plane需要维持一个令牌，每次从令牌所指的位置开始分配
            struct plane_info{
                int add_reg_ppn;                    //read，write时把地址传送到该变量，该变量代表地址寄存器。die由busy变为idle时，清除地址 //有可能因为一对多的映射，在一个读请求时，有多个相同的lpn，所以需要用ppn来区分
                unsigned int free_page;             //该plane中有多少free page
                unsigned int ers_invalid;           //记录该plane中擦除失效的块数
                unsigned int active_block;          //if a die has a active block, 该项表示其物理块号
                int can_erase_block;                //记录在一个plane中准备在gc操作中被擦除操作的块,-1表示还没有找到合适的块
                struct direct_erase *erase_node;    //用来记录可以直接删除的块号,在获取新的ppn时，每当出现invalid_page_num==64时，将其添加到这个指针上，供GC操作时直接删除
                struct blk_info *blk_head;
                struct pe_info *pe;
            } *plane_head;
        } *die_head;

        struct pe_info{
            int current_state;                  //pe has serveral states, including idle, command/address transfer,data transfer,unknown
            int next_state;
            int64_t current_time;               //记录该通道的当前时间
            int64_t next_state_predict_time;    //the predict time of next state, used to decide the sate at the moment
            unsigned int subs_idx;              //正在执行的sub request编号
        } *pe;

        struct cxlctrl_info{
            int current_state;                  
            int next_state;
            int64_t current_time;               
            int64_t next_state_predict_time;    //the predict time of next state, used to decide the sate at the moment
            int buf_size;
            int buf_used_size;
            // unsigned int subs_idx;              //正在执行的sub request编号
        } *cxlctrl;

        struct cxldram_info{
            int current_state;                  
            int next_state;
            int64_t current_time;               
            int64_t next_state_predict_time;    //the predict time of next state, used to decide the sate at the moment
            // unsigned int subs_idx;              //正在执行的sub request编号
        } *cxldram;

    } *chip_head;       // [Important] 指向chip结构体数组的首地址

    unsigned int chip_num;       //记录一个CXL中有多少个颗粒
};

struct blk_info{
    unsigned int erase_count;          //块的擦除次数，该项记录在ram中，用于GC
    unsigned int free_page_num;        //记录该块中的free页个数，同上
    unsigned int invalid_page_num;     //记录该块中失效页的个数，同上
    int last_write_page;               //记录最近一次写操作执行的页数,-1表示该块没有一页被写过
    struct page_info *page_head;       //记录每一子页的状态
};


struct page_info{                      //lpn记录该物理页存储的逻辑页，当该逻辑页有效时，valid_state大于0，free_state大于0；
    int valid_state;                   //indicate the page is valid or invalid
    int free_state;                    //each bit indicates the subpage is free or occupted. 1 indicates that the bit is free and 0 indicates that the bit is used
    unsigned int lpn;                 
    unsigned int written_count;        //记录该页被写的次数
};


struct dram_info{
    unsigned int dram_capacity;     
    int64_t current_time;

    struct dram_parameter *dram_paramters;      
    struct map_info *map;
    struct buffer_info *buffer; 

};


/*********************************************************************************************
 *buffer中的已写回的页的管理方法:在buffer_info中维持一个队列:written。这个队列有队首，队尾。
 *每次buffer management中，请求命中时，这个group要移到LRU的队首，同时看这个group中是否有已
 *写回的lsn，有的话，需要将这个group同时移动到written队列的队尾。这个队列的增长和减少的方法
 *如下:当需要通过删除已写回的lsn为新的写请求腾出空间时，在written队首中找出可以删除的lsn。
 *当需要增加新的写回lsn时，找到可以写回的页，将这个group加到指针written_insert所指队列written
 *节点前。我们需要再维持一个指针，在buffer的LRU队列中指向最老的一个写回了的页，下次要再写回时，
 *只需由这个指针回退到前一个group写回即可。
 **********************************************************************************************/
typedef struct buffer_group{
    TREE_NODE node;                     //树节点的结构一定要放在用户自定义结构的最前面，注意!
    struct buffer_group *LRU_link_next;	// next node in LRU list
    struct buffer_group *LRU_link_pre;	// previous node in LRU list

    unsigned int group;                 //the first data logic sector number of a group stored in buffer 
    unsigned int stored;                //indicate the sector is stored in buffer or not. 1 indicates the sector is stored and 0 indicate the sector isn't stored.EX.  00110011 indicates the first, second, fifth, sixth sector is stored in buffer.
    unsigned int dirty_clean;           //it is flag of the data has been modified, one bit indicates one subpage. EX. 0001 indicates the first subpage is dirty
    int flag;			                //indicates if this node is the last 20% of the LRU list	
}buf_node;


struct dram_parameter{
    float active_current;
    float sleep_current;
    float voltage;
    int clock_time;
};


struct map_info{
    struct entry *map_entry;            //该项是映射表结构体指针,each entry indicate a mapping information
    struct buffer_info *attach_info;	// info about attach map
};

struct npu_request{
    unsigned int type;
    unsigned int begin_time;
    unsigned int end_time;
    unsigned int idx;

    struct npu_request *next_node;
};

struct request{
    int64_t time;                      //请求到达的时间，单位为us,这里和通常的习惯不一样，通常的是ms为单位，这里需要有个单位变换过程
    unsigned int lsn;                  //请求的起始地址，逻辑地址
    unsigned int idx;                  //每个request都有自己独立的编号
    unsigned int size;                 //请求的大小，既多少个扇区
    unsigned int operation;            //请求的种类，1为读，0为写
    struct addr_info *addr;            //请求的地址信息的头指针
    unsigned int addr_num;             //请求的地址个数
    unsigned int output_size;          // for read compute request, the ouputsize of each sub_request/B
    unsigned int input_size;            // for read comptue request, the inputsize of each sub_request/B

    unsigned int* need_distr_flag;
    unsigned int complete_lsn_count;   //record the count of lsn served by buffer

    int distri_flag;		           // indicate whether this request has been distributed already
    int meet_gc_flag;                  // indicate whether this request is blocked by GC process
    int64_t meet_gc_remaining_time;

    int64_t begin_time;
    int64_t response_time;
    double energy_consumption;         //记录该请求的能量消耗，单位为uJ

    struct sub_request *subs;          //链接到属于该请求的所有子请求
    struct request *next_node;         //指向下一个请求结构体
};


struct sub_request{
    unsigned int lpn;                  //这里表示该子请求的逻辑页号
    unsigned int ppn;                  //分配那个物理子页给这个子请求。在multi_chip_page_mapping中，产生子页请求时可能就知道psn的值，其他时候psn的值由page_map_read,page_map_write等FTL最底层函数产生。 
    unsigned int operation;            //表示该子请求的类型，除了读1 写0，还有擦除，two plane等操作 
    int size;
    unsigned int idx;                  //每个sub request 都有一个idx，方便定位
    struct addr_info ** p_addr;           //请求的地址信息的头指针
    unsigned int addr_num;             //请求的地址个数
    unsigned int output_size;          // for read compute request, the ouputsize of each sub_request/B
    unsigned int input_size;            // for read comptue request, the inputsize of each sub_request/B

    unsigned int current_state;        //表示该子请求所处的状态，见宏定义sub request
    int64_t current_time;
    unsigned int next_state;
    int64_t next_state_predict_time;
    unsigned int state;              //使用state的最高位表示该子请求是否是一对多映射关系中的一个，是的话，需要读到buffer中。1表示是一对多，0表示不用写到buffer

    int64_t begin_time;               //子请求开始时间
    int64_t complete_time;            //记录该子请求的处理时间,既真正写入或者读出数据的时间

    struct local *location;           //在静态分配和混合分配方式中，已知lpn就知道该lpn该分配到那个channel，chip，die，plane，这个结构体用来保存计算得到的地址
    struct sub_request *next_subs;    //指向属于同一个request的子请求
    struct sub_request *next_node;    //指向同一个channel中下一个子请求结构体
    struct sub_request *update;       //因为在写操作中存在更新操作，因为在动态分配方式中无法使用copyback操作，需要将原来的页读出后才能进行写操作，所以，将因更新产生的读操作挂在这个指针上
};


/***********************************************************************
 *事件节点控制时间的增长，每次时间的增加是根据时间最近的一个事件来确定的
 ************************************************************************/
struct event_node{
    int type;                        //记录该事件的类型，1表示命令类型，2表示数据传输类型
    int64_t predict_time;            //记录这个时间开始的预计时间，防止提前执行这个时间
    struct event_node *next_node;
    struct event_node *pre_node;
};

struct parameter_value{
    unsigned int chip_num;          //记录一个SSD中有多少个颗粒
    unsigned int dram_capacity;     //记录SSD中DRAM capacity
    unsigned int cpu_sdram;         //记录片内有多少

    unsigned int gpu_channel_number;    //记录SSD中GPU侧有多少个通道，每个通道是单独的bus
    unsigned int cxl_channel_number;    //记录SSD中CXL侧有多少个通道，每个通道是单独的bus
    unsigned int chip_num_per_channel[100];     //设置SSD中每个CXL侧的channel上颗粒的数量

    unsigned int cxl_bandwidth;         //记录CXL侧的通道带宽
    unsigned int dram_bandwidth;        //记录DRAM的带宽

    unsigned int die_chip;    
    unsigned int plane_die;
    unsigned int block_plane;
    unsigned int page_block;
    unsigned int subpage_page;

    unsigned int page_capacity;
    unsigned int subpage_capacity;

    unsigned int chip_computing_power; // ssd每个chip内置pe的总算力，单位GOPS
    unsigned int npu_computing_power; // npu core内提供的算力，单位GOPs

    /* NDP PE timing model selector.  0 (default) = legacy formula,
     * (addr_num-1)*size/elem_size/GOPS, which collapses to ~1 ns for the
     * addr_num==1 sub-requests CLoRA emits.  1 = ops-based: 2 MAC ops per
     * element actually read from device DRAM.  Used by the reviewer
     * sensitivity study on NDP core throughput. */
    int ndp_compute_model;

    unsigned int dram_channel_num; // DRAM channel number of  each CXL device
    unsigned int dram_channel_bandwidth; // bandwidth of each DRAM channel in CXL device

    unsigned int cxlctrl_buf_size; // size of inst buffer in CXL controller
    unsigned int sub_req_inst_size; // size of each sub req inst

    unsigned int ers_limit;         //记录每个块可擦除的次数
    int address_mapping;            //记录映射的类型，1：page；2：block；3：fast
    int wear_leveling;              // WL算法
    int gc;                         //记录gc策略
    int clean_in_background;        //清除操作是否在前台完成
    float overprovide;
    float gc_threshold;             //当达到这个阈值时，开始GC操作，在主动写策略中，开始GC操作后可以临时中断GC操作，服务新到的请求；在普通策略中，GC不可中断

    double operating_current;       //NAND FLASH的工作电流单位是uA
    double supply_voltage;	
    double dram_active_current;     //cpu sdram work current   uA
    double dram_standby_current;    //cpu sdram work current   uA
    double dram_refresh_current;    //cpu sdram work current   uA
    double dram_voltage;            //cpu sdram work voltage  V

    int buffer_management;          //indicates that there are buffer management or not
    int scheduling_algorithm;       //记录使用哪种调度算法，1:FCFS
    float quick_radio;
    int related_mapping;

    unsigned int time_step;
    unsigned int small_large_write; //the threshould of large write, large write do not occupt buffer, which is written back to flash directly

    int striping;                   //表示是否使用了striping方式，0表示没有，1表示有
    int interleaving;
    int pipelining;
    int threshold_fixed_adjust;
    int threshold_value;
    int active_write;               //表示是否执行主动写操作1,yes;0,no
    float gc_hard_threshold;        //普通策略中用不到该参数，只有在主动写策略中，当满足这个阈值时，GC操作不可中断
    int allocation_scheme;          //记录分配方式的选择，0表示动态分配，1表示静态分配
    int static_allocation;          //记录是那种静态分配方式，如ICS09那篇文章所述的所有静态分配方式
    int dynamic_allocation;         //记录动态分配的方式
    int advanced_commands;  
    int ad_priority;                //record the priority between two plane operation and interleave operation
    int ad_priority2;               //record the priority of channel-level, 0 indicates that the priority order of channel-level is highest; 1 indicates the contrary
    int greed_CB_ad;                //0 don't use copyback advanced commands greedily; 1 use copyback advanced commands greedily
    int greed_MPW_ad;               //0 don't use multi-plane write advanced commands greedily; 1 use multi-plane write advanced commands greedily
    int aged;                       //1表示需要将这个SSD变成aged，0表示需要将这个SSD保持non-aged
    float aged_ratio; 
    int queue_length;               //请求队列的长度限制

    struct ac_time_characteristics time_characteristics;
    struct model_characteristics model_characteristics;
};

/********************************************************
 *mapping information,state的最高位表示是否有附加映射关系
 *********************************************************/
struct entry{                       
    unsigned int pn;                //物理号，既可以表示物理页号，也可以表示物理子页号，也可以表示物理块号
    int state;                      //十六进制表示的话是0000-FFFF，每位表示相应的子页是否有效（页映射）。比如在这个页中，0，1号子页有效，2，3无效，这个应该是0x0003.
};


struct local{          
    unsigned int channel;
    unsigned int chip;
    unsigned int die;
    unsigned int plane;
    unsigned int block;
    unsigned int page;
    unsigned int sub_page;
};

struct direct_erase{
    unsigned int block;
    struct direct_erase *next_node;
};


/**************************************************************************************
 *当产生一个GC操作时，将这个结构挂在相应的channel上，等待channel空闲时，发出GC操作命令
 *When a GC operation is generated, the structure is hung on the corresponding channel, and when the channel is idle, the GC operation command is issued.
 ***************************************************************************************/
struct gc_operation{          
    unsigned int chip;
    unsigned int die;
    unsigned int plane;
    unsigned int block;           //该参数只在可中断的gc函数中使用（gc_interrupt），用来记录已近找出来的目标块号 | This parameter is only used in the interruptible gc function (gc_interrupt) to record the target block number that has been found nearby.
    unsigned int page;            //该参数只在可中断的gc函数中使用（gc_interrupt），用来记录已经完成的数据迁移的页号 | This parameter is only used in the interruptible gc function (gc_interrupt), which is used to record the page number of the completed data migration.
    unsigned int state;           //记录当前gc请求的状态 | Record the status of the current gc request
    unsigned int priority;        //记录该gc操作的优先级，1表示不可中断，0表示可中断（软阈值产生的gc请求） | Record the priority of the gc operation, 1 means uninterruptible, 0 means interruptable (gc request generated by soft threshold)
    struct gc_operation *next_node;

    int64_t x_init_time;           // time when gc initialized, when the threshold is reached.
    int64_t x_expected_start_time; // (for time window mechanism) expected start time, beggining of current or next time window
    int64_t x_start_time;          // time when gc is started
    int64_t x_end_time;            // time when gc is done
    double x_free_percentage;      // free page percentage in the plane when gc is initialized.
    unsigned int x_moved_pages;    // the number of page moved during the gc process
};

struct cxl_switch_device_info *initiation(struct cxl_switch_device_info *);
struct parameter_value *load_parameters(char parameter_file[30]);
struct page_info * initialize_page(struct page_info * p_page);
struct blk_info * initialize_block(struct blk_info * p_block,struct parameter_value *parameter);
struct plane_info * initialize_plane(struct plane_info * p_plane,struct parameter_value *parameter );
struct die_info * initialize_die(struct die_info * p_die,struct parameter_value *parameter,long long current_time );
struct chip_info * initialize_chip(struct chip_info * p_chip,struct parameter_value *parameter,long long current_time );
struct cxl_switch_device_info * initialize_channels(struct cxl_switch_device_info * ssd );
struct dram_info * initialize_dram(struct cxl_switch_device_info * ssd);
struct npu_info * initialize_npu(struct cxl_switch_device_info * ssd);
void initSet(struct Set *s, size_t initial_capacity);
void addSet(struct Set *s, unsigned int element);
void deleteSet(struct Set *s, unsigned int element);
void freeSet(struct  Set *s);
void printSet(struct Set *s);

#endif

