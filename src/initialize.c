/*****************************************************************************************************************************
  This project was supported by the National Basic Research 973 Program of China under Grant No.2011CB302301
  Huazhong University of Science and Technology (HUST)   Wuhan National Laboratory for Optoelectronics

  FileName： initialize.c
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

#define _CRTDBG_MAP_ALLOC

#include <stdlib.h>
#include "../include/initialize.h"
#include "../include/pagemap.h"

#define FALSE		0
#define TRUE		1

#define ACTIVE_FIXED 0
#define ACTIVE_ADJUST 1


/************************************************************************
 * Compare function for AVL Tree                                        
 ************************************************************************/
extern int keyCompareFunc(TREE_NODE *p , TREE_NODE *p1)
{
    struct buffer_group *T1=NULL,*T2=NULL;

    T1=(struct buffer_group*)p;
    T2=(struct buffer_group*)p1;


    if(T1->group< T2->group) return 1;
    if(T1->group> T2->group) return -1;

    return 0;
}


extern int freeFunc(TREE_NODE *pNode)
{

    if(pNode!=NULL)
    {
        free((void *)pNode);
    }


    pNode=NULL;
    return 1;
}


/**********   initiation   ******************
 *modify by zhouwen
 *November 08,2011
 *initialize the ssd struct to simulate the ssd hardware
 *1.this function allocate memory for ssd structure 
 *2.set the infomation according to the parameter file
 *******************************************/
struct cxl_switch_device_info *initiation(struct cxl_switch_device_info *ssd) {
    char buffer[300];
    struct parameter_value *parameters;
    FILE *fp=NULL;

    //导入ssd的配置文件 | Import SSD configuration file
    parameters=load_parameters(ssd->parameterfilename);
    // printf("chip_per_channel %d %d\n", parameters->chip_num_per_channel[0], parameters->chip_num_per_channel[1]);
    ssd->parameter=parameters;
    ssd->min_lsn=0x7fffffff;    // 0b1111111111111111111111111111111 (32 bit, all 1)
    ssd->page=ssd->parameter->chip_num*ssd->parameter->die_chip*ssd->parameter->plane_die*ssd->parameter->block_plane*ssd->parameter->page_block;

    // 初始化NPU
    ssd->npu = (struct npu_info *)malloc(sizeof(struct npu_info));
    alloc_assert(ssd->npu,"ssd->npu");
    memset(ssd->npu,0,sizeof(struct npu_info));
    initialize_npu(ssd);

    //初始化 dram | initialize dram
    ssd->dram = (struct dram_info *)malloc(sizeof(struct dram_info));
    alloc_assert(ssd->dram,"ssd->dram");
    memset(ssd->dram,0,sizeof(struct dram_info));
    initialize_dram(ssd);

    //初始化通道 | initialize channel
    ssd->top_channel_head=(struct channel_info*)malloc(ssd->parameter->gpu_channel_number * sizeof(struct channel_info));
    alloc_assert(ssd->top_channel_head, "ssd->top_channel_head");
    memset(ssd->top_channel_head, 0, ssd->parameter->gpu_channel_number * sizeof(struct channel_info));
    ssd->bottom_channel_head=(struct channel_info*)malloc(ssd->parameter->cxl_channel_number * sizeof(struct channel_info));
    alloc_assert(ssd->bottom_channel_head, "ssd->bottom_channel_head");
    memset(ssd->bottom_channel_head, 0, ssd->parameter->cxl_channel_number * sizeof(struct channel_info));
    initialize_channels(ssd );

    ssd->outputfile=fopen(ssd->outputfilename,"w");
    if(ssd->outputfile==NULL)
    {
        printf("the output file can't open\n");
        return NULL;
    }

    ssd->statisticfile=fopen(ssd->statisticfilename,"w");
    if(ssd->statisticfile==NULL)
    {
        printf("the statistic file can't open\n");
        return NULL;
    }

    ssd->statisticfile2=fopen(ssd->statisticfilename2,"w");
    if(ssd->statisticfile2==NULL)
    {
        printf("the statistic2 file can't open\n");
        return NULL;
    }

    ssd->outfile_gc=fopen(ssd->outfile_gc_name,"w");
    if(ssd->outfile_gc==NULL)
    {
        printf("the outfile_gc file can't open\n");
        return NULL;
    }

    ssd->outfile_io=fopen(ssd->outfile_io_name,"w");
    if(ssd->outfile_io==NULL)
    {
        printf("the outfile_io file can't open\n");
        return NULL;
    }

    ssd->outfile_io_write=fopen(ssd->outfile_io_write_name,"w");
    if(ssd->outfile_io_write==NULL)
    {
        printf("the outfile_io_write file can't open\n");
        return NULL;
    }

    ssd->outfile_io_read=fopen(ssd->outfile_io_read_name,"w");
    if(ssd->outfile_io_read==NULL)
    {
        printf("the outfile_io_read file can't open\n");
        return NULL;
    }


    fprintf(ssd->outputfile,"parameter file: %s\n",ssd->parameterfilename); 
    fprintf(ssd->outputfile,"trace file: %s\n",ssd->tracefilename);
    fprintf(ssd->statisticfile,"parameter file: %s\n",ssd->parameterfilename); 
    fprintf(ssd->statisticfile,"trace file: %s\n",ssd->tracefilename);

    fflush(ssd->outputfile);
    fflush(ssd->statisticfile);

    fp=fopen(ssd->parameterfilename,"r");
    if(fp==NULL)
    {
        printf("\nthe parameter file can't open!\n");
        return NULL;
    }

    //fp=fopen(ssd->parameterfilename,"r");

    fprintf(ssd->outputfile,"-----------------------parameter file----------------------\n");
    fprintf(ssd->statisticfile,"-----------------------parameter file----------------------\n");
    while(fgets(buffer,300,fp))
    {
        fprintf(ssd->outputfile,"%s",buffer);
        fflush(ssd->outputfile);
        fprintf(ssd->statisticfile,"%s",buffer);
        fflush(ssd->statisticfile);
    }

    fprintf(ssd->outputfile,"\n");
    fprintf(ssd->outputfile,"-----------------------simulation output----------------------\n");
    fflush(ssd->outputfile);

    fprintf(ssd->statisticfile,"\n");
    fprintf(ssd->statisticfile,"-----------------------simulation output----------------------\n");
    fflush(ssd->statisticfile);

    fclose(fp);
    // printf("initiation is completed!\n");

    return ssd;
}

struct npu_info * initialize_npu(struct cxl_switch_device_info * ssd)
{
    unsigned int page_num;

    struct npu_info *npu = ssd->npu;
    npu->req_idx = 0;
    npu->step = 0;


    int length = sizeof(npu->step_issue) / sizeof(npu->step_issue[0]);
    for (int i = 0; i < length; ++i) {
        npu->step_issue[i] = 0;
    }

    for (int i = 0; i < 30; i++){
        npu->step_wait_reqs[i] = (struct Set *)malloc(sizeof(struct Set));
        alloc_assert(npu->step_wait_reqs[i], "npu->step_wait_reqs");
        memset(npu->step_wait_reqs[i], 0, sizeof(struct Set));
        initSet(npu->step_wait_reqs[i], 10);
    }

    npu->finished_reqs = (struct Set *)malloc(sizeof(struct Set));
    alloc_assert(npu->finished_reqs, "npu->finished_reqs");
    memset(npu->finished_reqs, 0, sizeof(struct Set));
    initSet(npu->finished_reqs, 10);

    return npu;
}

struct dram_info * initialize_dram(struct cxl_switch_device_info * ssd)
{
    unsigned int page_num;

    struct dram_info *dram=ssd->dram;
    dram->dram_capacity = ssd->parameter->dram_capacity;		
    dram->buffer = (tAVLTree *)avlTreeCreate((void*)keyCompareFunc , (void *)freeFunc);
    dram->buffer->max_buffer_sector=ssd->parameter->dram_capacity/SECTOR; //512

    dram->map = (struct map_info *)malloc(sizeof(struct map_info));
    alloc_assert(dram->map,"dram->map");
    memset(dram->map,0, sizeof(struct map_info));

    page_num = ssd->parameter->page_block*ssd->parameter->block_plane*ssd->parameter->plane_die*ssd->parameter->die_chip*ssd->parameter->chip_num;

    dram->map->map_entry = (struct entry *)malloc(sizeof(struct entry) * page_num); //每个物理页和逻辑页都有对应关系 | Every physical page and logical page have a corresponding relationship
    alloc_assert(dram->map->map_entry,"dram->map->map_entry");
    memset(dram->map->map_entry,0,sizeof(struct entry) * page_num);

    return dram;
}

struct page_info * initialize_page(struct page_info * p_page )
{
    p_page->valid_state =0;
    p_page->free_state = PG_SUB;
    p_page->lpn = -1;
    p_page->written_count=0;
    return p_page;
}

struct blk_info * initialize_block(struct blk_info * p_block,struct parameter_value *parameter)
{
    unsigned int i;
    struct page_info * p_page;

    p_block->free_page_num = parameter->page_block;	// all pages are free
    p_block->last_write_page = -1;	// no page has been programmed

    p_block->page_head = (struct page_info *)malloc(parameter->page_block * sizeof(struct page_info));
    alloc_assert(p_block->page_head,"p_block->page_head");
    memset(p_block->page_head,0,parameter->page_block * sizeof(struct page_info));

    for(i = 0; i<parameter->page_block; i++)
    {
        p_page = &(p_block->page_head[i]);
        initialize_page(p_page );
    }
    return p_block;

}

struct plane_info * initialize_plane(struct plane_info * p_plane,struct parameter_value *parameter )
{
    unsigned int i;
    struct blk_info * p_block;
    p_plane->add_reg_ppn = -1;  //plane 里面的额外寄存器additional register -1 表示无数据
    p_plane->free_page=parameter->block_plane*parameter->page_block;

    p_plane->blk_head = (struct blk_info *)malloc(parameter->block_plane * sizeof(struct blk_info));
    alloc_assert(p_plane->blk_head,"p_plane->blk_head");
    memset(p_plane->blk_head,0,parameter->block_plane * sizeof(struct blk_info));

    for(i = 0; i<parameter->block_plane; i++)
    {
        p_block = &(p_plane->blk_head[i]);
        initialize_block( p_block ,parameter);			
    }
    return p_plane;
}

struct die_info * initialize_die(struct die_info * p_die,struct parameter_value *parameter,long long current_time )
{
    unsigned int i;
    struct plane_info * p_plane;

    p_die->token=0;

    p_die->plane_head = (struct plane_info*)malloc(parameter->plane_die * sizeof(struct plane_info));
    alloc_assert(p_die->plane_head,"p_die->plane_head");
    memset(p_die->plane_head,0,parameter->plane_die * sizeof(struct plane_info));

    for (i = 0; i<parameter->plane_die; i++)
    {
        p_plane = &(p_die->plane_head[i]);
        initialize_plane(p_plane,parameter );
    }

    return p_die;
}

struct pe_info * initialize_pe(struct pe_info * p_pe,struct parameter_value *parameter,long long current_time )
{
    p_pe->current_state = PE_IDLE;
    p_pe->next_state = PE_IDLE;
    p_pe->current_time = current_time;
    p_pe->next_state_predict_time = 0;		
    p_pe->subs_idx = -1;	

    return p_pe;
}


struct cxlctrl_info * initialize_cxlctrl(struct cxlctrl_info * p_cxlctrl, struct parameter_value *parameter,long long current_time )
{
    p_cxlctrl->current_state = CC_IDLE;
    p_cxlctrl->next_state = CC_IDLE;
    p_cxlctrl->current_time = current_time;
    p_cxlctrl->next_state_predict_time = 0;	
    p_cxlctrl->buf_size = parameter->cxlctrl_buf_size;	
    p_cxlctrl->buf_used_size = 0;
    // p_cxlctrl->subs_idx = -1;	

    return p_cxlctrl;
}

struct cxldram_info * initialize_cxldram(struct cxldram_info * p_cxldram, struct parameter_value *parameter,long long current_time )
{
    p_cxldram->current_state = DRAM_IDLE;
    p_cxldram->next_state = DRAM_IDLE;
    p_cxldram->current_time = current_time;
    p_cxldram->next_state_predict_time = 0;		
    // p_cxlctrl->subs_idx = -1;	

    return p_cxldram;
}

struct chip_info * initialize_chip(struct chip_info * p_chip,struct parameter_value *parameter,long long current_time )
{
    unsigned int i;
    struct die_info *p_die;

    p_chip->current_state = CHIP_IDLE;
    p_chip->next_state = CHIP_IDLE;
    p_chip->current_time = current_time;
    p_chip->next_state_predict_time = 0;			
    p_chip->die_num = parameter->die_chip;
    p_chip->plane_num_die = parameter->plane_die;
    p_chip->block_num_plane = parameter->block_plane;
    p_chip->page_num_block = parameter->block_plane;
    p_chip->subpage_num_page = parameter->subpage_page;
    p_chip->ers_limit = parameter->ers_limit;
    p_chip->token=0;
    p_chip->ac_timing = parameter->time_characteristics;		
    p_chip->read_count = 0;
    p_chip->program_count = 0;
    p_chip->erase_count = 0;
    p_chip->subs_idx = -1;

    p_chip->die_head = (struct die_info *)malloc(parameter->die_chip * sizeof(struct die_info));
    alloc_assert(p_chip->die_head,"p_chip->die_head");
    memset(p_chip->die_head,0,parameter->die_chip * sizeof(struct die_info));

    for (i = 0; i<parameter->die_chip; i++)
    {
        p_die = &(p_chip->die_head[i]);
        initialize_die( p_die,parameter,current_time );	
    }

    p_chip->pe = (struct pe_info *)malloc(sizeof(struct pe_info));
    memset(p_chip->pe, 0, sizeof(struct pe_info));
    p_chip->cxlctrl = (struct cxlctrl_info *)malloc(sizeof(struct cxlctrl_info));
    memset(p_chip->cxlctrl, 0, sizeof(struct cxlctrl_info));
    p_chip->cxldram = (struct cxldram_info *)malloc(sizeof(struct cxldram_info));
    memset(p_chip->cxldram, 0, sizeof(struct cxldram_info));


    return p_chip;
}

struct cxl_switch_device_info * initialize_channels(struct cxl_switch_device_info * ssd )
{
    unsigned int i,j;
    struct channel_info * p_channel;
    struct chip_info * p_chip;

    for (i = 0; i< ssd->parameter->gpu_channel_number; i++)
    {
        p_channel = &(ssd->top_channel_head[i]);
        p_channel->current_state = CHANNEL_IDLE;
        p_channel->next_state = CHANNEL_IDLE;
        p_channel->subs_idx = 0;
    }
    for (i = 0; i< ssd->parameter->cxl_channel_number; i++)
    {
        // channel info
        p_channel = &(ssd->bottom_channel_head[i]);
        p_channel->current_state = CHANNEL_IDLE;
        p_channel->next_state = CHANNEL_IDLE;
        p_channel->subs_idx = 0;

        // cxl device info
        p_channel->cxl_device = (struct cxl_device_info *)malloc(sizeof(struct cxl_device_info));
        p_channel->cxl_device->chip_num = ssd->parameter->chip_num_per_channel[i];
        p_channel->cxl_device->chip_head = (struct chip_info *)malloc(p_channel->cxl_device->chip_num * sizeof(struct chip_info));
        p_channel->cxl_device->current_states = (int *)malloc(p_channel->cxl_device->chip_num * sizeof(int));
        p_channel->cxl_device->next_states = (int *)malloc(p_channel->cxl_device->chip_num * sizeof(int));
        p_channel->cxl_device->current_time = ssd->current_time;
        p_channel->cxl_device->next_state_predict_time = 0;

        // chip info
        alloc_assert(p_channel->cxl_device->chip_head,"p_channel->chip_head");
        memset(p_channel->cxl_device->chip_head, 0, p_channel->cxl_device->chip_num * sizeof(struct chip_info));
        for (j = 0; j < p_channel->cxl_device->chip_num; j++) {
            p_channel->cxl_device->current_states[j] = CHIP_IDLE;
            p_channel->cxl_device->next_states[j] = CHIP_IDLE;
            p_chip = &(p_channel->cxl_device->chip_head[j]);
            initialize_chip(p_chip, ssd->parameter, ssd->current_time);
            initialize_pe(p_chip->pe, ssd->parameter,ssd->current_time);
            initialize_cxlctrl(p_chip->cxlctrl, ssd->parameter,ssd->current_time);
            initialize_cxldram(p_chip->cxldram, ssd->parameter,ssd->current_time);
        }
    }

    return ssd;
}


/*************************************************
 *将config/parameters.conf里面的参数导入到ssd->parameter里
 *modify by zhouwen
 *November 8,2011
 **************************************************/
struct parameter_value *load_parameters(char parameter_file[30])
{
    FILE * fp;
    FILE * fp1;
    FILE * fp2;
    //errno_t ferr;
    struct parameter_value *p;
    char buf[BUFSIZE];
    int i;
    int pre_eql,next_eql;
    int res_eql;
    char *ptr;

    p = (struct parameter_value *)malloc(sizeof(struct parameter_value));
    alloc_assert(p,"parameter_value");
    memset(p,0,sizeof(struct parameter_value));
    p->queue_length=8;
    memset(buf,0,BUFSIZE);

    fp=fopen(parameter_file,"r");
    if(fp == NULL)
    {	
        printf("the file parameter_file error!\n");	
        return p;
    }
    // fp1=fopen("raw/parameters_name.txt","w");
    // if(fp1==NULL)
    // {	
    //     printf("the file parameter_name error!\n");	
    //     return p;
    // }
    // fp2=fopen("raw/parameters_value.txt","w");
    // if(fp2==NULL)
    // {	
    //     printf("the file parameter_value error!\n");	
    //     return p;
    // }

    while(fgets(buf,200,fp)){
        if(buf[0] =='#' || buf[0] == ' ') continue;
        ptr=strchr(buf,'=');
        if(!ptr) continue; 

        pre_eql = ptr - buf;
        next_eql = pre_eql + 1;

        while(buf[pre_eql-1] == ' ') pre_eql--;
        buf[pre_eql] = 0;
        if((res_eql=strcmp(buf,"chip number")) ==0){
            sscanf(buf + next_eql,"%d",&p->chip_num);
        }else if((res_eql=strcmp(buf,"dram capacity")) ==0){
            sscanf(buf + next_eql,"%d",&p->dram_capacity);
        }else if((res_eql=strcmp(buf,"cpu sdram")) ==0){
            sscanf(buf + next_eql,"%d",&p->cpu_sdram);
        }else if((res_eql=strcmp(buf,"gpu_channel_number")) ==0){
            sscanf(buf + next_eql,"%d",&p->gpu_channel_number); 
        }else if((res_eql=strcmp(buf,"cxl_channel_number")) ==0){
            sscanf(buf + next_eql,"%d",&p->cxl_channel_number); \
        }else if((res_eql=strcmp(buf,"ndp_compute_model")) ==0){
            sscanf(buf + next_eql,"%d",&p->ndp_compute_model);
        }else if((res_eql=strcmp(buf,"cxlctrl_buf_size")) ==0){
            sscanf(buf + next_eql,"%d",&p->cxlctrl_buf_size);
        }else if((res_eql=strcmp(buf,"sub_req_inst_size")) ==0){
            sscanf(buf + next_eql,"%d",&p->sub_req_inst_size); 
        }else if((res_eql=strcmp(buf,"die number")) ==0){
            sscanf(buf + next_eql,"%d",&p->die_chip); 
        }else if((res_eql=strcmp(buf,"plane number")) ==0){
            sscanf(buf + next_eql,"%d",&p->plane_die); 
        }else if((res_eql=strcmp(buf,"block number")) ==0){
            sscanf(buf + next_eql,"%d",&p->block_plane); 
        }else if((res_eql=strcmp(buf,"page number")) ==0){
            sscanf(buf + next_eql,"%d",&p->page_block); 
        // }else if((res_eql=strcmp(buf,"subpage page")) ==0){
        //     sscanf(buf + next_eql,"%d",&p->subpage_page); 
        }else if((res_eql=strcmp(buf,"page capacity")) ==0){
            sscanf(buf + next_eql,"%d",&p->page_capacity); 
        }else if((res_eql=strcmp(buf,"subpage capacity")) ==0){
            sscanf(buf + next_eql,"%d",&p->subpage_capacity); 
        }else if((res_eql=strcmp(buf,"chip_computing_power")) ==0){
            sscanf(buf + next_eql,"%d",&p->chip_computing_power);
        }else if((res_eql=strcmp(buf,"t_CMD_CXL")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tCMDCXL);
        }else if((res_eql=strcmp(buf,"t_CMD_DRAM")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tCMDDRAM);
        }else if((res_eql=strcmp(buf,"t_ANALYZE")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tANALYZE);
        }else if((res_eql=strcmp(buf,"L_CXL_switch")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.L_CXL_switch);
        }else if((res_eql=strcmp(buf,"L_read_compute_cmd")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.L_read_compute_cmd);
        }else if((res_eql=strcmp(buf,"cxl_bandwidth")) ==0){
            sscanf(buf + next_eql,"%d",&p->cxl_bandwidth);
        // }else if((res_eql=strcmp(buf,"dram_bandwidth")) ==0){
        //     sscanf(buf + next_eql,"%d",&p->dram_bandwidth);
        }else if((res_eql=strcmp(buf,"t_PROG")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tPROG); 
        }else if((res_eql=strcmp(buf,"t_DBSY")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tDBSY); 
        }else if((res_eql=strcmp(buf,"t_BERS")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tBERS); 
        }else if((res_eql=strcmp(buf,"t_CLS")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tCLS); 
        }else if((res_eql=strcmp(buf,"t_CLH")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tCLH); 
        }else if((res_eql=strcmp(buf,"t_CS")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tCS); 
        }else if((res_eql=strcmp(buf,"t_CH")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tCH); 
        }else if((res_eql=strcmp(buf,"t_WP")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tWP); 
        }else if((res_eql=strcmp(buf,"t_ALS")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tALS); 
        }else if((res_eql=strcmp(buf,"t_ALH")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tALH); 
        }else if((res_eql=strcmp(buf,"t_DS")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tDS); 
        }else if((res_eql=strcmp(buf,"t_DH")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tDH); 
        }else if((res_eql=strcmp(buf,"t_WC")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tWC); 
        }else if((res_eql=strcmp(buf,"t_WH")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tWH); 
        }else if((res_eql=strcmp(buf,"t_ADL")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tADL); 
        }else if((res_eql=strcmp(buf,"t_R")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tR); 
        }else if((res_eql=strcmp(buf,"t_AR")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tAR); 
        }else if((res_eql=strcmp(buf,"t_CLR")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tCLR); 
        }else if((res_eql=strcmp(buf,"t_RR")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tRR); 
        }else if((res_eql=strcmp(buf,"t_RP")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tRP); 
        }else if((res_eql=strcmp(buf,"t_WB")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tWB); 
        }else if((res_eql=strcmp(buf,"t_RC")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tRC); 
        }else if((res_eql=strcmp(buf,"t_REA")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tREA); 
        }else if((res_eql=strcmp(buf,"t_CEA")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tCEA); 
        }else if((res_eql=strcmp(buf,"t_RHZ")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tRHZ); 
        }else if((res_eql=strcmp(buf,"t_CHZ")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tCHZ); 
        }else if((res_eql=strcmp(buf,"t_RHOH")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tRHOH); 
        }else if((res_eql=strcmp(buf,"t_RLOH")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tRLOH); 
        }else if((res_eql=strcmp(buf,"t_COH")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tCOH); 
        }else if((res_eql=strcmp(buf,"t_REH")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tREH); 
        }else if((res_eql=strcmp(buf,"t_IR")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tIR); 
        }else if((res_eql=strcmp(buf,"t_RHW")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tRHW); 
        }else if((res_eql=strcmp(buf,"t_WHR")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tWHR); 
        }else if((res_eql=strcmp(buf,"t_RST")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tRST); 
        }else if((res_eql=strcmp(buf,"erase limit")) ==0){
            sscanf(buf + next_eql,"%d",&p->ers_limit); 
        }else if((res_eql=strcmp(buf,"flash operating current")) ==0){
            sscanf(buf + next_eql,"%lf",&p->operating_current); 
        }else if((res_eql=strcmp(buf,"flash supply voltage")) ==0){
            sscanf(buf + next_eql,"%lf",&p->supply_voltage); 
        }else if((res_eql=strcmp(buf,"dram active current")) ==0){
            sscanf(buf + next_eql,"%lf",&p->dram_active_current); 
        }else if((res_eql=strcmp(buf,"dram standby current")) ==0){
            sscanf(buf + next_eql,"%lf",&p->dram_standby_current); 
        }else if((res_eql=strcmp(buf,"dram refresh current")) ==0){
            sscanf(buf + next_eql,"%lf",&p->dram_refresh_current); 
        }else if((res_eql=strcmp(buf,"dram voltage")) ==0){
            sscanf(buf + next_eql,"%lf",&p->dram_voltage); 
        }else if((res_eql=strcmp(buf,"address mapping")) ==0){
            sscanf(buf + next_eql,"%d",&p->address_mapping); 
        }else if((res_eql=strcmp(buf,"wear leveling")) ==0){
            sscanf(buf + next_eql,"%d",&p->wear_leveling); 
        }else if((res_eql=strcmp(buf,"gc")) ==0){
            sscanf(buf + next_eql,"%d",&p->gc); 
        }else if((res_eql=strcmp(buf,"clean in background")) ==0){
            sscanf(buf + next_eql,"%d",&p->clean_in_background); 
        }else if((res_eql=strcmp(buf,"overprovide")) ==0){
            sscanf(buf + next_eql,"%f",&p->overprovide); 
        }else if((res_eql=strcmp(buf,"gc threshold")) ==0){
            sscanf(buf + next_eql,"%f",&p->gc_threshold); 
        }else if((res_eql=strcmp(buf,"buffer management")) ==0){
            sscanf(buf + next_eql,"%d",&p->buffer_management); 
        }else if((res_eql=strcmp(buf,"scheduling algorithm")) ==0){
            sscanf(buf + next_eql,"%d",&p->scheduling_algorithm); 
        }else if((res_eql=strcmp(buf,"quick table radio")) ==0){
            sscanf(buf + next_eql,"%f",&p->quick_radio); 
        }else if((res_eql=strcmp(buf,"related mapping")) ==0){
            sscanf(buf + next_eql,"%d",&p->related_mapping); 
        }else if((res_eql=strcmp(buf,"striping")) ==0){
            sscanf(buf + next_eql,"%d",&p->striping); 
        }else if((res_eql=strcmp(buf,"interleaving")) ==0){
            sscanf(buf + next_eql,"%d",&p->interleaving); 
        }else if((res_eql=strcmp(buf,"pipelining")) ==0){
            sscanf(buf + next_eql,"%d",&p->pipelining); 
        }else if((res_eql=strcmp(buf,"time_step")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_step); 
        }else if((res_eql=strcmp(buf,"small large write")) ==0){
            sscanf(buf + next_eql,"%d",&p->small_large_write); 
        }else if((res_eql=strcmp(buf,"active write threshold")) ==0){
            sscanf(buf + next_eql,"%d",&p->threshold_fixed_adjust); 
        }else if((res_eql=strcmp(buf,"threshold value")) ==0){
            sscanf(buf + next_eql,"%d",&p->threshold_value); 
        }else if((res_eql=strcmp(buf,"active write")) ==0){
            sscanf(buf + next_eql,"%d",&p->active_write); 
        }else if((res_eql=strcmp(buf,"gc hard threshold")) ==0){
            sscanf(buf + next_eql,"%f",&p->gc_hard_threshold); 
        }else if((res_eql=strcmp(buf,"allocation")) ==0){
            sscanf(buf + next_eql,"%d",&p->allocation_scheme); 
        }else if((res_eql=strcmp(buf,"static_allocation")) ==0){
            sscanf(buf + next_eql,"%d",&p->static_allocation); 
        }else if((res_eql=strcmp(buf,"dynamic_allocation")) ==0){
            sscanf(buf + next_eql,"%d",&p->dynamic_allocation); 
        }else if((res_eql=strcmp(buf,"advanced command")) ==0){
            sscanf(buf + next_eql,"%d",&p->advanced_commands); 
        }else if((res_eql=strcmp(buf,"advanced command priority")) ==0){
            sscanf(buf + next_eql,"%d",&p->ad_priority); 
        }else if((res_eql=strcmp(buf,"advanced command priority2")) ==0){
            sscanf(buf + next_eql,"%d",&p->ad_priority2); 
        }else if((res_eql=strcmp(buf,"greed CB command")) ==0){
            sscanf(buf + next_eql,"%d",&p->greed_CB_ad); 
        }else if((res_eql=strcmp(buf,"greed MPW command")) ==0){
            sscanf(buf + next_eql,"%d",&p->greed_MPW_ad); 
        }else if((res_eql=strcmp(buf,"aged")) ==0){
            sscanf(buf + next_eql,"%d",&p->aged); 
        }else if((res_eql=strcmp(buf,"aged ratio")) ==0){
            sscanf(buf + next_eql,"%f",&p->aged_ratio); 
        }else if((res_eql=strcmp(buf,"queue_length")) ==0){
            sscanf(buf + next_eql,"%d",&p->queue_length);
        }else if((res_eql=strcmp(buf,"npu computing power")) ==0){
            sscanf(buf + next_eql,"%d",&p->npu_computing_power); 
        }else if((res_eql=strcmp(buf,"model dim")) ==0){
            sscanf(buf + next_eql,"%d",&p->model_characteristics.model_dim); 
        }else if((res_eql=strcmp(buf,"input length")) ==0){
            sscanf(buf + next_eql,"%d",&p->model_characteristics.input_length); 
        }else if((res_eql=strcmp(buf,"data type")) ==0){
            sscanf(buf + next_eql,"%d",&p->model_characteristics.data_type); 
        }else if((res_eql=strcmp(buf,"dram_channel_num")) ==0){
            sscanf(buf + next_eql,"%d",&p->dram_channel_num); 
        }else if((res_eql=strcmp(buf,"dram_channel_bandwidth")) ==0){
            sscanf(buf + next_eql,"%d",&p->dram_channel_bandwidth); 
        }else if((res_eql=strcmp(buf,"t_DRAM_READ_LATENCY")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tDRAMRL); 
        }else if((res_eql=strcmp(buf,"t_DRAM_READ_LATENCY")) ==0){
            sscanf(buf + next_eql,"%d",&p->time_characteristics.tDRAMWL); 
        }else if((res_eql=strcmp(buf,"chip_number")) ==0){
            int chip_per_channel;
            sscanf(buf + next_eql,"%d",&chip_per_channel);
            for (int j = 0; j < p->cxl_channel_number; j++) {
                p->chip_num_per_channel[j] = chip_per_channel;
            }
        }else{
            printf("don't match\t %s\n",buf);
        }

        memset(buf,0,BUFSIZE);

    }
    fclose(fp);

    // p->subpage_capacity = p->page_capacity / p->subpage_page;
    p->subpage_page = p->page_capacity / p->subpage_capacity;
    p->model_characteristics.ffn_dim = 4 * p->model_characteristics.model_dim;
    p->dram_bandwidth = p->dram_channel_num * p->dram_channel_bandwidth;

    return p;
}


// 初始化集合
void initSet(struct Set *s, size_t initial_capacity) {
    s->array = malloc(initial_capacity * sizeof(unsigned int));
    if (s->array == NULL) {
        perror("Unable to allocate memory for the set");
        exit(EXIT_FAILURE);
    }
    s->used = 0;
    s->capacity = initial_capacity;
}

// 添加元素到集合
void addSet(struct Set *s, unsigned int element) {
    // 检查是否需要扩展数组
    if (s->used == s->capacity) {
        s->capacity *= 2;
        unsigned int *new_array = realloc(s->array, s->capacity * sizeof(unsigned int));
        if (new_array == NULL) {
            perror("Unable to reallocate memory for the set");
            free(s->array);
            exit(EXIT_FAILURE);
        }
        s->array = new_array;
    }
    // 添加元素
    s->array[s->used++] = element;
}

// 从集合中删除元素（如果存在）
void deleteSet(struct Set *s, unsigned int element) {
    for (size_t i = 0; i < s->used; i++) {
        if (s->array[i] == element) {
            // 向左移动元素覆盖删除的元素
            for (size_t j = i; j < s->used - 1; j++) {
                s->array[j] = s->array[j + 1];
            }
            s->used--;
            return;
        }
    }
}

// 销毁集合并释放内存
void freeSet(struct Set *s) {
    free(s->array);
    s->array = NULL;
    s->used = s->capacity = 0;
}

// 打印集合中的所有元素
void printSet(struct Set *s) {
    printf("Set: ");
    for (size_t i = 0; i < s->used; i++) {
        printf("%d ", s->array[i]);
    }
    printf("\n");
}