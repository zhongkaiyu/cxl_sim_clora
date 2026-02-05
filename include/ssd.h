/*****************************************************************************************************************************
  This project was supported by the National Basic Research 973 Program of China under Grant No.2011CB302301
  Huazhong University of Science and Technology (HUST)   Wuhan National Laboratory for Optoelectronics

FileName: ssd.h
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
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <sys/types.h>
#include "initialize.h"
#include "flash.h"
#include "pagemap.h"
#include "readJson.h"

#define MAX_INT64  0x7fffffffffffffffll

struct cxl_switch_device_info *simulate(struct cxl_switch_device_info *, char * trace_filename);
int get_requests(struct cxl_switch_device_info *);
int npu_process(struct cxl_switch_device_info *, JsonData *);
struct cxl_switch_device_info *buffer_management(struct cxl_switch_device_info *);
unsigned int lpn2ppn(struct cxl_switch_device_info * , unsigned int lsn);
struct cxl_switch_device_info *distribute(struct cxl_switch_device_info *);
void trace_output(struct cxl_switch_device_info* );
void statistic_output(struct cxl_switch_device_info *);
unsigned int size(unsigned int);
unsigned int transfer_size(struct cxl_switch_device_info *, int, unsigned int, struct request *);
int64_t find_nearest_event(struct cxl_switch_device_info *);
int64_t find_nearest_event_sys(struct cxl_switch_device_info *);
void free_all_node(struct cxl_switch_device_info *);
struct cxl_switch_device_info *make_aged(struct cxl_switch_device_info *);
struct cxl_switch_device_info *slice_request(struct cxl_switch_device_info *);
struct cxl_switch_device_info *warmup(struct cxl_switch_device_info *);
struct cxl_switch_device_info *init_gc(struct cxl_switch_device_info *);
void print_gc_node(struct cxl_switch_device_info* ssd);

int parse_user_args(int, char *[], struct user_args *);
struct cxl_switch_device_info *initialize_ssd(struct cxl_switch_device_info*, struct user_args *);
struct cxl_switch_device_info *parse_args(struct cxl_switch_device_info *, int, char *[]);
void close_file(struct cxl_switch_device_info *);

void get_current_time(char *current_time);

/********************************************
* Function to display info and help
* Added by Fadhil Imam (fadhilimamk@gmail.com) 
* to print help and usage info | 29/06/2018
*********************************************/
void display_title();
void display_help();
void display_simulation_intro(struct cxl_switch_device_info *);