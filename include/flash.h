/*****************************************************************************************************************************
  This project was supported by the National Basic Research 973 Program of China under Grant No.2011CB302301
  Huazhong University of Science and Technology (HUST)   Wuhan National Laboratory for Optoelectronics

  FileName： flash.h
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

#ifndef FLASH_H
#define FLASH_H 100000

#include <stdlib.h>
#include "pagemap.h"

struct cxl_switch_device_info *process(struct cxl_switch_device_info *);
struct cxl_switch_device_info *insert2buffer(struct cxl_switch_device_info *, unsigned int, int, struct sub_request *, struct request *);

struct sub_request * creat_sub_request(struct cxl_switch_device_info * ssd, unsigned int lpn, int size, unsigned int state, struct request * req, unsigned int operation);
struct sub_request * creat_sub_request_simple(struct cxl_switch_device_info * ssd, struct addr_info ** p_addr, unsigned int addr_num, unsigned int chip, int size, struct request * req);

int  find_active_block(struct cxl_switch_device_info *ssd, unsigned int channel, unsigned int chip, unsigned int die, unsigned int plane);
int allocate_location(struct cxl_switch_device_info * ssd , struct sub_request *sub_req);


int go_one_step_in_channel(struct cxl_switch_device_info * ssd, struct sub_request * sub, unsigned int aim_state);
int go_one_step_in_cxlctrl(struct cxl_switch_device_info * ssd, struct sub_request * sub, unsigned int aim_state);

#endif
