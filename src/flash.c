/*****************************************************************************************************************************
  This project was supported by the National Basic Research 973 Program of China under Grant No.2011CB302301
  Huazhong University of Science and Technology (HUST)   Wuhan National Laboratory for Optoelectronics

  FileName： flash.c
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

#include "../include/flash.h"
#include "../include/ssd.h"

/**********************
 *这个函数只作用于写请求 | This function only works on write requests.
 ***********************/
Status allocate_location(struct cxl_switch_device_info * ssd , struct sub_request *sub_req)
{
    struct sub_request * update=NULL;
    unsigned int channel_num=0,chip_num=0,die_num=0,plane_num=0;
    struct local *location=NULL;

    channel_num=ssd->parameter->cxl_channel_number;
    chip_num=ssd->parameter->chip_num_per_channel[0];
    die_num=ssd->parameter->die_chip;
    plane_num=ssd->parameter->plane_die;


    if (ssd->parameter->allocation_scheme==0)                                          /*动态分配的情况*/
    {
        /******************************************************************
         * 在动态分配中，因为页的更新操作使用不了copyback操作，
         *需要产生一个读请求，并且只有这个读请求完成后才能进行这个页的写操作
         *******************************************************************/
        if (ssd->dram->map->map_entry[sub_req->lpn].state!=0)    
        {
            if ((sub_req->state&ssd->dram->map->map_entry[sub_req->lpn].state)!=ssd->dram->map->map_entry[sub_req->lpn].state)
            {
                ssd->read_count++;
                ssd->in_read_size+=ssd->parameter->subpage_page;
                ssd->update_read_count++;

                update=(struct sub_request *)malloc(sizeof(struct sub_request));
                alloc_assert(update,"update");
                memset(update,0, sizeof(struct sub_request));

                if(update==NULL)
                {
                    return ERROR;
                }
                update->location=NULL;
                update->next_node=NULL;
                update->next_subs=NULL;
                update->update=NULL;						
                location = find_location(ssd,ssd->dram->map->map_entry[sub_req->lpn].pn);
                update->location=location;
                update->begin_time = ssd->current_time;
                update->current_state = SR_WAIT;
                update->current_time=MAX_INT64;
                update->next_state = SR_CHANNEL_R_CA_TRANSFER;
                update->next_state_predict_time=MAX_INT64;
                update->lpn = sub_req->lpn;
                update->state=((ssd->dram->map->map_entry[sub_req->lpn].state^sub_req->state)&0x7fffffff);
                update->size=size(update->state);
                update->ppn = ssd->dram->map->map_entry[sub_req->lpn].pn;
                update->operation = READ;

                if (ssd->bottom_channel_head[location->channel].subs_r_tail != NULL)            /*产生新的读请求，并且挂到channel的subs_r_tail队列尾*/
                {
                    ssd->bottom_channel_head[location->channel].subs_r_tail->next_node=update;
                    ssd->bottom_channel_head[location->channel].subs_r_tail=update;
                } 
                else
                {
                    ssd->bottom_channel_head[location->channel].subs_r_tail=update;
                    ssd->bottom_channel_head[location->channel].subs_r_head=update;
                }
            }
        }
        /***************************************
         *一下是动态分配的几种情况
         *0：全动态分配
         *1：表示channel定package，die，plane动态
         ****************************************/
        switch(ssd->parameter->dynamic_allocation)
        {
            case 0:
                {
                    sub_req->location->channel=-1;
                    sub_req->location->chip=-1;
                    sub_req->location->die=-1;
                    sub_req->location->plane=-1;
                    sub_req->location->block=-1;
                    sub_req->location->page=-1;

                    if (ssd->subs_w_tail!=NULL)
                    {
                        ssd->subs_w_tail->next_node=sub_req;
                        ssd->subs_w_tail=sub_req;
                    } 
                    else
                    {
                        ssd->subs_w_tail=sub_req;
                        ssd->subs_w_head=sub_req;
                    }

                    if (update!=NULL)
                    {
                        sub_req->update=update;
                    }

                    break;
                }
            case 1:
                {

                    sub_req->location->channel=sub_req->lpn%ssd->parameter->cxl_channel_number;
                    sub_req->location->chip=-1;
                    sub_req->location->die=-1;
                    sub_req->location->plane=-1;
                    sub_req->location->block=-1;
                    sub_req->location->page=-1;

                    if (update!=NULL)
                    {
                        sub_req->update=update;
                    }

                    break;
                }
            case 2:
                {
                    break;
                }
            case 3:
                {
                    break;
                }
        }

    }
    else                                                                          
    {   /***************************************************************************
         *是静态分配方式，所以可以将这个子请求的最终channel，chip，die，plane全部得出
         *总共有0,1,2,3,4,5,这六种静态分配方式。
         *Is a static allocation method, so you can get the final channel, chip, die, plane of this subrequest
         *There are a total of 0, 1, 2, 3, 4, 5, these six static allocation methods.
         ****************************************************************************/
        switch (ssd->parameter->static_allocation)
        {
            case 0:         //no striping static allocation
                {
                    sub_req->location->channel=(sub_req->lpn/(plane_num*die_num*chip_num))%channel_num;
                    sub_req->location->chip=sub_req->lpn%chip_num;
                    sub_req->location->die=(sub_req->lpn/chip_num)%die_num;
                    sub_req->location->plane=(sub_req->lpn/(die_num*chip_num))%plane_num;
                    break;
                }
            case 1:
                {
                    sub_req->location->channel=sub_req->lpn%channel_num;
                    sub_req->location->chip=(sub_req->lpn/channel_num)%chip_num;
                    sub_req->location->die=(sub_req->lpn/(chip_num*channel_num))%die_num;
                    sub_req->location->plane=(sub_req->lpn/(die_num*chip_num*channel_num))%plane_num;

                    break;
                }
            case 2:
                {
                    sub_req->location->channel=sub_req->lpn%channel_num;
                    sub_req->location->chip=(sub_req->lpn/(plane_num*channel_num))%chip_num;
                    sub_req->location->die=(sub_req->lpn/(plane_num*chip_num*channel_num))%die_num;
                    sub_req->location->plane=(sub_req->lpn/channel_num)%plane_num;
                    break;
                }
            case 3:
                {
                    sub_req->location->channel=sub_req->lpn%channel_num;
                    sub_req->location->chip=(sub_req->lpn/(die_num*channel_num))%chip_num;
                    sub_req->location->die=(sub_req->lpn/channel_num)%die_num;
                    sub_req->location->plane=(sub_req->lpn/(die_num*chip_num*channel_num))%plane_num;
                    break;
                }
            case 4:  
                {
                    sub_req->location->channel=sub_req->lpn%channel_num;
                    sub_req->location->chip=(sub_req->lpn/(plane_num*die_num*channel_num))%chip_num;
                    sub_req->location->die=(sub_req->lpn/(plane_num*channel_num))%die_num;
                    sub_req->location->plane=(sub_req->lpn/channel_num)%plane_num;

                    break;
                }
            case 5:   
                {
                    sub_req->location->channel=sub_req->lpn%channel_num;
                    sub_req->location->chip=(sub_req->lpn/(plane_num*die_num*channel_num))%chip_num;
                    sub_req->location->die=(sub_req->lpn/channel_num)%die_num;
                    sub_req->location->plane=(sub_req->lpn/(die_num*channel_num))%plane_num;

                    break;
                }
            default : return ERROR;

        }
        if (ssd->dram->map->map_entry[sub_req->lpn].state!=0)
        {                                                                              /*这个写回的子请求的逻辑页不可以覆盖之前被写回的数据 需要产生读请求*/ 
            if ((sub_req->state&ssd->dram->map->map_entry[sub_req->lpn].state)!=ssd->dram->map->map_entry[sub_req->lpn].state)  
            {
                ssd->read_count++;
                ssd->in_read_size+=ssd->parameter->subpage_page;
                ssd->update_read_count++;
                update=(struct sub_request *)malloc(sizeof(struct sub_request));
                alloc_assert(update,"update");
                memset(update,0, sizeof(struct sub_request));

                if(update==NULL)
                {
                    return ERROR;
                }
                update->location=NULL;
                update->next_node=NULL;
                update->next_subs=NULL;
                update->update=NULL;						
                location = find_location(ssd,ssd->dram->map->map_entry[sub_req->lpn].pn);
                update->location=location;
                update->begin_time = ssd->current_time;
                update->current_state = SR_WAIT;
                update->current_time=MAX_INT64;
                update->next_state = SR_CHANNEL_R_CA_TRANSFER;
                update->next_state_predict_time=MAX_INT64;
                update->lpn = sub_req->lpn;
                update->state=((ssd->dram->map->map_entry[sub_req->lpn].state^sub_req->state)&0x7fffffff);
                update->size=size(update->state);
                update->ppn = ssd->dram->map->map_entry[sub_req->lpn].pn;
                update->operation = READ;

                if (ssd->bottom_channel_head[location->channel].subs_r_tail != NULL)
                {
                    ssd->bottom_channel_head[location->channel].subs_r_tail->next_node=update;
                    ssd->bottom_channel_head[location->channel].subs_r_tail=update;
                } 
                else
                {
                    ssd->bottom_channel_head[location->channel].subs_r_tail=update;
                    ssd->bottom_channel_head[location->channel].subs_r_head=update;
                }
            }

            if (update!=NULL)
            {
                sub_req->update=update;

                sub_req->state=(sub_req->state|update->state);
                sub_req->size=size(sub_req->state);
            }

        }
    }
    if ((ssd->parameter->allocation_scheme!=0)||(ssd->parameter->dynamic_allocation!=0))
    {
        if (ssd->bottom_channel_head[sub_req->location->channel].subs_w_tail != NULL)
        {
            ssd->bottom_channel_head[sub_req->location->channel].subs_w_tail->next_node=sub_req;
            ssd->bottom_channel_head[sub_req->location->channel].subs_w_tail=sub_req;
        } 
        else
        {
            ssd->bottom_channel_head[sub_req->location->channel].subs_w_tail=sub_req;
            ssd->bottom_channel_head[sub_req->location->channel].subs_w_head=sub_req;
        }
    }
    return SUCCESS;					
}	

/*******************************************************************************
 *insert2buffer这个函数是专门为写请求分配子请求服务的在buffer_management中被调用。
 ********************************************************************************/
struct cxl_switch_device_info * insert2buffer(struct cxl_switch_device_info *ssd, unsigned int lpn, int state, struct sub_request *sub, struct request *req)
{
    int write_back_count,flag=0;                                                             /*flag表示为写入新数据腾空间是否完成，0表示需要进一步腾，1表示已经腾空*/
    unsigned int i,lsn,hit_flag,add_flag,sector_count,active_region_flag=0,free_sector=0;
    struct buffer_group *buffer_node=NULL,*pt,*new_node=NULL,key;
    struct sub_request *sub_req=NULL,*update=NULL;


    unsigned int sub_req_state=0, sub_req_size=0,sub_req_lpn=0;

#ifdef DEBUG
    printf("enter insert2buffer,  current time:%lld, lpn:%d, state:%d,\n",ssd->current_time,lpn,state);
#endif

    sector_count=size(state);                                                                /*需要写到buffer的sector个数*/
    key.group=lpn;
    buffer_node= (struct buffer_group*)avlTreeFind(ssd->dram->buffer, (TREE_NODE *)&key);    /*在平衡二叉树中寻找buffer node*/ 

    /************************************************************************************************
     *没有命中。
     *第一步根据这个lpn有多少子页需要写到buffer，去除已写回的lsn，为该lpn腾出位置，
     *首先即要计算出free sector（表示还有多少可以直接写的buffer节点）。
     *如果free_sector>=sector_count，即有多余的空间够lpn子请求写，不需要产生写回请求
     *否则，没有多余的空间供lpn子请求写，这时需要释放一部分空间，产生写回请求。就要creat_sub_request()
     *************************************************************************************************/
    if(buffer_node==NULL)
    {
        free_sector=ssd->dram->buffer->max_buffer_sector-ssd->dram->buffer->buffer_sector_count;   
        if(free_sector>=sector_count)
        {
            flag=1;    
        }
        if(flag==0)     
        {
            write_back_count=sector_count-free_sector;
            ssd->dram->buffer->write_miss_hit=ssd->dram->buffer->write_miss_hit+write_back_count;
            while(write_back_count>0)
            {
                sub_req=NULL;
                sub_req_state=ssd->dram->buffer->buffer_tail->stored; 
                sub_req_size=size(ssd->dram->buffer->buffer_tail->stored);
                sub_req_lpn=ssd->dram->buffer->buffer_tail->group;
                sub_req=creat_sub_request(ssd,sub_req_lpn,sub_req_size,sub_req_state,req,WRITE);

                /**********************************************************************************
                 *req不为空，表示这个insert2buffer函数是在buffer_management中调用，传递了request进来
                 *req为空，表示这个函数是在process函数中处理一对多映射关系的读的时候，需要将这个读出
                 *的数据加到buffer中，这可能产生实时的写回操作，需要将这个实时的写回操作的子请求挂在
                 *这个读请求的总请求上
                 ***********************************************************************************/
                if(req!=NULL)                                             
                {
                }
                else    
                {
                    sub_req->next_subs=sub->next_subs;
                    sub->next_subs=sub_req;
                }

                /*********************************************************************
                 *写请求插入到了平衡二叉树，这时就要修改dram的buffer_sector_count；
                 *维持平衡二叉树调用avlTreeDel()和AVL_TREENODE_FREE()函数；维持LRU算法；
                 **********************************************************************/
                ssd->dram->buffer->buffer_sector_count=ssd->dram->buffer->buffer_sector_count-sub_req->size;
                pt = ssd->dram->buffer->buffer_tail;
                avlTreeDel(ssd->dram->buffer, (TREE_NODE *) pt);
                if(ssd->dram->buffer->buffer_head->LRU_link_next == NULL){
                    ssd->dram->buffer->buffer_head = NULL;
                    ssd->dram->buffer->buffer_tail = NULL;
                }else{
                    ssd->dram->buffer->buffer_tail=ssd->dram->buffer->buffer_tail->LRU_link_pre;
                    ssd->dram->buffer->buffer_tail->LRU_link_next=NULL;
                }
                pt->LRU_link_next=NULL;
                pt->LRU_link_pre=NULL;
                AVL_TREENODE_FREE(ssd->dram->buffer, (TREE_NODE *) pt);
                pt = NULL;

                write_back_count=write_back_count-sub_req->size;                            /*因为产生了实时写回操作，需要将主动写回操作区域增加*/
            }
        }

        /******************************************************************************
         *生成一个buffer node，根据这个页的情况分别赋值个各个成员，添加到队首和二叉树中
         *******************************************************************************/
        new_node=NULL;
        new_node=(struct buffer_group *)malloc(sizeof(struct buffer_group));
        alloc_assert(new_node,"buffer_group_node");
        memset(new_node,0, sizeof(struct buffer_group));

        new_node->group=lpn;
        new_node->stored=state;
        new_node->dirty_clean=state;
        new_node->LRU_link_pre = NULL;
        new_node->LRU_link_next=ssd->dram->buffer->buffer_head;
        if(ssd->dram->buffer->buffer_head != NULL){
            ssd->dram->buffer->buffer_head->LRU_link_pre=new_node;
        }else{
            ssd->dram->buffer->buffer_tail = new_node;
        }
        ssd->dram->buffer->buffer_head=new_node;
        new_node->LRU_link_pre=NULL;
        avlTreeAdd(ssd->dram->buffer, (TREE_NODE *) new_node);
        ssd->dram->buffer->buffer_sector_count += sector_count;
    }
    /****************************************************************************************
     *在buffer中命中的情况
     *算然命中了，但是命中的只是lpn，有可能新来的写请求，只是需要写lpn这一page的某几个sub_page
     *这时有需要进一步的判断
     *****************************************************************************************/
    else
    {
        for(i=0;i<ssd->parameter->subpage_page;i++)
        {
            /*************************************************************
             *判断state第i位是不是1
             *并且判断第i个sector是否存在buffer中，1表示存在，0表示不存在。
             **************************************************************/
            if((state>>i)%2!=0)                                                         
            {
                lsn=lpn*ssd->parameter->subpage_page+i;
                hit_flag=0;
                hit_flag=(buffer_node->stored)&(0x00000001<<i);

                if(hit_flag!=0)				                                          /*命中了，需要将该节点移到buffer的队首，并且将命中的lsn进行标记*/
                {	
                    active_region_flag=1;                                             /*用来记录在这个buffer node中的lsn是否被命中，用于后面对阈值的判定*/

                    if(req!=NULL)
                    {
                        if(ssd->dram->buffer->buffer_head!=buffer_node)     
                        {				
                            if(ssd->dram->buffer->buffer_tail==buffer_node)
                            {				
                                ssd->dram->buffer->buffer_tail=buffer_node->LRU_link_pre;
                                buffer_node->LRU_link_pre->LRU_link_next=NULL;					
                            }				
                            else if(buffer_node != ssd->dram->buffer->buffer_head)
                            {					
                                buffer_node->LRU_link_pre->LRU_link_next=buffer_node->LRU_link_next;				
                                buffer_node->LRU_link_next->LRU_link_pre=buffer_node->LRU_link_pre;
                            }				
                            buffer_node->LRU_link_next=ssd->dram->buffer->buffer_head;	
                            ssd->dram->buffer->buffer_head->LRU_link_pre=buffer_node;
                            buffer_node->LRU_link_pre=NULL;				
                            ssd->dram->buffer->buffer_head=buffer_node;					
                        }					
                        ssd->dram->buffer->write_hit++;
                        req->complete_lsn_count++;                                        /*关键 当在buffer中命中时 就用req->complete_lsn_count++表示往buffer中写了数据。*/					
                    }
                    else
                    {
                    }				
                }			
                else                 			
                {
                    /************************************************************************************************************
                     *该lsn没有命中，但是节点在buffer中，需要将这个lsn加到buffer的对应节点中
                     *从buffer的末端找一个节点，将一个已经写回的lsn从节点中删除(如果找到的话)，更改这个节点的状态，同时将这个新的
                     *lsn加到相应的buffer节点中，该节点可能在buffer头，不在的话，将其移到头部。如果没有找到已经写回的lsn，在buffer
                     *节点找一个group整体写回，将这个子请求挂在这个请求上。可以提前挂在一个channel上。
                     *第一步:将buffer队尾的已经写回的节点删除一个，为新的lsn腾出空间，这里需要修改队尾某节点的stored状态这里还需要
                     *       增加，当没有可以之间删除的lsn时，需要产生新的写子请求，写回LRU最后的节点。
                     *第二步:将新的lsn加到所述的buffer节点中。
                     *************************************************************************************************************/	
                    ssd->dram->buffer->write_miss_hit++;

                    if(ssd->dram->buffer->buffer_sector_count>=ssd->dram->buffer->max_buffer_sector)
                    {
                        if (buffer_node==ssd->dram->buffer->buffer_tail)                  /*如果命中的节点是buffer中最后一个节点，交换最后两个节点*/
                        {
                            pt = ssd->dram->buffer->buffer_tail->LRU_link_pre;
                            ssd->dram->buffer->buffer_tail->LRU_link_pre=pt->LRU_link_pre;
                            ssd->dram->buffer->buffer_tail->LRU_link_pre->LRU_link_next=ssd->dram->buffer->buffer_tail;
                            ssd->dram->buffer->buffer_tail->LRU_link_next=pt;
                            pt->LRU_link_next=NULL;
                            pt->LRU_link_pre=ssd->dram->buffer->buffer_tail;
                            ssd->dram->buffer->buffer_tail=pt;

                        }
                        sub_req=NULL;
                        sub_req_state=ssd->dram->buffer->buffer_tail->stored; 
                        sub_req_size=size(ssd->dram->buffer->buffer_tail->stored);
                        sub_req_lpn=ssd->dram->buffer->buffer_tail->group;
                        sub_req=creat_sub_request(ssd,sub_req_lpn,sub_req_size,sub_req_state,req,WRITE);

                        if(req!=NULL)           
                        {

                        }
                        else if(req==NULL)   
                        {
                            sub_req->next_subs=sub->next_subs;
                            sub->next_subs=sub_req;
                        }

                        ssd->dram->buffer->buffer_sector_count=ssd->dram->buffer->buffer_sector_count-sub_req->size;
                        pt = ssd->dram->buffer->buffer_tail;	
                        avlTreeDel(ssd->dram->buffer, (TREE_NODE *) pt);

                        /************************************************************************/
                        /* 改:  挂在了子请求，buffer的节点不应立即删除，						*/
                        /*			需等到写回了之后才能删除									*/
                        /************************************************************************/
                        if(ssd->dram->buffer->buffer_head->LRU_link_next == NULL)
                        {
                            ssd->dram->buffer->buffer_head = NULL;
                            ssd->dram->buffer->buffer_tail = NULL;
                        }else{
                            ssd->dram->buffer->buffer_tail=ssd->dram->buffer->buffer_tail->LRU_link_pre;
                            ssd->dram->buffer->buffer_tail->LRU_link_next=NULL;
                        }
                        pt->LRU_link_next=NULL;
                        pt->LRU_link_pre=NULL;
                        AVL_TREENODE_FREE(ssd->dram->buffer, (TREE_NODE *) pt);
                        pt = NULL;	
                    }

                    /*第二步:将新的lsn加到所述的buffer节点中*/	
                    add_flag=0x00000001<<(lsn%ssd->parameter->subpage_page);

                    if(ssd->dram->buffer->buffer_head!=buffer_node)                      /*如果该buffer节点不在buffer的队首，需要将这个节点提到队首*/
                    {				
                        if(ssd->dram->buffer->buffer_tail==buffer_node)
                        {					
                            buffer_node->LRU_link_pre->LRU_link_next=NULL;					
                            ssd->dram->buffer->buffer_tail=buffer_node->LRU_link_pre;
                        }			
                        else						
                        {			
                            buffer_node->LRU_link_pre->LRU_link_next=buffer_node->LRU_link_next;						
                            buffer_node->LRU_link_next->LRU_link_pre=buffer_node->LRU_link_pre;								
                        }								
                        buffer_node->LRU_link_next=ssd->dram->buffer->buffer_head;			
                        ssd->dram->buffer->buffer_head->LRU_link_pre=buffer_node;
                        buffer_node->LRU_link_pre=NULL;	
                        ssd->dram->buffer->buffer_head=buffer_node;							
                    }					
                    buffer_node->stored=buffer_node->stored|add_flag;		
                    buffer_node->dirty_clean=buffer_node->dirty_clean|add_flag;	
                    ssd->dram->buffer->buffer_sector_count++;
                }			

            }
        }
    }

    return ssd;
}

/**************************************************************************************
 *函数的功能是寻找活跃快，应为每个plane中都只有一个活跃块，只有这个活跃块中才能进行操作
 *The function of the function is to find the active fast, there should be only one active block in each plane, and only this active block can be operated.
 ***************************************************************************************/
Status find_active_block(struct cxl_switch_device_info *ssd, unsigned int channel, unsigned int chip, unsigned int die, unsigned int plane)
{
    unsigned int active_block;
    unsigned int free_page_num=0;
    unsigned int count=0;

    active_block=ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].active_block;
    free_page_num=ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].free_page_num;
    //last_write_page=ssd->channel_head[channel].chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].free_page_num;
    while((free_page_num==0)&&(count<ssd->parameter->block_plane))
    {
        active_block=(active_block+1)%ssd->parameter->block_plane;	
        free_page_num=ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].free_page_num;
        count++;
    }
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].active_block=active_block;
    if(count<ssd->parameter->block_plane)
    {
        return SUCCESS;
    }
    else
    {
        return FAILURE;
    }
}

/**********************************************
 *这个函数的功能是根据channel，chip，size创建子请求
 *The function of this function is to create sub-requests based on lpn, size, state.
 **********************************************/
struct sub_request * creat_sub_request_simple(struct cxl_switch_device_info * ssd, struct addr_info ** p_addr, unsigned int addr_num, unsigned int chip, int size, struct request * req)
{
    struct sub_request* sub=NULL,* sub_r=NULL;
    struct channel_info * p_ch=NULL;
    struct local * loc=NULL;
    unsigned int flag=0;

    sub = (struct sub_request*)malloc(sizeof(struct sub_request));                        /*申请一个子请求的结构*/
    alloc_assert(sub,"sub_request");
    memset(sub,0, sizeof(struct sub_request));

    loc = (struct local*)malloc(sizeof(struct local));                        /*申请一个子请求的结构*/
    alloc_assert(loc,"localation info");
    memset(loc,0, sizeof(struct local));

    if(sub==NULL)
    {
        return NULL;
    }
    sub->location=NULL;
    sub->next_node=NULL;
    sub->next_subs=NULL;
    sub->update=NULL;
    sub->p_addr = p_addr;
    sub->addr_num = addr_num;

    if(req!=NULL)
    {
        sub->next_subs = req->subs; // 把新的sub加到req->subs的队列头
        req->subs = sub;
    }

    /*************************************************************************************
     *读操作
     **************************************************************************************/
    if (req->operation == READ)
    {
        loc->channel = p_addr[0]->cxl_id;
        loc->chip = chip;
    
        sub->location=loc;
        sub->begin_time = ssd->current_time;
        sub->current_state = SR_WAIT;
        sub->current_time=MAX_INT64;
        sub->next_state = SR_CHANNEL_R_CA_TRANSFER;
        sub->next_state_predict_time=MAX_INT64;
        sub->lpn = 0;
        sub->size=size;                     /*需要计算出该子请求的请求大小*/
        sub->idx = ssd->bottom_channel_head[loc->channel].subs_idx;
        ssd->bottom_channel_head[loc->channel].subs_idx++;

        p_ch = &ssd->bottom_channel_head[loc->channel];
        sub->ppn = 0;
        sub->operation = READ;
        // sub->state=(ssd->dram->map->map_entry[lpn].state&0x7fffffff);                                    
  
        if (p_ch->subs_r_tail!=NULL)
        {
            p_ch->subs_r_tail->next_node=sub;
            p_ch->subs_r_tail=sub;
        } 
        else
        {
            p_ch->subs_r_head=sub;
            p_ch->subs_r_tail=sub;
        }
    }
    else if(req->operation == READ_COMPUTE)
    {
        loc->channel = p_addr[0]->cxl_id;
        loc->chip = chip;
    
        sub->location=loc;
        sub->begin_time = ssd->current_time;
        sub->current_state = SR_WAIT;
        sub->current_time=MAX_INT64;
        sub->next_state = SR_CHANNEL_RC_CA_TRANSFER;
        sub->next_state_predict_time=MAX_INT64;
        sub->lpn = 0;
        sub->size=ssd->parameter->subpage_page;                                                             /*需要计算出该子请求的请求大小*/
        sub->idx = ssd->bottom_channel_head[loc->channel].subs_idx;
        ssd->bottom_channel_head[loc->channel].subs_idx++;

        p_ch = &ssd->bottom_channel_head[loc->channel];
        sub->ppn = 0;
        sub->operation = READ_COMPUTE;
        // sub->state=(ssd->dram->map->map_entry[lpn].state&0x7fffffff);     
        sub->input_size = p_addr[0]->input_size;
        sub->output_size = p_addr[0]->output_size;                               
  
        if (p_ch->subs_rc_tail!=NULL)
        {
            p_ch->subs_rc_tail->next_node=sub;
            p_ch->subs_rc_tail=sub;
        } 
        else
        {
            p_ch->subs_rc_head=sub;
            p_ch->subs_rc_tail=sub;
        }
    }
    else if(req->operation == WRITE)
    {
        loc->channel = p_addr[0]->cxl_id;
        loc->chip = chip;
    
        sub->location=loc;
        sub->begin_time = ssd->current_time;
        sub->current_state = SR_WAIT;
        sub->current_time=MAX_INT64;
        sub->next_state = SR_CHANNEL_W_TRANSFER;
        sub->next_state_predict_time=MAX_INT64;
        sub->lpn = 0;
        sub->size=size;                                                               /*需要计算出该子请求的请求大小*/
        sub->idx = ssd->bottom_channel_head[loc->channel].subs_idx;
        ssd->bottom_channel_head[loc->channel].subs_idx++;

        p_ch = &ssd->bottom_channel_head[loc->channel];
        sub->ppn = 0;
        sub->operation = WRITE;
        // sub->state=(ssd->dram->map->map_entry[lpn].state&0x7fffffff);                                    
  
        if (p_ch->subs_w_tail!=NULL)
        {
            p_ch->subs_w_tail->next_node=sub;
            p_ch->subs_w_tail=sub;
        } 
        else
        {
            p_ch->subs_w_tail=sub;
            p_ch->subs_w_head=sub;
        }
    }
    else
    {
        free(sub->location);
        sub->location=NULL;
        free(sub);
        sub=NULL;
        printf("\nERROR ! Unexpected command.\n");
        exit(100);
        return NULL;
    }

    if (req->operation==READ) {
        ssd->read_subreq_count++;
    }
    if (req->operation==WRITE) {
        ssd->write_subreq_count++;
    }

    return sub;
}

/**********************************************
 *这个函数的功能是根据lpn，size，state创建子请求
 *The function of this function is to create sub-requests based on lpn, size, state.
 **********************************************/
struct sub_request * creat_sub_request(struct cxl_switch_device_info * ssd, unsigned int lpn, int size, unsigned int state, struct request * req, unsigned int operation)
{
    struct sub_request* sub=NULL,* sub_r=NULL;
    struct channel_info * p_ch=NULL;
    struct local * loc=NULL;
    unsigned int flag=0;

    sub = (struct sub_request*)malloc(sizeof(struct sub_request));                        /*申请一个子请求的结构*/
    alloc_assert(sub,"sub_request");
    memset(sub,0, sizeof(struct sub_request));

    if(sub==NULL)
    {
        return NULL;
    }
    sub->location=NULL;
    sub->next_node=NULL;
    sub->next_subs=NULL;
    sub->update=NULL;

    if(req!=NULL)
    {
        sub->next_subs = req->subs; // 把新的sub加到req->subs的队列头
        req->subs = sub;
    }

    /*************************************************************************************
     *在读操作的情况下，有一点非常重要就是要预先判断读子请求队列中是否有与这个子请求相同的，
     *有的话，新子请求就不必再执行了，将新的子请求直接赋为完成
     *In the case of a read operation, it is very important to pre-determine whether there is any identical to the sub-request in the read sub-request queue.
     *If so, the new subrequest will not have to be executed, and the new subrequest will be directly assigned to completion.
     **************************************************************************************/
    if (operation == READ)
    {
        loc = find_location(ssd,ssd->dram->map->map_entry[lpn].pn);
        sub->location=loc;
        sub->begin_time = ssd->current_time;
        sub->current_state = SR_WAIT;
        sub->current_time=MAX_INT64;
        sub->next_state = SR_CHANNEL_R_CA_TRANSFER;
        sub->next_state_predict_time=MAX_INT64;
        sub->lpn = lpn;
        sub->size=size;                                                               /*需要计算出该子请求的请求大小*/

        p_ch = &ssd->bottom_channel_head[loc->channel];
        sub->ppn = ssd->dram->map->map_entry[lpn].pn;
        sub->operation = READ;
        sub->state=(ssd->dram->map->map_entry[lpn].state&0x7fffffff);
        sub_r=p_ch->subs_r_head;                                                      /*一以下几行包括flag用于判断该读子请求队列中是否有与这个子请求相同的，有的话，将新的子请求直接赋为完成*/
        flag=0;
        while (sub_r!=NULL)
        {
            if (sub_r->ppn==sub->ppn)
            {
                flag=1;
                break;
            }
            sub_r=sub_r->next_node;
        }
        if (flag==0)
        {
            if (p_ch->subs_r_tail!=NULL)
            {
                p_ch->subs_r_tail->next_node=sub;
                p_ch->subs_r_tail=sub;
            } 
            else
            {
                p_ch->subs_r_head=sub;
                p_ch->subs_r_tail=sub;
            }
        }
        else
        {
            sub->current_state = SR_CHANNEL_R_DATA_TRANSFER;
            sub->current_time=ssd->current_time;
            sub->next_state = SR_COMPLETE;
            sub->next_state_predict_time=ssd->current_time+1000;
            sub->complete_time=ssd->current_time+1000;
        }
    }
    /*************************************************************************************
     *写请求的情况下，就需要利用到函数allocate_location(ssd ,sub)来处理静态分配和动态分配了
     *In the case of a write request, you need to use the function allocate_location(ssd, sub) to handle static allocation and dynamic allocation.
     **************************************************************************************/
    else if(operation == WRITE)
    {
        sub->ppn=0;
        sub->operation = WRITE;
        sub->location=(struct local *)malloc(sizeof(struct local));
        alloc_assert(sub->location,"sub->location");
        memset(sub->location,0, sizeof(struct local));

        sub->current_state=SR_WAIT;
        sub->current_time=ssd->current_time;
        sub->lpn=lpn;
        sub->size=size;
        sub->state=state;
        sub->begin_time=ssd->current_time;

        if (allocate_location(ssd ,sub)==ERROR)
        {
            free(sub->location);
            sub->location=NULL;
            free(sub);
            sub=NULL;
            return NULL;
        }

    }
    else
    {
        free(sub->location);
        sub->location=NULL;
        free(sub);
        sub=NULL;
        printf("\nERROR ! Unexpected command.\n");
        exit(100);
        return NULL;
    }

    // Added by Fadhil to mark every IO that meets GC, and record GC remaining time in the IO
    if (sub->location != NULL &&
        ssd->bottom_channel_head[sub->location->channel].next_state_predict_time != MAX_INT64 &&
        ssd->bottom_channel_head[sub->location->channel].next_state_predict_time != 0 &&
        ssd->bottom_channel_head[sub->location->channel].current_state == CHANNEL_GC &&
        ssd->bottom_channel_head[sub->location->channel].next_state == CHANNEL_IDLE &&
        sub->begin_time >= ssd->bottom_channel_head[sub->location->channel].current_time &&
        sub->begin_time <= ssd->bottom_channel_head[sub->location->channel].next_state_predict_time) {
        req->meet_gc_flag = 1;
        if (ssd->bottom_channel_head[sub->location->channel].next_state_predict_time - sub->begin_time > req->meet_gc_remaining_time) {
            req->meet_gc_remaining_time = ssd->bottom_channel_head[sub->location->channel].next_state_predict_time - sub->begin_time;
        }
    }

    if (operation==READ) {
        ssd->read_subreq_count++;
    }
    if (operation==WRITE) {
        ssd->write_subreq_count++;
    }

    return sub;
}

/*********************************************************************************************
 *专门为读子请求服务的函数
 *1，只有当读子请求的当前状态是SR_R_C_A_TRANSFER
 *2，读子请求的当前状态是SR_COMPLETE或者下一状态是SR_COMPLETE并且下一状态到达的时间比当前时间小
 *============================================================================================
 *a function specifically for reading sub-requests
 *1, only when the current state of the read subrequest is SR_CHANNEL_R_CA_TRANSFER
 *2. The current state of the read subrequest is SR_COMPLETE or the next state is SR_COMPLETE and the next state arrives less than the current time.
 **********************************************************************************************/
Status services_read_complete_in_channel(struct cxl_switch_device_info * ssd) {
    for (int i = 0; i <
                    ssd->parameter->cxl_channel_number; i++) {/*这个循环处理不需要channel的时间(读命令已经到达chip，chip由ready变为busy)，当读请求完成时，将其从channel的队列中取出*/
        // Read subrequest
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_r_head, *pre_sub = NULL;
        while (sub != NULL) {
            if ((sub->current_state == SR_COMPLETE) || ((sub->next_state == SR_COMPLETE) &&
                                                        (sub->next_state_predict_time <=
                                                         ssd->current_time)))  /*if the request is completed, we delete it from read queue */
            {
                if (sub != ssd->bottom_channel_head[i].subs_r_head) // 遇到了没结束的sub request才会触发这种case
                {
                    pre_sub->next_node = sub->next_node;
                } else {
                    if (ssd->bottom_channel_head[i].subs_r_head !=
                        ssd->bottom_channel_head[i].subs_r_tail) // more than one sub request in the queue
                    {
                        ssd->bottom_channel_head[i].subs_r_head = sub->next_node;
                    } else    // only on sub request in the queue
                    {
                        ssd->bottom_channel_head[i].subs_r_head = NULL;
                        ssd->bottom_channel_head[i].subs_r_tail = NULL;
                    }
                }
            }
            // 继续遍历下一个sub request
            pre_sub = sub;
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_read_from_CATransfer_to_CAAnalyze_from_channel_to_cxlctrl(struct cxl_switch_device_info * ssd) {
    for (int i = 0; i < ssd->parameter->cxl_channel_number; i++) {
        /*这个循环处理不需要channel的时间(读命令已经到达chip，chip由ready变为busy)，当读请求完成时，将其从channel的队列中取出*/
        // Read subrequest
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_r_head;
        while (sub != NULL) {
            if ((sub->current_state == SR_CHANNEL_R_CA_TRANSFER) & (sub->next_state_predict_time <= ssd->current_time)) {
                if ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxlctrl->current_state ==
                    CC_IDLE) || 
                    ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxlctrl->next_state ==
                      CC_IDLE) &&
                     (ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxlctrl->next_state_predict_time <=
                      ssd->current_time))) // CXL Controler idle
                {
                    go_one_step_in_cxlctrl(ssd, sub, SR_CXLCTRL_R_CA_ANALYZE);
                }
            }
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_readCompute_complete_in_channel(struct cxl_switch_device_info * ssd) {
    for (int i = 0; i <
                    ssd->parameter->cxl_channel_number; i++) {/*这个循环处理不需要channel的时间(读命令已经到达chip，chip由ready变为busy)，当读请求完成时，将其从channel的队列中取出*/
        // Read compute subrequest
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_rc_head, *pre_sub = NULL;
        while (sub != NULL) {
            if ((sub->current_state == SR_COMPLETE) || ((sub->next_state == SR_COMPLETE) &&
                                                        (sub->next_state_predict_time <=
                                                         ssd->current_time)))  /*if the request is completed, we delete it from read queue */
            {
                if (sub != ssd->bottom_channel_head[i].subs_rc_head) // why this case?
                {
                    pre_sub->next_node = sub->next_node;
                } else {
                    if (ssd->bottom_channel_head[i].subs_rc_head !=
                        ssd->bottom_channel_head[i].subs_rc_tail) // more than one sub request in the queue
                    {
                        ssd->bottom_channel_head[i].subs_rc_head = sub->next_node;
                    } else    // only on sub request in the queue
                    {
                        ssd->bottom_channel_head[i].subs_rc_head = NULL;
                        ssd->bottom_channel_head[i].subs_rc_tail = NULL;
                    }
                }
            }
            // 继续遍历下一个sub request
            pre_sub = sub;
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_readCompute_from_CATransfer_to_CAAnalyze_from_channel_to_cxlctrl(struct cxl_switch_device_info * ssd)
{
    for(int i = 0; i < ssd->parameter->cxl_channel_number; i++) {/*这个循环处理不需要channel的时间(读命令已经到达chip，chip由ready变为busy)，当读请求完成时，将其从channel的队列中取出*/
        // Read compute subrequest
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_rc_head;
        while(sub!=NULL)
        {
            // printf("cxl ctrl state: %d", ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxlctrl->current_state);
            if ((sub->current_state == SR_CHANNEL_RC_CA_TRANSFER) & (sub->next_state_predict_time <= ssd->current_time))
            {   
                printf("cxl ctrl state: %d\n", ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxlctrl->current_state);
                if((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxlctrl->current_state == 
                    CC_IDLE) || 
                    ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxlctrl->next_state ==
                      CC_IDLE) &&
                     (ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxlctrl->next_state_predict_time <=
                      ssd->current_time)))
                {
                    go_one_step_in_cxlctrl(ssd, sub, SR_CXLCTRL_RC_CA_ANALYZE);
                }
            }
            sub = sub->next_node;
        }
    }

    return SUCCESS;
}

Status services_write_complete_in_cxlctrl(struct cxl_switch_device_info * ssd) {
    for(int i = 0;i<ssd->parameter->cxl_channel_number;i++) {/*这个循环处理不需要channel的时间(读命令已经到达chip，chip由ready变为busy)，当读请求完成时，将其从channel的队列中取出*/
        // Write subrequest
        struct sub_request *sub=ssd->bottom_channel_head[i].subs_w_head, *pre_sub = NULL;
        while(sub!=NULL) {
            if((sub->current_state==SR_COMPLETE)||((sub->next_state==SR_COMPLETE)&&(sub->next_state_predict_time<=ssd->current_time)))  /*if the request is completed, we delete it from read queue */
            {	
                if(sub!=ssd->bottom_channel_head[i].subs_w_head) // 遇到了没结束的sub request才会触发这种case
                {
                    pre_sub->next_node=sub->next_node;
                }			
                else					
                {	
                    if (ssd->bottom_channel_head[i].subs_w_head != ssd->bottom_channel_head[i].subs_w_tail) // more than one sub request in the queue
                    {
                        ssd->bottom_channel_head[i].subs_w_head=sub->next_node;
                    } 
                    else    // only on sub request in the queue
                    {
                        ssd->bottom_channel_head[i].subs_w_head=NULL;
                        ssd->bottom_channel_head[i].subs_w_tail=NULL;
                    }							
                }			
            }
            pre_sub = sub;
            sub=sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_read_from_dataTransfer_to_dataTransfer_from_cxlctrl_to_channel(struct cxl_switch_device_info * ssd, unsigned int channel, unsigned int * channel_busy_flag, int * gpu_channel_states) {
    if (*channel_busy_flag == 1){
        return SUCCESS;
    }
    struct sub_request *sub = ssd->bottom_channel_head[channel].subs_r_head;
    while(sub != NULL) {
        if(sub->current_state == SR_CXLCTRL_R_READ && (sub->next_state_predict_time <= ssd->current_time)) {
            if(gpu_channel_states[sub->p_addr[0]->gpu_id] == 0) {
                go_one_step_in_channel(ssd, sub, SR_CHANNEL_R_DATA_TRANSFER);
                *channel_busy_flag = 1;
                gpu_channel_states[sub->p_addr[0]->gpu_id] = 1;
                break;
            }
        }
        sub = sub->next_node;
    }
    return SUCCESS;
}

Status services_readCompute_from_dataTransfer_to_dataTransfer_from_cxlctrl_to_channel(struct cxl_switch_device_info * ssd, unsigned int channel, unsigned int * channel_busy_flag, int * gpu_channel_states) {
    if (*channel_busy_flag == 1){
        return SUCCESS;
    }
    struct sub_request *sub = ssd->bottom_channel_head[channel].subs_rc_head;
    while(sub != NULL) {
        if(sub->current_state == SR_CXLCTRL_RC_DATA_TRANSFER && (sub->next_state_predict_time <= ssd->current_time)) {
            if(gpu_channel_states[sub->p_addr[0]->gpu_id] == 0) {
                go_one_step_in_channel(ssd, sub, SR_CHANNEL_RC_DATA_TRANSFER);
                *channel_busy_flag = 1;
                gpu_channel_states[sub->p_addr[0]->gpu_id] = 1;
                break;
            }
        }
        sub = sub->next_node;
    }

    return SUCCESS;
}

Status services_read_from_wait_to_CATransfer_in_channel(struct cxl_switch_device_info * ssd, unsigned int channel, unsigned int * channel_busy_flag, int * gpu_channel_states) {
    if (*channel_busy_flag == 1){
        return SUCCESS;
    }
    // Read request
    int buf_used_size;
    int buf_size; 
    struct sub_request *sub = ssd->bottom_channel_head[channel].subs_r_head;
    while (sub != NULL)  /*if there are read requests in queue, send one of them to target chip*/
    {
        unsigned int gpu_id = sub->p_addr[0]->gpu_id;
        if (gpu_channel_states[sub->p_addr[0]->gpu_id] == 0 &&
            (ssd->top_channel_head[gpu_id].current_state == CHANNEL_IDLE) ||
            (ssd->top_channel_head[gpu_id].next_state == CHANNEL_IDLE &&
             ssd->top_channel_head[gpu_id].next_state_predict_time <=
             ssd->current_time)) // channel idle 是才能触发的操作，包括在channel开始CA传输/data传输
        {
            if (sub->current_state == SR_WAIT) {
                buf_used_size = ssd->bottom_channel_head[channel].cxl_device->chip_head[sub->location->chip].cxlctrl->buf_used_size;
                buf_size = ssd->bottom_channel_head[channel].cxl_device->chip_head[sub->location->chip].cxlctrl->buf_size;
                if (buf_used_size + ssd->parameter->sub_req_inst_size <= buf_size){
                    go_one_step_in_channel(ssd, sub, SR_CHANNEL_R_CA_TRANSFER);
                    *channel_busy_flag = 1;
                    gpu_channel_states[sub->p_addr[0]->gpu_id] = 1;  
                }
            }
        }
        sub = sub->next_node;
    }
    return SUCCESS;
}

Status services_readCompute_from_wait_to_CATransfer_in_channel(struct cxl_switch_device_info * ssd, unsigned int channel, unsigned int * channel_busy_flag, int * gpu_channel_states) {
    if (*channel_busy_flag == 1){
        return SUCCESS;
    }
    // Read Compute request
    int buf_used_size;
    int buf_size; 
    struct sub_request *sub = ssd->bottom_channel_head[channel].subs_rc_head;
    while (sub != NULL)  /*if there are read requests in queue, send one of them to target chip*/
    {
        if (*channel_busy_flag == 1)      // channel 优先分配给read request;
        {
            break;
        }

        unsigned int gpu_id = sub->p_addr[0]->gpu_id;
        if (gpu_channel_states[sub->p_addr[0]->gpu_id] == 0 &&
            (ssd->top_channel_head[gpu_id].current_state == CHANNEL_IDLE) ||
            (ssd->top_channel_head[gpu_id].next_state == CHANNEL_IDLE &&
             ssd->top_channel_head[gpu_id].next_state_predict_time <=
             ssd->current_time)) // channel idle 是才能触发的操作，包括在channel开始CA传输/data传输
        {
            if (sub->current_state == SR_WAIT) {
                buf_used_size = ssd->bottom_channel_head[channel].cxl_device->chip_head[sub->location->chip].cxlctrl->buf_used_size;
                buf_size = ssd->bottom_channel_head[channel].cxl_device->chip_head[sub->location->chip].cxlctrl->buf_size;
                if (buf_used_size + ssd->parameter->sub_req_inst_size <= buf_size){
                    go_one_step_in_channel(ssd, sub, SR_CHANNEL_RC_CA_TRANSFER);
                    *channel_busy_flag = 1;
                    gpu_channel_states[sub->p_addr[0]->gpu_id] = 1;   
                }           
            }
        }
        sub = sub->next_node;
    }
    return SUCCESS;
}

Status services_write_from_wait_to_dataAndCATransfer_in_channel(struct cxl_switch_device_info * ssd, unsigned int channel, unsigned int * channel_busy_flag, int * gpu_channel_states) {
    if (*channel_busy_flag == 1){
        return SUCCESS;
    }
    int buf_used_size;
    int buf_size; 
    struct sub_request *sub=ssd->bottom_channel_head[channel].subs_w_head;
    while(sub != NULL)
    {
        unsigned int gpu_id = sub->p_addr[0]->gpu_id;
        if(gpu_channel_states[sub->p_addr[0]->gpu_id]==0 && (ssd->top_channel_head[gpu_id].current_state == CHANNEL_IDLE) || (ssd->top_channel_head[gpu_id].next_state == CHANNEL_IDLE && ssd->top_channel_head[gpu_id].next_state_predict_time <= ssd->current_time)) // channel idle 是才能触发的操作，包括在channel开始CA传输/data传输
        {
            if(sub->current_state==SR_WAIT){
                buf_used_size = ssd->bottom_channel_head[channel].cxl_device->chip_head[sub->location->chip].cxlctrl->buf_used_size;
                buf_size = ssd->bottom_channel_head[channel].cxl_device->chip_head[sub->location->chip].cxlctrl->buf_size;
                if (buf_used_size + ssd->parameter->sub_req_inst_size <= buf_size){	                         
                    go_one_step_in_channel(ssd, sub, SR_CHANNEL_W_TRANSFER);
                    *channel_busy_flag=1;
                    gpu_channel_states[sub->p_addr[0]->gpu_id] = 1; 
                }                                                
            }
        }
        sub=sub->next_node;								
    }

    return SUCCESS;
}

Status services_write_from_dataAndCATransfer_to_CAAnalyze_from_channel_to_cxlctrl(struct cxl_switch_device_info * ssd) {
    for (int i = 0; i < ssd->parameter->cxl_channel_number; i++) {
        /*这个循环处理不需要channel的时间(读命令已经到达chip，chip由ready变为busy)，当读请求完成时，将其从channel的队列中取出*/
        // Write subrequest
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_w_head;
        while (sub != NULL) {
            if ((sub->current_state == SR_CHANNEL_W_TRANSFER) & (sub->next_state_predict_time <= ssd->current_time)) {
                if ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxlctrl->current_state ==
                    CC_IDLE) || 
                    ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxlctrl->next_state ==
                      CC_IDLE) &&
                     (ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxlctrl->next_state_predict_time <=
                      ssd->current_time))) // FIXME: 介质读通路没有被读计算请求占用
                {
                    go_one_step_in_cxlctrl(ssd, sub, SR_CXLCTRL_W_CA_ANALYZE);
                }
            }
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_read_from_CAAnalyze_to_CATransfer_in_cxlctrl(struct cxl_switch_device_info * ssd) {
    for(int i = 0; i < ssd->parameter->cxl_channel_number; i++) {
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_r_head;
        while(sub != NULL) {
            if((sub->current_state == SR_CXLCTRL_R_CA_ANALYZE) & (sub->next_state_predict_time <= ssd->current_time)) {
                go_one_step_in_cxlctrl(ssd, sub, SR_CXLCTRL_R_CA_TRANSFER);
            }
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_read_from_CATransfer_to_dataTransfer_in_cxlctrl(struct cxl_switch_device_info * ssd) {
    for(int i = 0; i < ssd->parameter->cxl_channel_number; i++) {
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_r_head;
        while(sub != NULL) {
            if((sub->current_state == SR_CXLCTRL_R_CA_TRANSFER) & (sub->next_state_predict_time <= ssd->current_time)) {
                if ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxldram->current_state ==
                    DRAM_IDLE) || 
                    ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxldram->next_state ==
                      DRAM_IDLE) &&
                     (ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxldram->next_state_predict_time <=
                      ssd->current_time))) // FIXME: 介质读通路没有被读计算请求占用
                {
                    go_one_step_in_cxlctrl(ssd, sub, SR_CXLCTRL_R_READ);
                }
            }
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_write_from_CAAnalyze_to_write_in_cxlctrl(struct cxl_switch_device_info * ssd) {
    for (int i = 0; i < ssd->parameter->cxl_channel_number; i++) {
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_w_head;
        while (sub != NULL) {
            if ((sub->current_state == SR_CXLCTRL_W_CA_ANALYZE) & (sub->next_state_predict_time <= ssd->current_time)) {
                if ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxldram->current_state ==
                    DRAM_IDLE) ||
                    ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxldram->next_state ==
                      DRAM_IDLE) &&
                     (ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxldram->next_state_predict_time <=
                      ssd->current_time))) // FIXME: 介质读通路没有被读计算请求占用
                {
                    go_one_step_in_cxlctrl(ssd, sub, SR_CXLCTRL_W_WRITE);
                }
            }
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_write_from_write_to_confirm_in_cxlctrl(struct cxl_switch_device_info * ssd) {
    for (int i = 0; i < ssd->parameter->cxl_channel_number; i++) {
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_w_head;
        while (sub != NULL) {
            if ((sub->current_state == SR_CXLCTRL_W_WRITE) & (sub->next_state_predict_time <= ssd->current_time)) {
                go_one_step_in_cxlctrl(ssd, sub, SR_CXLCTRL_W_CONFIRM);
            }
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_readCompute_from_CAAnalyze_to_CATransfer_in_cxlctrl(struct cxl_switch_device_info * ssd) {
    for(int i = 0; i < ssd->parameter->cxl_channel_number; i++) {
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_rc_head;
        while(sub != NULL) {
            if((sub->current_state == SR_CXLCTRL_RC_CA_ANALYZE) & (sub->next_state_predict_time <= ssd->current_time)) {
                go_one_step_in_cxlctrl(ssd, sub, SR_CXLCTRL_RC_CA_TRANSFER);
            }
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_readCompute_from_CATransfer_to_read_in_cxlctrl(struct cxl_switch_device_info * ssd) {
    for(int i = 0; i < ssd->parameter->cxl_channel_number; i++) {
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_rc_head;
        while(sub != NULL) {
            if((sub->current_state == SR_CXLCTRL_RC_CA_TRANSFER) & (sub->next_state_predict_time <= ssd->current_time)) {
                if ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxldram->current_state ==
                    DRAM_IDLE) ||
                    ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxldram->next_state ==
                      DRAM_IDLE) &&
                     (ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].cxldram->next_state_predict_time <=
                      ssd->current_time)))
                {
                    go_one_step_in_cxlctrl(ssd, sub, SR_CXLCTRL_RC_READ);
                }
            }
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_readCompute_from_read_to_compute_in_cxlctrl(struct cxl_switch_device_info * ssd) {
    for(int i = 0; i < ssd->parameter->cxl_channel_number; i++) {/*这个循环处理不需要channel的时间(读命令已经到达chip，chip由ready变为busy)，当读请求完成时，将其从channel的队列中取出*/
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_rc_head;
        while(sub != NULL) {
            if((sub->current_state==SR_CXLCTRL_RC_READ) & (sub->next_state_predict_time<=ssd->current_time)) {
                if ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].pe->current_state ==
                    PE_IDLE) || 
                    ((ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].pe->next_state ==
                      PE_IDLE) &&
                     (ssd->bottom_channel_head[sub->location->channel].cxl_device->chip_head[sub->location->chip].pe->next_state_predict_time <=
                      ssd->current_time)))
                {
                go_one_step_in_cxlctrl(ssd, sub, SR_CXLCTRL_RC_COMPUTE);
                }
            }
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

Status services_readCompute_from_compute_to_dataTransfer_in_cxlctrl(struct cxl_switch_device_info * ssd) {
    for(int i = 0; i < ssd->parameter->cxl_channel_number; i++) {
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_rc_head;
        while(sub != NULL) {
            if((sub->current_state == SR_CXLCTRL_RC_COMPUTE) & (sub->next_state_predict_time <= ssd->current_time)) {
                go_one_step_in_cxlctrl(ssd, sub, SR_CXLCTRL_RC_DATA_TRANSFER);
            }
            sub = sub->next_node;
        }
    }
    return SUCCESS;
}

/**************************************************************************
 *这个函数非常重要，读子请求的状态转变，以及时间的计算都通过这个函数来处理
 *还有写子请求的执行普通命令时的状态，以及时间的计算也是通过这个函数来处理的
 *This function is very important, the state transition of the read subrequest, and the calculation of the time are handled by this function.
 *There is also the state when the ordinary command is executed to write the sub-request, and the calculation of the time is also handled by this function.
 ****************************************************************************/
Status go_one_step_in_channel(struct cxl_switch_device_info * ssd, struct sub_request * sub, unsigned int aim_state) {
    if(sub==NULL) return ERROR;
    struct local *location = sub->location; struct addr_info *addr = sub->p_addr[0];
    /***************************************************************************************************
     *处理普通命令时，读子请求的目标状态分为以下几种情况SR_R_READ，SR_CHANNEL_R_CA_TRANSFER，SR_CHANNEL_R_DATA_TRANSFER
     *写子请求的目标状态只有SR_W_TRANSFER
     ****************************************************************************************************/
    switch(aim_state) {
        case SR_CHANNEL_R_CA_TRANSFER:
            {
                /*******************************************************************************************************
                 *目标状态是命令地址传输时，sub的下一个状态就是SR_R_READ
                 *这个状态与channel，chip有关，所以要修改channel，chip的状态分别为CHANNEL_C_A_TRANSFER，CHIP_CA_TRANSFER
                 *下一状态分别为CHANNEL_IDLE，CHIP_READ_BUSY
                 *******************************************************************************************************/
                sub->current_time=ssd->current_time;
                sub->current_state=SR_CHANNEL_R_CA_TRANSFER;
                sub->next_state=SR_CXLCTRL_R_CA_ANALYZE;
                sub->next_state_predict_time=ssd->current_time + ssd->parameter->time_characteristics.tCMDCXL
                    + ssd->parameter->time_characteristics.L_CXL_switch;
                sub->begin_time=ssd->current_time;

                ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].subs_idx=sub->idx;
                ssd->read_count++;
                ssd->in_read_size+=ssd->parameter->subpage_page;

                ssd->bottom_channel_head[location->channel].current_state=CHANNEL_CA_TRANSFER;
                ssd->bottom_channel_head[location->channel].current_time=ssd->current_time;
                ssd->bottom_channel_head[location->channel].next_state=CHANNEL_IDLE;
                ssd->bottom_channel_head[location->channel].next_state_predict_time=sub->next_state_predict_time;
                ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->buf_used_size +=
                ssd->parameter->sub_req_inst_size;

                ssd->top_channel_head[addr->gpu_id].current_state=CHANNEL_CA_TRANSFER;
                ssd->top_channel_head[addr->gpu_id].current_time=ssd->current_time;
                ssd->top_channel_head[addr->gpu_id].next_state=CHANNEL_IDLE;
                ssd->top_channel_head[addr->gpu_id].next_state_predict_time=sub->next_state_predict_time;

                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_state=CHIP_IDLE;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_time=ssd->current_time;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state=CHIP_IDLE;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state_predict_time=sub->next_state_predict_time;

                break;
            }
        case SR_CHANNEL_R_DATA_TRANSFER:
            {
                /**************************************************************************************************************
                 *目标状态是数据传输时，sub的下一个状态就是完成状态SR_COMPLETE
                 *这个状态的处理也与channel，chip有关，所以channel，chip的当前状态变为CHANNEL_DATA_TRANSFER，CHIP_DATA_TRANSFER
                 *下一个状态分别为CHANNEL_IDLE，CHIP_IDLE。
                 ***************************************************************************************************************/
                sub->current_time=ssd->current_time;
                sub->current_state=SR_CHANNEL_R_DATA_TRANSFER;
                sub->next_state=SR_COMPLETE;
                uint64_t read_data_size = 0; 
                for(int i = 0; i < sub->addr_num; i++) {
                    read_data_size += sub->p_addr[i]->size;
                }
                sub->next_state_predict_time=ssd->current_time + read_data_size / ssd->parameter->cxl_bandwidth
                    + ssd->parameter->time_characteristics.L_CXL_switch;
                if (sub->next_state_predict_time<=ssd->current_time){
                    sub->next_state_predict_time = ssd->current_time + 1;
                }
                sub->complete_time=sub->next_state_predict_time;

                ssd->bottom_channel_head[location->channel].current_state=CHANNEL_DATA_TRANSFER;
                ssd->bottom_channel_head[location->channel].current_time=ssd->current_time;
                ssd->bottom_channel_head[location->channel].next_state=CHANNEL_IDLE;
                ssd->bottom_channel_head[location->channel].next_state_predict_time=sub->next_state_predict_time;
                ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->buf_used_size -=
                ssd->parameter->sub_req_inst_size;

                ssd->top_channel_head[addr->gpu_id].current_state=CHANNEL_DATA_TRANSFER;
                ssd->top_channel_head[addr->gpu_id].current_time=ssd->current_time;
                ssd->top_channel_head[addr->gpu_id].next_state=CHANNEL_IDLE;
                ssd->top_channel_head[addr->gpu_id].next_state_predict_time=sub->next_state_predict_time;

                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_state=CHIP_IDLE;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_time=ssd->current_time;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state=CHIP_IDLE;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state_predict_time=sub->next_state_predict_time;

                ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].subs_idx=-1;

                break;
            }
        case SR_CHANNEL_RC_CA_TRANSFER:
            {
                /*******************************************************************************************************
                 *目标状态是命令地址传输时，sub的下一个状态就是SR_R_READ
                 *这个状态与channel，chip有关，所以要修改channel，chip的状态分别为CHANNEL_C_A_TRANSFER，CHIP_CA_TRANSFER
                 *下一状态分别为CHANNEL_IDLE，CHIP_READ_BUSY
                 *******************************************************************************************************/
                sub->current_time=ssd->current_time;
                sub->current_state=SR_CHANNEL_RC_CA_TRANSFER;
                sub->next_state=SR_CXLCTRL_RC_CA_ANALYZE;
                sub->next_state_predict_time = ssd->current_time + ssd->parameter->time_characteristics.tCMDCXL +
                                            sub->input_size / ssd->parameter->cxl_bandwidth
                                            + ssd->parameter->time_characteristics.L_CXL_switch;
                if (sub->next_state_predict_time<=ssd->current_time){
                    sub->next_state_predict_time=ssd->current_time + 1;
                }
                sub->begin_time=ssd->current_time;

                ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->subs_idx=sub->idx;
                ssd->read_count++;
                ssd->in_read_size+=ssd->parameter->subpage_page; // TODO

                ssd->bottom_channel_head[location->channel].current_state=CHANNEL_CA_TRANSFER;
                ssd->bottom_channel_head[location->channel].current_time=ssd->current_time;
                ssd->bottom_channel_head[location->channel].next_state=CHANNEL_IDLE;
                ssd->bottom_channel_head[location->channel].next_state_predict_time=sub->next_state_predict_time;
                ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->buf_used_size +=
                ssd->parameter->sub_req_inst_size;

                ssd->top_channel_head[addr->gpu_id].current_state=CHANNEL_CA_TRANSFER;
                ssd->top_channel_head[addr->gpu_id].current_time=ssd->current_time;
                ssd->top_channel_head[addr->gpu_id].next_state=CHANNEL_IDLE;
                ssd->top_channel_head[addr->gpu_id].next_state_predict_time=sub->next_state_predict_time;

                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->current_state=PE_IDLE;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->current_time=ssd->current_time;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->next_state=PE_IDLE;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->next_state_predict_time=sub->next_state_predict_time;

                break;
            }
        case SR_CHANNEL_RC_DATA_TRANSFER:
            {
                /**************************************************************************************************************
                 *目标状态是数据传输时，sub的下一个状态就是完成状态SR_COMPLETE
                 *这个状态的处理也与channel，chip有关，所以channel，chip的当前状态变为CHANNEL_DATA_TRANSFER，CHIP_DATA_TRANSFER
                 *下一个状态分别为CHANNEL_IDLE，CHIP_IDLE。
                 ***************************************************************************************************************/
                sub->current_time=ssd->current_time;
                sub->current_state=SR_CHANNEL_RC_DATA_TRANSFER;
                sub->next_state=SR_COMPLETE;
                sub->next_state_predict_time = ssd->current_time + sub->output_size / ssd->parameter->cxl_bandwidth
                    + ssd->parameter->time_characteristics.L_CXL_switch;
                if (sub->next_state_predict_time <= ssd->current_time){
                    sub->next_state_predict_time = ssd->current_time + 1;
                }
                sub->complete_time=sub->next_state_predict_time;

                ssd->bottom_channel_head[location->channel].current_state=CHANNEL_DATA_TRANSFER;
                ssd->bottom_channel_head[location->channel].current_time=ssd->current_time;
                ssd->bottom_channel_head[location->channel].next_state=CHANNEL_IDLE;
                ssd->bottom_channel_head[location->channel].next_state_predict_time=sub->next_state_predict_time;
                ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->buf_used_size -=
                ssd->parameter->sub_req_inst_size;

                ssd->top_channel_head[addr->gpu_id].current_state=CHANNEL_DATA_TRANSFER;
                ssd->top_channel_head[addr->gpu_id].current_time=ssd->current_time;
                ssd->top_channel_head[addr->gpu_id].next_state=CHANNEL_IDLE;
                ssd->top_channel_head[addr->gpu_id].next_state_predict_time=sub->next_state_predict_time;

                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->current_state=PE_IDLE;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->current_time=ssd->current_time;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->next_state=PE_IDLE;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->next_state_predict_time=sub->next_state_predict_time;

                ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->subs_idx=-1;

                break;
            }
        case SR_CHANNEL_W_TRANSFER:
            {
                /******************************************************************************************************
                 *这是处理写子请求时，状态的转变以及时间的计算
                 *虽然写子请求的处理状态也像读子请求那么多，但是写请求都是从上往plane中传输数据
                 *这样就可以把几个状态当一个状态来处理，就当成SR_W_TRANSFER这个状态来处理，sub的下一个状态就是完成状态了
                 *此时channel，chip的当前状态变为CHANNEL_TRANSFER，CHIP_WRITE_BUSY
                 *下一个状态变为CHANNEL_IDLE，CHIP_IDLE
                 *******************************************************************************************************/
                sub->current_time=ssd->current_time;
                sub->current_state=SR_CHANNEL_W_TRANSFER;
                uint64_t write_data_size = 0; 
                for(int i = 0; i < sub->addr_num; i++) {
                    write_data_size += sub->p_addr[i]->size;
                }
                sub->next_state_predict_time=ssd->current_time + ssd->parameter->time_characteristics.tCMDCXL
                    + write_data_size / ssd->parameter->cxl_bandwidth
                    + ssd->parameter->time_characteristics.L_CXL_switch;
                if (sub->next_state_predict_time<=ssd->current_time){
                    sub->next_state_predict_time=ssd->current_time+1;
                }
                sub->next_state = SR_CXLCTRL_W_CA_ANALYZE;
                sub->begin_time=ssd->current_time;

                ssd->bottom_channel_head[location->channel].current_state= CHANNEL_CA_TRANSFER + CHANNEL_DATA_TRANSFER;
                ssd->bottom_channel_head[location->channel].current_time=ssd->current_time;
                ssd->bottom_channel_head[location->channel].next_state=CHANNEL_IDLE;
                ssd->bottom_channel_head[location->channel].next_state_predict_time=sub->next_state_predict_time;
                ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->buf_used_size +=
                ssd->parameter->sub_req_inst_size;

                ssd->top_channel_head[addr->gpu_id].current_state= CHANNEL_CA_TRANSFER + CHANNEL_DATA_TRANSFER;
                ssd->top_channel_head[addr->gpu_id].current_time=ssd->current_time;
                ssd->top_channel_head[addr->gpu_id].next_state=CHANNEL_IDLE;
                ssd->top_channel_head[addr->gpu_id].next_state_predict_time=sub->next_state_predict_time;

                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_state=CHIP_IDLE;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_time=ssd->current_time;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state=CHIP_IDLE;
                // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state_predict_time=sub->next_state_predict_time;

                break;
            }
        default :  return ERROR;
    }
    return SUCCESS;
}

Status go_one_step_in_cxlctrl(struct cxl_switch_device_info * ssd, struct sub_request * sub, unsigned int aim_state) {
    if(sub==NULL) return ERROR;
    struct local *location = sub->location;
    /***************************************************************************************************
     *处理普通命令时，读子请求的目标状态分为以下几种情况SR_R_READ，SR_CHANNEL_R_CA_TRANSFER，SR_CHANNEL_R_DATA_TRANSFER
     *写子请求的目标状态只有SR_W_TRANSFER
     ****************************************************************************************************/
    switch(aim_state) {
        case SR_CXLCTRL_R_CA_ANALYZE:
        {
            /*****************************************************************************************************
             *这个目标状态是指flash处于读数据的状态，sub的下一状态就应该是传送数据SR_R_DATA_TRANSFER
             *这时与channel无关，只与chip有关所以要修改chip的状态为CHIP_READ_BUSY，下一个状态就是CHIP_DATA_TRANSFER
             ******************************************************************************************************/
            sub->current_time=ssd->current_time;
            sub->current_state=SR_CXLCTRL_R_CA_ANALYZE;
            sub->next_state=SR_CXLCTRL_R_CA_TRANSFER;
            sub->next_state_predict_time=ssd->current_time + ssd->parameter->time_characteristics.tANALYZE;

            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_state=CHIP_IDLE;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_time=ssd->current_time;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state=CHIP_CA_TRANSFER;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state_predict_time=sub->next_state_predict_time;

            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->current_state = CC_ANALYZE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->current_time=ssd->current_time;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->next_state = CC_CA_TRANSFER;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->next_state_predict_time=sub->next_state_predict_time;


            break;
        }
        case SR_CXLCTRL_R_CA_TRANSFER:
        {
            sub->current_time=ssd->current_time;
            sub->current_state=SR_CXLCTRL_R_CA_TRANSFER;
            sub->next_state=SR_CXLCTRL_R_READ;
            sub->next_state_predict_time=ssd->current_time + ssd->parameter->time_characteristics.tCMDDRAM;

            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_state=CHIP_CA_TRANSFER;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_time=ssd->current_time;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state=CHIP_READ_BUSY;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state_predict_time=sub->next_state_predict_time;

            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->current_state = CC_CA_TRANSFER;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->current_time=ssd->current_time;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->next_state = CC_IDLE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->next_state_predict_time=sub->next_state_predict_time;

            break;
        }
        case SR_CXLCTRL_R_READ:
        {
            sub->current_time=ssd->current_time;
            sub->current_state=SR_CXLCTRL_R_READ;
            sub->next_state=SR_CHANNEL_R_DATA_TRANSFER;
            uint64_t read_data_size = 0; 
            for(int i = 0; i < sub->addr_num; i++){
                read_data_size += sub->p_addr[i]->size;
            }
            sub->next_state_predict_time = ssd->current_time + ssd->parameter->time_characteristics.tDRAMRL  + 
                                            read_data_size / ssd->parameter->dram_bandwidth;
            if (sub->next_state_predict_time<=ssd->current_time)  sub->next_state_predict_time=ssd->current_time+1;

            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_state=CHIP_READ_BUSY;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_time=ssd->current_time;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state=CHIP_IDLE;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state_predict_time=sub->next_state_predict_time;

            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->current_state = DRAM_READ;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->current_time = ssd->current_time;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->next_state = DRAM_IDLE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->next_state_predict_time = sub->next_state_predict_time;

            break;
        }
        case SR_CXLCTRL_W_CA_ANALYZE:
        {
            sub->current_time=ssd->current_time;
            sub->current_state=SR_CXLCTRL_W_CA_ANALYZE;
            sub->next_state=SR_CXLCTRL_W_WRITE;
            sub->next_state_predict_time=ssd->current_time + ssd->parameter->time_characteristics.tANALYZE;

            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_state=CHIP_IDLE;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_time=ssd->current_time;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state=CHIP_WRITE_BUSY;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state_predict_time=sub->next_state_predict_time;

            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->current_state = CC_ANALYZE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->current_time = ssd->current_time;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->next_state = CC_IDLE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->next_state_predict_time = sub->next_state_predict_time;

            break;
        }
        case SR_CXLCTRL_W_WRITE:
        {
            sub->current_time=ssd->current_time;
            sub->current_state=SR_CXLCTRL_W_WRITE;
            sub->next_state=SR_CXLCTRL_W_CONFIRM;
            uint64_t write_data_size = 0; 
            for(int i = 0; i < sub->addr_num; i++) {
                write_data_size += sub->p_addr[i]->size;
            }
            sub->next_state_predict_time=ssd->current_time + write_data_size / ssd->parameter->dram_bandwidth; // FIXME：使用 dram_bandwidth 来代表写速度是否合适？
            if (sub->next_state_predict_time<=ssd->current_time)  sub->next_state_predict_time=ssd->current_time+1;

            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_state=CHIP_WRITE_BUSY;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_time=ssd->current_time;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state=CHANNEL_CA_TRANSFER;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state_predict_time=sub->next_state_predict_time;

            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->current_state = DRAM_WRITE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->current_time=ssd->current_time;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->next_state = DRAM_IDLE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->next_state_predict_time=sub->next_state_predict_time;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->buf_used_size -= ssd->parameter->sub_req_inst_size;

            break;
        }
        case SR_CXLCTRL_W_CONFIRM:
        {
            sub->current_time=ssd->current_time;
            sub->current_state=SR_CXLCTRL_W_CONFIRM;
            sub->next_state=SR_COMPLETE;
            sub->next_state_predict_time=ssd->current_time + ssd->parameter->time_characteristics.tCMDDRAM;
            sub->complete_time=sub->next_state_predict_time; // DONE

            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_state=CHIP_CA_TRANSFER;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].current_time=ssd->current_time;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state=CHIP_IDLE;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].next_state_predict_time=sub->next_state_predict_time;

            break;
        }
        case SR_CXLCTRL_RC_CA_ANALYZE:
        {
            /*****************************************************************************************************
             *这个目标状态是指flash处于读数据的状态，sub的下一状态就应该是传送数据SR_R_DATA_TRANSFER
             *这时与channel无关，只与chip有关所以要修改chip的状态为CHIP_READ_BUSY，下一个状态就是CHIP_DATA_TRANSFER
             ******************************************************************************************************/
            sub->current_time=ssd->current_time;
            sub->current_state=SR_CXLCTRL_RC_CA_ANALYZE;
            sub->next_state=SR_CXLCTRL_RC_CA_TRANSFER;
            sub->next_state_predict_time=ssd->current_time+ssd->parameter->time_characteristics.tANALYZE
                + ssd->parameter->time_characteristics.L_read_compute_cmd;

            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->current_state=CC_ANALYZE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->current_time=ssd->current_time;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->next_state=CC_CA_TRANSFER;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->next_state_predict_time=sub->next_state_predict_time;

            break;
        }
        case SR_CXLCTRL_RC_CA_TRANSFER:
        {
            ssd->current_time=ssd->current_time;
            sub->current_state=SR_CXLCTRL_RC_CA_TRANSFER;
            sub->next_state=SR_CXLCTRL_RC_READ;
            sub->next_state_predict_time=ssd->current_time+ssd->parameter->time_characteristics.tCMDDRAM;

            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->current_state=CC_CA_TRANSFER;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->current_time=ssd->current_time;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->next_state=CC_IDLE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxlctrl->next_state_predict_time=sub->next_state_predict_time;

            break;
        }
        case SR_CXLCTRL_RC_READ:
        {
            sub->current_time=ssd->current_time;
            sub->current_state=SR_CXLCTRL_RC_READ;
            sub->next_state=SR_CXLCTRL_RC_COMPUTE;
            uint64_t read_data_size = 0; 
            for(int i = 0; i < sub->addr_num; i++) {
                read_data_size += sub->p_addr[i]->size;
            }
            sub->next_state_predict_time = ssd->current_time+ssd->parameter->time_characteristics.tDRAMRL + 
                                            read_data_size / ssd->parameter->dram_bandwidth;
            if (sub->next_state_predict_time<=ssd->current_time) {
                sub->next_state_predict_time=ssd->current_time+1;
            }

            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->current_state = DRAM_READ;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->current_time = ssd->current_time;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->next_state = DRAM_IDLE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].cxldram->next_state_predict_time = sub->next_state_predict_time;

            break;
        }
        case SR_CXLCTRL_RC_COMPUTE:
        {
            sub->current_time=ssd->current_time;
            sub->current_state=SR_CXLCTRL_RC_COMPUTE;
            sub->next_state=SR_CXLCTRL_RC_DATA_TRANSFER;
            if (ssd->parameter->ndp_compute_model == 1) {
                /* ops-based PE model: 2 ops (MAC) per element read from
                 * device DRAM; chip_computing_power is GOPS == ops/ns */
                int64_t read_data_size = 0;
                for (int i = 0; i < sub->addr_num; i++) {
                    read_data_size += sub->p_addr[i]->size;
                }
                int64_t elements = read_data_size /
                    (ssd->parameter->model_characteristics.data_type / 8);
                sub->next_state_predict_time = ssd->current_time +
                    2 * elements / ssd->parameter->chip_computing_power;
            } else {
                sub->next_state_predict_time = ssd->current_time + ((sub->addr_num-1) * sub->p_addr[0]->size / (ssd->parameter->model_characteristics.data_type / 8) ) / ssd->parameter->chip_computing_power;//操作数除以算力
            }
            if (sub->next_state_predict_time<=ssd->current_time)  sub->next_state_predict_time=ssd->current_time+1;

            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->current_state=PE_COMPUTE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->current_time=ssd->current_time;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->next_state=PE_IDLE;
            ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->next_state_predict_time=sub->next_state_predict_time;

            break;
        }
        case SR_CXLCTRL_RC_DATA_TRANSFER:
        {
            sub->current_time=ssd->current_time;
            sub->current_state=SR_CXLCTRL_RC_DATA_TRANSFER;
            sub->next_state=SR_CHANNEL_RC_DATA_TRANSFER;
            // uint64_t read_data_size = 0; 
            // for(int i = 0; i < sub->addr_num; i++) read_data_size += sub->p_addr[i]->size;
            // sub->next_state_predict_time=ssd->current_time + read_data_size / ssd->parameter->cxl_bandwidth;

            sub->next_state_predict_time=ssd->current_time + 1;
            if (sub->next_state_predict_time<=ssd->current_time)  sub->next_state_predict_time=ssd->current_time+1;

            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->current_state=PE_DATA_TRANSFER;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->current_time=ssd->current_time;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->next_state=PE_IDLE;
            // ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].pe->next_state_predict_time=sub->next_state_predict_time;

            break;
        }
        default :  return ERROR;
    }
    return SUCCESS;
}

/**************************************************************************
 * All requests that needs the same channel follows a random priority
 ****************************************************************************/
void shuffle(int *array, size_t n) {
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

Status services_all_requests_using_channel(struct cxl_switch_device_info * ssd, unsigned int channel, unsigned int * channel_busy_flag, int * gpu_channel_states){
    srand(time(NULL));
    
    // Array of function pointers
    Status (*functions[])(struct cxl_switch_device_info *, unsigned int, unsigned int *, int *) = {
        services_read_from_wait_to_CATransfer_in_channel,
        services_readCompute_from_wait_to_CATransfer_in_channel,
        services_write_from_wait_to_dataAndCATransfer_in_channel,
        services_read_from_dataTransfer_to_dataTransfer_from_cxlctrl_to_channel,
        services_readCompute_from_dataTransfer_to_dataTransfer_from_cxlctrl_to_channel};
    // int indices[] = {4, 1, 0, 2, 3};
    // int indices[] = {4, 0, 1, 2, 3};
    int indices[] = {0, 1, 2, 3, 4};
    size_t n = sizeof(indices) / sizeof(indices[0]);

    // // Shuffle the indices
    // shuffle(indices, n);

    // // Print shuffled indices
    // printf("Shuffled indices: ");
    // for (size_t i = 0; i < n; i++) {
    //     printf("%d ", indices[i]);
    // }
    // printf("\n");


    // Execute functions in random order
    for (size_t ii = 0; ii < n; ii++) {
        functions[indices[ii]](ssd, channel, channel_busy_flag, gpu_channel_states);
    }

    return SUCCESS;
}



/********************************************************
 *这个函数的主要功能是主控读子请求和写子请求的状态变化处理
 *======================================================
 *The main function of this function is the state change processing of the master read subrequest and the write subrequest.
 *********************************************************/
struct cxl_switch_device_info *process(struct cxl_switch_device_info *ssd) {
#ifdef DEBUG
    printf("enter process,  current time:%lld\n",ssd->current_time);
#endif

    // 1. 这里处理的是已经complete的请求，其中 read 和 readCompute 是直接从 channel 中的 Data 传输到终止，而 write 是在 cxl mem 中写完就终止
    services_read_complete_in_channel(ssd);
    services_write_complete_in_cxlctrl(ssd);
    services_readCompute_complete_in_channel(ssd);

    // 2. 这里处理的都是 sub request 从 channel 中到 cxl mem 中的状态跳转
    services_read_from_CATransfer_to_CAAnalyze_from_channel_to_cxlctrl(ssd);
    services_write_from_dataAndCATransfer_to_CAAnalyze_from_channel_to_cxlctrl(ssd);
    services_readCompute_from_CATransfer_to_CAAnalyze_from_channel_to_cxlctrl(ssd);

    // 3. 这里处理的是在 cxl mem 内部的状态跳转
    services_read_from_CAAnalyze_to_CATransfer_in_cxlctrl(ssd);
    services_read_from_CATransfer_to_dataTransfer_in_cxlctrl(ssd);
    services_write_from_CAAnalyze_to_write_in_cxlctrl(ssd);
    services_write_from_write_to_confirm_in_cxlctrl(ssd);
    services_readCompute_from_CAAnalyze_to_CATransfer_in_cxlctrl(ssd);
    services_readCompute_from_CATransfer_to_read_in_cxlctrl(ssd);
    services_readCompute_from_read_to_compute_in_cxlctrl(ssd);
    services_readCompute_from_compute_to_dataTransfer_in_cxlctrl(ssd);

    // 这里是一些常数的定义，channel_busy_flag 为1代表channel被占用，为0表示空闲
    unsigned int channel_busy_flag = 0; // channel_busy_flag 为1代表channel被占用, change_current_time_flag 为1表示需要调整当前时间 | change_current_time_flag 1 means that the current time needs to be adjusted
    unsigned long random_num = rand() % ssd->parameter->cxl_channel_number; /*产生一个随机数，保证每次从不同的channel开始查询 | Generate a random number to ensure that each query starts from a different channel*/
    int * gpu_channel_states = (int *)malloc(sizeof(int) * ssd->parameter->gpu_channel_number); // 设置一个记录gpu channel是否busy的标志位数组，记录所有的gpu channel是否busy，1为busy
    memset(gpu_channel_states, 0, sizeof(int) * ssd->parameter->gpu_channel_number);

    for(int ori_i = 0; ori_i < ssd->parameter->cxl_channel_number; ori_i++) {
        unsigned int i = (random_num + ori_i) % ssd->parameter->cxl_channel_number; //随机选择一个channel | Randomly select a channel
        channel_busy_flag = 0; /*每次进入channel时，flag都要置为0 | Each time you enter the channel, the channel_busy_flag must be set to 0*/
        if(ssd->bottom_channel_head[i].current_state == CHANNEL_IDLE || (ssd->bottom_channel_head[i].next_state == CHANNEL_IDLE
                                                                         && ssd->bottom_channel_head[i].next_state_predict_time <= ssd->current_time)) {
            // // 4. 这里处理的是仅仅在 channel 内部的操作，和 cxl mem 无关，不涉及到 cxl mem 的状态跳转，因此不需要修改，sub request 处于等待状态，准备开始CA传输，是把未装载的 sub request 装载到 channel 上
            // services_read_from_wait_to_CATransfer_in_channel(ssd, i, &channel_busy_flag, gpu_channel_states);    /*处理处于等待状态的读/读计算请求，准备开始CA传输 | Handling read child requests in wait state*/
            // services_readCompute_from_wait_to_CATransfer_in_channel(ssd, i, &channel_busy_flag, gpu_channel_states);    /*处理处于等待状态的读/读计算请求，准备开始CA传输 | Handling read child requests in wait state*/
            // services_write_from_wait_to_dataAndCATransfer_in_channel(ssd, i, &channel_busy_flag, gpu_channel_states);    /*处理处于等待状态的写请求，准备开始DATA&CA传输 | Handling write child requests in wait state*/
            // services_read_from_dataTransfer_to_dataTransfer_from_cxlctrl_to_channel(ssd, i, &channel_busy_flag, gpu_channel_states);
            // services_readCompute_from_dataTransfer_to_dataTransfer_from_cxlctrl_to_channel(ssd, i, &channel_busy_flag, gpu_channel_states);

            services_all_requests_using_channel(ssd, i, &channel_busy_flag, gpu_channel_states);

            
        }
    }
    return ssd;
}
