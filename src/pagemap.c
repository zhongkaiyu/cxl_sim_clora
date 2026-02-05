/*****************************************************************************************************************************
  This project was supported by the National Basic Research 973 Program of China under Grant No.2011CB302301
  Huazhong University of Science and Technology (HUST)   Wuhan National Laboratory for Optoelectronics

  FileName： pagemap.h
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

#include "../include/pagemap.h"
#include "../include/flash.h"
#include "../include/ssd.h"

/*****************************************************
 *断言,当申请内存空间失败时，输出“malloc 变量名 error”
 ******************************************************/
void alloc_assert(void *p,char *s)//断言
{
    if(p!=NULL) return;
    printf("malloc %s error\n",s);
    getchar();
    exit(-1);
}

/************************************************************************************
 *函数的功能是根据物理页号ppn查找该物理页所在的channel，chip，die，plane，block，page
 *得到的channel，chip，die，plane，block，page放在结构location中并作为返回值
 *************************************************************************************/
struct local *find_location(struct cxl_switch_device_info *ssd, unsigned int ppn)
{
    struct local *location=NULL;
    unsigned int i=0;
    int pn,ppn_value=ppn;
    int page_plane=0,page_die=0,page_chip=0,page_channel=0;

    pn = ppn;

#ifdef DEBUG
    printf("enter find_location\n");
#endif

    location=(struct local *)malloc(sizeof(struct local));
    alloc_assert(location,"location");
    memset(location,0, sizeof(struct local));

    page_plane=ssd->parameter->page_block*ssd->parameter->block_plane;
    page_die=page_plane*ssd->parameter->plane_die;
    page_chip=page_die*ssd->parameter->die_chip;
    page_channel=page_chip*ssd->parameter->chip_num_per_channel[0];

    /*******************************************************************************
     *page_channel是一个channel中page的数目， ppn/page_channel就得到了在哪个channel中
     *用同样的办法可以得到chip，die，plane，block，page
     ********************************************************************************/
    location->channel = ppn/page_channel;
    location->chip = (ppn%page_channel)/page_chip;
    location->die = ((ppn%page_channel)%page_chip)/page_die;
    location->plane = (((ppn%page_channel)%page_chip)%page_die)/page_plane;
    location->block = ((((ppn%page_channel)%page_chip)%page_die)%page_plane)/ssd->parameter->page_block;
    location->page = (((((ppn%page_channel)%page_chip)%page_die)%page_plane)%ssd->parameter->page_block)%ssd->parameter->page_block;

    return location;
}


/*****************************************************************************
 *这个函数的功能是根据参数channel，chip，die，plane，block，page，找到该物理页号
 *函数的返回值就是这个物理页号
 * The function of this function is to find the physical page number according to the parameters channel, chip, die, plane, block, page.
 * The return value of the function is the physical page number
 ******************************************************************************/
unsigned int find_ppn(struct cxl_switch_device_info * ssd, unsigned int channel, unsigned int chip, unsigned int die, unsigned int plane, unsigned int block, unsigned int page)
{
    unsigned int ppn=0;
    unsigned int i=0;
    int page_plane=0,page_die=0,page_chip=0;
    int page_channel[100];                  /*这个数组存放的是每个channel的page数目 | This array stores the number of pages per channel*/

#ifdef DEBUG
    printf("enter find_psn,channel:%d, chip:%d, die:%d, plane:%d, block:%d, page:%d\n",channel,chip,die,plane,block,page);
#endif

    /*********************************************
     *计算出plane，die，chip，channel中的page的数目 | Calculate the number of pages in plane, die, chip, channel
     **********************************************/
    page_plane=ssd->parameter->page_block*ssd->parameter->block_plane;
    page_die=page_plane*ssd->parameter->plane_die;
    page_chip=page_die*ssd->parameter->die_chip;
    while(i<ssd->parameter->cxl_channel_number)
    {
        page_channel[i]= ssd->parameter->chip_num_per_channel[i] * page_chip;
        i++;
    }

    /****************************************************************************
     *计算物理页号ppn，ppn是channel，chip，die，plane，block，page中page个数的总和
     *Calculate the physical page number ppn, ppn is the sum of the number of pages in channel, chip, die, plane, block, page
     *****************************************************************************/
    i=0;
    while(i<channel)
    {
        ppn=ppn+page_channel[i];
        i++;
    }
    ppn=ppn+page_chip*chip+page_die*die+page_plane*plane+block*ssd->parameter->page_block+page;

    return ppn;
}

/***************************************************************************************************
 *函数功能是在所给的channel，chip，die，plane里面找到一个active_block然后再在这个block里面找到一个页，
 *再利用find_ppn找到ppn。
 * Function is to find an active_block in the given channel, chip, die, plane and then find a page in this block,
 * Use find_ppn to find ppn.
 ****************************************************************************************************/
struct cxl_switch_device_info *get_ppn(struct cxl_switch_device_info *ssd, unsigned int channel, unsigned int chip, unsigned int die, unsigned int plane, struct sub_request *sub)
{
    int old_ppn=-1;
    unsigned int ppn,lpn,full_page;
    unsigned int active_block;
    unsigned int block;
    unsigned int page,flag=0,flag1=0;
    unsigned int old_state=0,state=0,copy_subpage=0;
    unsigned int is_in_tw=0, is_gc_inited=1;
    struct local *location;
    struct direct_erase *direct_erase_node,*new_direct_erase;
    struct gc_operation *gc_node;

    unsigned int i=0,j=0,k=0,l=0,m=0,n=0;

#ifdef DEBUG
    printf("enter get_ppn,channel:%d, chip:%d, die:%d, plane:%d\n",channel,chip,die,plane);
#endif

    full_page=~(0xffffffff<<(ssd->parameter->subpage_page));
    lpn=sub->lpn;

    /*************************************************************************************
     *利用函数find_active_block在channel，chip，die，plane找到活跃block
     *并且修改这个channel，chip，die，plane，active_block下的last_write_page和free_page_num
     * Use the find_active_block function to find active blocks on channel, chip, die, plane
     * and modify the last_write_page and free_page_num under this channel, chip, die, plane, active_block
     **************************************************************************************/
    if(find_active_block(ssd,channel,chip,die,plane)==FAILURE)                      
    {
        printf("ERROR :there is no free page in channel:%d, chip:%d, die:%d, plane:%d\n",channel,chip,die,plane);	
        return ssd;
    }

    active_block=ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].active_block;
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].last_write_page++;
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].free_page_num--;

    if(ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].last_write_page >= ssd->parameter->page_block)
    {
        printf("error! the last write page larger than %d!!\n", ssd->parameter->page_block);
        while(1){}
    }

    block=active_block;	
    page=ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].last_write_page;

    if(ssd->dram->map->map_entry[lpn].state==0)                                       /*this is the first logical page*/
    {
        if(ssd->dram->map->map_entry[lpn].pn!=0)
        {
            printf("Error in get_ppn()\n");
        }
        ssd->dram->map->map_entry[lpn].pn=find_ppn(ssd,channel,chip,die,plane,block,page);
        ssd->dram->map->map_entry[lpn].state=sub->state;
    }
    else                                                                            /*这个逻辑页进行了更新，需要将原来的页置为失效*/
    {
        ppn=ssd->dram->map->map_entry[lpn].pn;
        location=find_location(ssd,ppn);
        if(ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].page_head[location->page].lpn != lpn)
        {
            printf("\nError in get_ppn()\n");
        }

        ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].page_head[location->page].valid_state=0;             /*表示某一页失效，同时标记valid和free状态都为0*/
        ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].page_head[location->page].free_state=0;              /*表示某一页失效，同时标记valid和free状态都为0*/
        ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].page_head[location->page].lpn=0;
        ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].invalid_page_num++;

        /*******************************************************************************************
         *该block中全是invalid的页，可以直接删除，就在创建一个可擦除的节点，挂在location下的plane下面
         *The block is all invalid pages, you can delete directly, just create an erasable node, hung under the plane under the location
         ********************************************************************************************/
        if (ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].die_head[location->die].plane_head[location->plane].blk_head[location->block].invalid_page_num == ssd->parameter->page_block)
        {
            new_direct_erase=(struct direct_erase *)malloc(sizeof(struct direct_erase));
            alloc_assert(new_direct_erase,"new_direct_erase");
            memset(new_direct_erase,0, sizeof(struct direct_erase));

            new_direct_erase->block=location->block;
            new_direct_erase->next_node=NULL;
            direct_erase_node=ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].die_head[location->die].plane_head[location->plane].erase_node;
            if (direct_erase_node==NULL)
            {
                ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].die_head[location->die].plane_head[location->plane].erase_node=new_direct_erase;
            } 
            else
            {
                new_direct_erase->next_node=ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].die_head[location->die].plane_head[location->plane].erase_node;
                ssd->bottom_channel_head[location->channel].cxl_device->chip_head[location->chip].die_head[location->die].plane_head[location->plane].erase_node=new_direct_erase;
            }
        }

        free(location);
        location=NULL;
        ssd->dram->map->map_entry[lpn].pn=find_ppn(ssd,channel,chip,die,plane,block,page);
        ssd->dram->map->map_entry[lpn].state=(ssd->dram->map->map_entry[lpn].state|sub->state);
    }


    sub->ppn=ssd->dram->map->map_entry[lpn].pn;                                      /*修改sub子请求的ppn，location等变量*/
    sub->location->channel=channel;
    sub->location->chip=chip;
    sub->location->die=die;
    sub->location->plane=plane;
    sub->location->block=active_block;
    sub->location->page=page;

    ssd->program_count++;                                                           /*修改ssd的program_count,free_page等变量*/
    ssd->in_program_size+=ssd->parameter->subpage_page;
    ssd->bottom_channel_head[channel].program_count++;
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].program_count++;
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].free_page--;
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].page_head[page].lpn=lpn;
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].page_head[page].valid_state=sub->state;
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].page_head[page].free_state=((~(sub->state)) & full_page);
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].page_head[page].written_count++;
    ssd->write_flash_count++;

    if (ssd->parameter->active_write==0)                                            /*如果没有主动策略，只采用gc_hard_threshold，并且无法中断GC过程 | If there is no active policy, only gc_hard_threshold is used, and the GC process cannot be interrupted.*/
    {                                                                               /*如果plane中的free_page的数目少于gc_hard_threshold所设定的阈值就产生gc操作*/
        if (ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].free_page < (ssd->parameter->page_block * ssd->parameter->block_plane * ssd->parameter->gc_hard_threshold))
        {
            // check whether gc process already initialized for this plane
            is_gc_inited=1;
            gc_node=ssd->bottom_channel_head[channel].gc_command;
            while(gc_node!=NULL) {
                if (gc_node->chip==chip && gc_node->die==die && gc_node->plane==plane) {
                    is_gc_inited = 0;
                    break;
                }
                gc_node=gc_node->next_node;
            }

            // only initialized gc if it wasn't initialized previously
            if (is_gc_inited) {
                gc_node=(struct gc_operation *)malloc(sizeof(struct gc_operation));
                alloc_assert(gc_node,"gc_node");
                memset(gc_node,0, sizeof(struct gc_operation));

                gc_node->next_node=NULL;
                gc_node->chip=chip;
                gc_node->die=die;
                gc_node->plane=plane;
                gc_node->block=0xffffffff;
                gc_node->page=0;
                gc_node->state=GC_WAIT;
                gc_node->priority=GC_UNINTERRUPT;
                gc_node->next_node=ssd->bottom_channel_head[channel].gc_command;
                gc_node->x_init_time = ssd->bottom_channel_head[channel].current_time;
                gc_node->x_free_percentage = (double) ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].free_page / (double) (ssd->parameter->page_block * ssd->parameter->block_plane) * (double) 100;
                gc_node->x_moved_pages=0;

                ssd->bottom_channel_head[channel].gc_command=gc_node;
                ssd->gc_request++;
            }
        }
    } 

    return ssd;
}
/*****************************************************************************************
 *这个函数功能是为gc操作寻找新的ppn，因为在gc操作中需要找到新的物理块存放原来物理块上的数据
 *在gc中寻找新物理块的函数，不会引起循环的gc操作
 ******************************************************************************************/
unsigned int get_ppn_for_gc(struct cxl_switch_device_info *ssd, unsigned int channel, unsigned int chip, unsigned int die, unsigned int plane)
{
    unsigned int ppn;
    unsigned int active_block,block,page;

#ifdef DEBUG
    printf("enter get_ppn_for_gc,channel:%d, chip:%d, die:%d, plane:%d\n",channel,chip,die,plane);
#endif

    if(find_active_block(ssd,channel,chip,die,plane)!=SUCCESS)
    {
        printf("\n\n Error int get_ppn_for_gc().\n");
        return 0xffffffff;
    }

    active_block=ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].active_block;

    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].last_write_page++;
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].free_page_num--;

    if(ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].last_write_page >= ssd->parameter->page_block)
    {
        printf("error! the last write page larger than %d!!\n", ssd->parameter->page_block);
        while(1){}
    }

    block=active_block;	
    page=ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].last_write_page;

    ppn=find_ppn(ssd,channel,chip,die,plane,block,page);

    ssd->program_count++;
    ssd->in_program_size+=ssd->parameter->subpage_page;
    ssd->bottom_channel_head[channel].program_count++;
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].program_count++;
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].free_page--;
    ssd->bottom_channel_head[channel].cxl_device->chip_head[chip].die_head[die].plane_head[plane].blk_head[active_block].page_head[page].written_count++;
    ssd->write_flash_count++;

    return ppn;

}

