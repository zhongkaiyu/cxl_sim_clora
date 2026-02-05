#include <sys/stat.h>
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "../include/ssd.h"

// parse_user_args function parses optional and required options and arguments
// for ssdsim, the parsed arguments can be accessed through uargs.
// This function will return -1 if an error occurrs.
int parse_user_args(int argc, char *argv[], struct user_args* uargs) {
    char **positionals;
    int raidtype = -1;
    int ndisk = 0, diskid = 0;
    int64_t gc_time_window = 0;

    static struct option long_options[] = {
        {"raid0", no_argument, 0, '0'},
        {"raid5", no_argument, 0, '5'},
        {"gcsync", no_argument, 0, 's'},
        {"gclock", no_argument, 0, 'l'},
        {"gcdefer", no_argument, 0, 'd'},
        {"ndisk", required_argument, 0, 'n'},

        {"timestamp", required_argument, 0, 't'},       // simulation timestamp, for logging purpose
        {"diskid", required_argument, 0, 'i'},          // for gcsync purpose
        {"gc_time_window", required_argument, 0, 'g'},  // for gcsync purpose, in ns
        {"parameter", required_argument, 0, 'p'},       // parameter file

        {"file", required_argument, 0, 'f'}, // trace file
        {"debug", no_argument, 0, 'b'},     // debug mode
        {0, 0, 0, 0}   // end of options，这个必须要放到最后，否则会导致后面的参数无法解析
    };
    
    // Parsing program options
    int long_index = 0;
    int opt = 0;
    int is_debug = 0;
    while ((opt = getopt_long(argc, argv,"05n:", long_options, &long_index )) != -1) {
        switch (opt) {
            case 'l':
                uargs->is_gclock = 1;
                break;
            case 'd':
                uargs->is_gcdefer = 1;
                break;    
            case 's':
                uargs->is_gcsync = 1;
                break;
            case 'n':
                ndisk = atoi(optarg);
                if (ndisk == 0) {
                    printf("Error! wrong number of disk!\n");
                    return -1;
                }
                uargs->num_disk = ndisk;
                break;
            case 't':
                strcpy(uargs->simulation_timestamp, optarg);
                break;
            case 'i':
                diskid = atoi(optarg);
                if (diskid < 0) {
                    printf("Error! wrong diskid, it must be >= 0, but get %d!\n", diskid);
                    return -1;
                }
                uargs->diskid = diskid;
                break;
            case 'g':
                gc_time_window = atoll(optarg);
                if (gc_time_window < 0) {
                    printf("Error! wrong gc_time_window, it must be > 0, but get %lld!\n", gc_time_window);
                    return -1;
                }
                uargs->gc_time_window = gc_time_window;
                break;
            case 'p':
                strcpy(uargs->parameter_filename, optarg);
                break;
            case 'f':
                strcpy(uargs->trace_filename, optarg);
                break;
            case 'b':
                is_debug = 1;
                break;
            default:
                printf("Error! parse arguments failed.\n");
                return -1;
        }
    }

    // // Parsing tracefile
    // if (optind == argc) {
    //     printf("Error! require tracefile to run simulation\n");
    //     return -1;
    // }
    // strcpy(uargs->trace_filename, argv[optind]);

    // Additional constraints
    if (uargs->is_gcsync + uargs->is_gclock + uargs->is_gcdefer > 1) {
        printf("Error! multiple gc scheduling algorithm activated!\n");
        return -1;
    }
    if (uargs->is_gcsync && !uargs->gc_time_window && !uargs->num_disk) {
        printf("Error! GCSync mode need ndisk, diskid, and gc_time_window!\n");
        return -1;
    }

    return is_debug;
}

// initialize_ssd function initializes ssd struct based on user arguments and also default value.
// the most important arguments to be initialized is the tracefile and also
// ssd parameter config file. This function also prepare all log file to store information about single ssd simulation
struct cxl_switch_device_info *initialize_ssd(struct cxl_switch_device_info* ssd, struct user_args* uargs) {
    int i;
    char *opt;
    char *current_time;
    char logdir[30];
    char logdirname[60];

    // Prepare log directory for this ssd
    current_time = (char*) malloc(sizeof(char)*16);
    if (strlen(uargs->simulation_timestamp) != 0) {
        strcpy(current_time, uargs->simulation_timestamp);
    } else {
        get_current_time(current_time);
    }

    // 创建 raw 目录，如果不存在
    if (0 != mkdir("raw", 0777) && errno != EEXIST) {
        perror("mkdir raw");
        exit(1);
    }

    // 初始化 logdir 并拼接时间戳
    strcpy(logdir, "raw/");
    strcat(logdir, current_time);

    // 创建 logdir 目录
    if (0 != mkdir(logdir,0777)) {
        printf("When executing: mkdir(\"%s\")\n", logdir);
        perror("mkdir");
        exit(1);
    }
    strcat(logdir, "/");

    // Assign default value
    strcpy(logdirname, logdir); strcat(logdirname, "ex.out");
    strcpy(ssd->outputfilename, logdirname);
    strcpy(logdirname, logdir); strcat(logdirname, "statistic10.dat");
    strcpy(ssd->statisticfilename, logdirname);
    strcpy(logdirname, logdir); strcat(logdirname, "statistic2.dat");
    strcpy(ssd->statisticfilename2, logdirname);
    strcpy(logdirname, logdir); strcat(logdirname, "io.dat");
    strcpy(ssd->outfile_io_name, logdirname);
    strcpy(logdirname, logdir); strcat(logdirname, "io_write.dat");
    strcpy(ssd->outfile_io_write_name, logdirname);
    strcpy(logdirname, logdir); strcat(logdirname, "io_read.dat");
    strcpy(ssd->outfile_io_read_name, logdirname);
    strcpy(logdirname, logdir); strcat(logdirname, "gc.dat");
    strcpy(ssd->outfile_gc_name, logdirname);

    // Assign ssd parameter config file
    if (strlen(uargs->parameter_filename) == 0)
        strcpy(ssd->parameterfilename,"config/parameters.conf");
    else
        strcpy(ssd->parameterfilename, uargs->parameter_filename);

    // Assign tracefilename
    if (strnlen(uargs->trace_filename, 1) == 0)
        strcpy(ssd->tracefilename, "tracefile.txt");
    else
        strcpy(ssd->tracefilename, uargs->trace_filename);
    
    // Assign all var related to GCSync
    if (uargs->is_gcsync) {
        ssd->ndisk = uargs->num_disk;
        ssd->diskid = uargs->diskid;
        ssd->is_gcsync = 1;
        ssd->gc_time_window = uargs->gc_time_window;
    }

    // Assign all var related to GCLock
    if (uargs->is_gclock) {
        ssd->ndisk = uargs->num_disk;
        ssd->diskid = uargs->diskid;
        ssd->is_gclock = 1;
    }

    // Assign all var related to GCDefer
    if (uargs->is_gcdefer) {
        ssd->ndisk = uargs->num_disk;
        ssd->diskid = uargs->diskid;
        ssd->is_gcdefer = 1;
    }

    free(current_time);
    return ssd;
}

/********    get_request    ******************************************************
 *	1.get requests that arrived already
 *	2.add those request node to ssd->reuqest_queue
 *	return	0: reach the end of the trace
 *			-1: no request has been added
 *			1: add one request to list
 *SSD模拟器有三种驱动方式:时钟驱动(精确，太慢) 事件驱动(本程序采用) trace驱动()，
 *两种方式推进事件：channel/chip状态改变、trace文件请求达到。
 *channel/chip状态改变和trace文件请求到达是散布在时间轴上的点，每次从当前状态到达
 *下一个状态都要到达最近的一个状态，每到达一个点执行一次process
 ********************************************************************************/
int get_requests(struct cxl_switch_device_info *ssd)
{  
    char buffer[200];
    unsigned int lsn=0;
    int device,  size, ope, large_lsn, i = 0, j=0;
    struct request *request1;
    int flag = 1;
    long filepoint; 
    int64_t time_t = 0;
    int64_t nearest_event_time;    

#ifdef DEBUG
    printf("enter get_requests,  current time:%lld\n",ssd->current_time);
#endif

    // If not EOF, try to add new request
    if(!feof(ssd->tracefile)) {
        filepoint = ftell(ssd->tracefile);
        fgets(buffer, 200, ssd->tracefile);
        sscanf(buffer,"%lld %d %d %d %d",&time_t,&device,&lsn,&size,&ope);
        if (filepoint == 0) {
            ssd->simulation_start_time = time_t;
        }

    // If EOF, continue to process the request queue until empty
    } else {
        nearest_event_time=find_nearest_event(ssd);
        ssd->current_time=nearest_event_time;
        ssd->simulation_end_time = ssd->current_time;
        return 0;
    }

    if ((device<0)&&(lsn<0)&&(size<0)&&(ope<0)) {
        printf("Error! wrong io request from trace file\n");
        return 100;
    }

    if (lsn<ssd->min_lsn) 
        ssd->min_lsn=lsn;
    if (lsn>ssd->max_lsn)
        ssd->max_lsn=lsn;
    /******************************************************************************************************
     *上层文件系统发送给SSD的任何读写命令包括两个部分（LSN，size） LSN是逻辑扇区号，对于文件系统而言，它所看到的存
     *储空间是一个线性的连续空间。例如，读请求（260，6）表示的是需要读取从扇区号为260的逻辑扇区开始，总共6个扇区。
     *large_lsn: channel下面有多少个subpage，即多少个sector。overprovide系数：SSD中并不是所有的空间都可以给用户使用，
     *比如32G的SSD可能有10%的空间保留下来留作他用，所以乘以1-provide
     ***********************************************************************************************************/
    large_lsn=(int)((ssd->parameter->subpage_page*ssd->parameter->page_block*ssd->parameter->block_plane*ssd->parameter->plane_die*ssd->parameter->die_chip*ssd->parameter->chip_num)*(1-ssd->parameter->overprovide));
    lsn = lsn%large_lsn;
    nearest_event_time=find_nearest_event(ssd);
    if (nearest_event_time==MAX_INT64)
    {
        ssd->current_time=time_t;           
    }
    else
    {
        if(nearest_event_time<time_t)
        {
            /*******************************************************************************
             *回滚，即如果没有把time_t赋给ssd->current_time，则trace文件已读的一条记录回滚
             *filepoint记录了执行fgets之前的文件指针位置，回滚到文件头+filepoint处
             *int fseek(FILE *stream, long offset, int fromwhere);函数设置文件指针stream的位置。
             *如果执行成功，stream将指向以fromwhere（偏移起始位置：文件头0，当前位置1，文件尾2）为基准，
             *偏移offset（指针偏移量）个字节的位置。如果执行失败(比如offset超过文件自身大小)，则不改变stream指向的位置。
             *文本文件只能采用文件头0的定位方式，本程序中打开文件方式是"r":以只读方式打开文本文件	
             **********************************************************************************/
            fseek(ssd->tracefile,filepoint,0); 
            if(ssd->current_time<=nearest_event_time)
                ssd->current_time=nearest_event_time;
            return -1;
        }
        else // nearest_event_time >= time_t
        {
            if (ssd->request_queue_length>=ssd->parameter->queue_length)
            {
                fseek(ssd->tracefile,filepoint,0);
                ssd->current_time=nearest_event_time;
                return -1;
            } 
            else
            {
                ssd->current_time=time_t;
            }
        }
    }

    if(time_t < 0)
    {
        printf("error!\n");
        while(1){}
    }

    // if(feof(ssd->tracefile))
    // {
    //     request1=NULL;
    //     return 0;
    // }

    request1 = (struct request*)malloc(sizeof(struct request));
    alloc_assert(request1,"request");
    memset(request1,0, sizeof(struct request));

    request1->time = time_t;
    request1->lsn = lsn;
    request1->size = size;
    request1->operation = ope;	
    request1->begin_time = time_t;
    request1->response_time = 0;	
    request1->energy_consumption = 0;	
    request1->next_node = NULL;
    request1->distri_flag = 0;              // indicate whether this request has been distributed already
    request1->subs = NULL;
    request1->need_distr_flag = NULL;
    request1->complete_lsn_count=0;         //record the count of lsn served by buffer
    filepoint = ftell(ssd->tracefile);		// set the file point

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

    if (request1->operation==READ)             //计算平均请求大小 1为读 0为写
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
    else
    {
        ssd->ave_write_size=(ssd->ave_write_size*ssd->write_request_count+request1->size)/(ssd->write_request_count+1);
        ssd->write_request_size+=request1->size;
    }


    // filepoint = ftell(ssd->tracefile);	
    // fgets(buffer, 200, ssd->tracefile);    //寻找下一条请求的到达时间
    // // sscanf(buffer,"%lld %d %d %d %d",&time_t,&device,&lsn,&size,&ope);
    // sscanf(buffer,"%lld %d %d %d %d %d %d %d %d",&time_t,&device,&lsn,&size,&ope,&aw,&ah,&bw,&bh);
    // ssd->next_request_time=time_t;
    // fseek(ssd->tracefile,filepoint,0);

    return 1;
}

/**********************************************************************************************************************************************
 *首先buffer是个写buffer，就是为写请求服务的，因为读flash的时间tR为20us，写flash的时间tprog为200us，所以为写服务更能节省时间
 *  读操作：如果命中了buffer，从buffer读，不占用channel的I/O总线，没有命中buffer，从flash读，占用channel的I/O总线，但是不进buffer了
 *  写操作：首先request分成sub_request子请求，如果是动态分配，sub_request挂到ssd->sub_request上，因为不知道要先挂到哪个channel的sub_request上
 *          如果是静态分配则sub_request挂到channel的sub_request链上,同时不管动态分配还是静态分配sub_request都要挂到request的sub_request链上
 *		   因为每处理完一个request，都要在traceoutput文件中输出关于这个request的信息。处理完一个sub_request,就将其从channel的sub_request链
 *		   或ssd的sub_request链上摘除，但是在traceoutput文件输出一条后再清空request的sub_request链。
 *		   sub_request命中buffer则在buffer里面写就行了，并且将该sub_page提到buffer链头(LRU)，若没有命中且buffer满，则先将buffer链尾的sub_request
 *		   写入flash(这会产生一个sub_request写请求，挂到这个请求request的sub_request链上，同时视动态分配还是静态分配挂到channel或ssd的
 *		   sub_request链上),在将要写的sub_page写入buffer链头
 * Read operation: If you hit the buffer, read from the buffer, do not occupy the channel I / O bus, do not hit the buffer, read from the flash, occupy the channel I / O bus, but do not enter the buffer
 ***********************************************************************************************************************************************/
struct cxl_switch_device_info *buffer_management(struct cxl_switch_device_info *ssd)
{   
    unsigned int j,lsn,lpn,last_lpn,first_lpn,index,complete_flag=0, state,full_page;
    unsigned int flag=0,need_distb_flag,lsn_flag,flag1=1,active_region_flag=0;           
    struct request *new_request;
    struct buffer_group *buffer_node,key;
    unsigned int mask=0,offset1=0,offset2=0;

#ifdef DEBUG
    printf("enter buffer_management,  current time:%lld\n",ssd->current_time);
#endif
    ssd->dram->current_time=ssd->current_time;
    full_page=~(0xffffffff<<ssd->parameter->subpage_page);

    new_request=ssd->request_tail;
    lsn=new_request->lsn;
    lpn=new_request->lsn/ssd->parameter->subpage_page;
    last_lpn=(new_request->lsn+new_request->size-1)/ssd->parameter->subpage_page;
    first_lpn=new_request->lsn/ssd->parameter->subpage_page;

    new_request->need_distr_flag=(unsigned int*)malloc(sizeof(unsigned int)*((last_lpn-first_lpn+1)*ssd->parameter->subpage_page/32+1));
    alloc_assert(new_request->need_distr_flag,"new_request->need_distr_flag");
    memset(new_request->need_distr_flag, 0, sizeof(unsigned int)*((last_lpn-first_lpn+1)*ssd->parameter->subpage_page/32+1));

    if(new_request->operation==READ) 
    {	
        while(lpn<=last_lpn)      		
        {
            /************************************************************************************************
             *need_distb_flag表示是否需要执行distribution函数，1表示需要执行，buffer中没有，0表示不需要执行
             *即1表示需要分发，0表示不需要分发，对应点初始全部赋为1
             *************************************************************************************************/
            need_distb_flag=full_page;   
            key.group=lpn;
            buffer_node= (struct buffer_group*)avlTreeFind(ssd->dram->buffer, (TREE_NODE *)&key);		// buffer node 

            while((buffer_node!=NULL)&&(lsn<(lpn+1)*ssd->parameter->subpage_page)&&(lsn<=(new_request->lsn+new_request->size-1)))
            {
                lsn_flag=full_page;
                mask=1 << (lsn%ssd->parameter->subpage_page);
                if(mask>255) // 4KB page
                {
                    printf("the subpage number is larger than 8!add some cases");
                    getchar(); 		   
                }
                else if((buffer_node->stored & mask)==mask)
                {
                    flag=1;
                    lsn_flag=lsn_flag&(~mask);
                }

                if(flag==1)				
                {	//如果该buffer节点不在buffer的队首，需要将这个节点提到队首，实现了LRU算法，这个是一个双向队列。		       		
                    if(ssd->dram->buffer->buffer_head!=buffer_node)     
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
                    ssd->dram->buffer->read_hit++;					
                    new_request->complete_lsn_count++;											
                }		
                else if(flag==0)
                {
                    ssd->dram->buffer->read_miss_hit++;
                }

                need_distb_flag=need_distb_flag&lsn_flag;

                flag=0;		
                lsn++;						
            }	

            index=(lpn-first_lpn)/(32/ssd->parameter->subpage_page); 			
            new_request->need_distr_flag[index]=new_request->need_distr_flag[index]|(need_distb_flag<<(((lpn-first_lpn)%(32/ssd->parameter->subpage_page))*ssd->parameter->subpage_page));            
            lpn++;

        }
    }  
    else if(new_request->operation==WRITE)
    {
        while(lpn<=last_lpn)           	
        {	
            need_distb_flag=full_page;
            mask=~(0xffffffff<<(ssd->parameter->subpage_page));
            state=mask;

            if(lpn==first_lpn)
            {
                offset1=ssd->parameter->subpage_page-((lpn+1)*ssd->parameter->subpage_page-new_request->lsn);
                state=state&(0xffffffff<<offset1);
            }
            if(lpn==last_lpn)
            {
                offset2=ssd->parameter->subpage_page-((lpn+1)*ssd->parameter->subpage_page-(new_request->lsn+new_request->size));
                state=state&(~(0xffffffff<<offset2));
            }

            ssd=insert2buffer(ssd, lpn, state,NULL,new_request);
            lpn++;
        }
    }
    complete_flag = 1;
    for(j=0;j<=(last_lpn-first_lpn+1)*ssd->parameter->subpage_page/32;j++)
    {
        if(new_request->need_distr_flag[j] != 0)
        {
            complete_flag = 0;
        }
    }

    /*************************************************************
     *如果请求已经被全部由buffer服务，该请求可以被直接响应，输出结果
     *这里假设dram的服务时间为1000ns
     *If the request has been served entirely by the buffer, the request can be directly responded to the output.
     *This assumes that the service time of the dram is 1000ns.
     **************************************************************/
    if((complete_flag == 1)&&(new_request->subs==NULL))               
    {
        new_request->begin_time=ssd->current_time;
        new_request->response_time=ssd->current_time+1000;            
    }

    return ssd;
}

/*****************************
 *lpn向ppn的转换
 ******************************/
unsigned int lpn2ppn(struct cxl_switch_device_info *ssd, unsigned int lsn)
{
    int lpn, ppn;	
    struct entry *p_map = ssd->dram->map->map_entry;
#ifdef DEBUG
    printf("enter lpn2ppn,  current time:%lld\n",ssd->current_time);
#endif
    lpn = lsn/ssd->parameter->subpage_page;			//lpn
    ppn = (p_map[lpn]).pn;
    return ppn;
}

/**********************************************************************************
 *读请求分配子请求函数，这里只处理读请求，写请求已经在buffer_management()函数中处理了
 *根据请求队列和buffer命中的检查，将每个请求分解成子请求，将子请求队列挂在channel上，
 *不同的channel有自己的子请求队列
 *The read request allocates the sub-request function. Here only the read request is processed. The write request has been processed in the buffer_management() function.
 *According to the check of the request queue and the buffer hit, each request is decomposed into sub-requests, and the sub-request queue is hung on the channel.
 *Different channels have their own subrequest queues
 **********************************************************************************/
struct cxl_switch_device_info *distribute(struct cxl_switch_device_info *ssd)
{
    unsigned int start, end, first_lsn,last_lsn,lpn,flag=0,flag_attached=0,full_page;
    unsigned int j, k, sub_size;
    int i=0;
    struct request *req;
    struct sub_request *sub;
    unsigned int* complt;

#ifdef DEBUG
    printf("enter distribute,  current time:%lld\n",ssd->current_time);
#endif
    full_page=~(0xffffffff<<ssd->parameter->subpage_page);

    req = ssd->request_tail;
    if(req->response_time != 0){
        return ssd;
    }
    if (req->operation==WRITE)
    {
        return ssd;
    }

    if(req != NULL)
    {
        if(req->distri_flag == 0)
        {
            //如果还有一些读请求需要处理 | If there are still some read requests that need to be processed
            if(req->complete_lsn_count != ssd->request_tail->size)
            {
                first_lsn = req->lsn;				
                last_lsn = first_lsn + req->size;
                complt = req->need_distr_flag;
                start = first_lsn - first_lsn % ssd->parameter->subpage_page;
                end = (last_lsn/ssd->parameter->subpage_page + 1) * ssd->parameter->subpage_page;
                i = (end - start)/32;	

                while(i >= 0)
                {
                    /*************************************************************************************
                     *一个32位的整型数据的每一位代表一个子页，32/ssd->parameter->subpage_page就表示有多少页，
                     *这里的每一页的状态都存放在了 req->need_distr_flag中，也就是complt中，通过比较complt的
                     *每一项与full_page，就可以知道，这一页是否处理完成。如果没处理完成则通过creat_sub_request
                     函数创建子请求。

                     Each bit of a 32-bit integer data represents a subpage, and 32/ssd->parameter->subpage_page indicates 
                     how many pages there are. The state of each page here is stored in req->need_distr_flag, 
                     that is, In complt, by comparing each item of complt with full_page, you can know whether 
                     this page is processed or not. A subrequest is created by the creat_sub_request function 
                     if it is not processed.
                     *************************************************************************************/
                    for(j=0; j<32/ssd->parameter->subpage_page; j++)
                    {	
                        k = (complt[((end-start)/32-i)] >>(ssd->parameter->subpage_page*j)) & full_page;
                        if (k !=0)
                        {
                            lpn = start/ssd->parameter->subpage_page+ ((end-start)/32-i)*32/ssd->parameter->subpage_page + j;
                            sub_size=transfer_size(ssd,k,lpn,req);    
                            if (sub_size==0) 
                            {
                                continue;
                            }
                            else
                            {
                                sub=creat_sub_request(ssd,lpn,sub_size,0,req,req->operation);
                            }	
                        }
                    }
                    i = i-1;
                }

            }
            else
            {
                req->begin_time=ssd->current_time;
                req->response_time=ssd->current_time+1000;   
            }

        }
    }
    return ssd;
}


/**********************************************************************
 *trace_output()函数是在每一条请求的所有子请求经过process()函数处理完后，
 *打印输出相关的运行结果到outputfile文件中，这里的结果主要是运行的时间
 *====================================================================
 *The trace_output() function is executed after all sub-requests of each request have been processed by the process() function.
 *Print out the relevant running results to the outputfile, the result here is mainly the running time
 **********************************************************************/
void trace_output(struct cxl_switch_device_info* ssd){
    int flag = 1; 
    int64_t start_time, end_time, latency = -1;
    struct request *req, *pre_node;
    struct sub_request *sub, *tmp;

#ifdef DEBUG
    printf("enter trace_output,  current time:%lld\n",ssd->current_time);
#endif

    pre_node=NULL;
    req = ssd->request_queue;
    start_time = 0;
    end_time = 0;

    if(req == NULL)
        return;

    while(req != NULL)	
    {
        sub = req->subs;
        flag = 1;
        start_time = 0;
        end_time = 0;


        flag=1;
        while(sub != NULL) // 遍历当前request下的所有sub request，判断是否完成
        {
            if(start_time == 0)
                start_time = sub->begin_time;
            if(start_time > sub->begin_time)
                start_time = sub->begin_time;
            if(end_time < sub->complete_time)
                end_time = sub->complete_time;
            if((sub->current_state == SR_COMPLETE)||((sub->next_state==SR_COMPLETE)&&(sub->next_state_predict_time<=ssd->current_time)))	// if any sub-request is not completed, the request is not completed
            {
                sub = sub->next_subs; //当前sub request 已经完成
            }
            else        //当前sub request未完成
            {
                flag=0;
                break;
            }

        }

        if (flag == 1)  //当前request的所有sub request都已完成
        {		
            req->response_time = end_time;
            latency = end_time-req->time;
            addSet(ssd->npu->finished_reqs, req->idx); // 记录执行完成的requests

            if(end_time-start_time==0)
            {
                printf("the response time is 0?? \n");
                getchar();
            }

            if (req->operation==READ)
            {
                ssd->read_request_count++;
                ssd->read_avg=ssd->read_avg+(end_time-req->time);
            } 
            else
            {
                ssd->write_request_count++;
                ssd->write_avg=ssd->write_avg+(end_time-req->time);
            }

            while(req->subs!=NULL) // 释放已经结束的sub request占用的空间
            {
                tmp = req->subs;
                req->subs = tmp->next_subs;
                if (tmp->update!=NULL)
                {
                    free(tmp->update->location);
                    tmp->update->location=NULL;
                    free(tmp->update);
                    tmp->update=NULL;
                }
                free(tmp->location);
                tmp->location=NULL;
                free(tmp);
                tmp=NULL;

            }

            if(pre_node == NULL) // 当前req是整个ssd中第一个
            {
                if(req->next_node == NULL) //当前request是最后一个
                {
                    free(req->need_distr_flag);
                    req->need_distr_flag=NULL;
                    free(req);
                    req = NULL;
                    ssd->request_queue = NULL;
                    ssd->request_tail = NULL;
                    ssd->request_queue_length--;
                }
                else
                {
                    ssd->request_queue = req->next_node;
                    pre_node = req;
                    req = req->next_node;
                    free(pre_node->need_distr_flag);
                    pre_node->need_distr_flag=NULL;
                    free(pre_node);
                    pre_node = NULL;
                    ssd->request_queue_length--;
                }
            }
            else
            {
                if(req->next_node == NULL)
                {
                    pre_node->next_node = NULL;
                    free(req->need_distr_flag);
                    req->need_distr_flag=NULL;
                    free(req);
                    req = NULL;
                    ssd->request_tail = pre_node;	
                    ssd->request_queue_length--;
                }
                else
                {
                    pre_node->next_node = req->next_node;
                    free(req->need_distr_flag);
                    req->need_distr_flag=NULL;
                    free(req);
                    req = pre_node->next_node;
                    ssd->request_queue_length--;
                }

            }
        }
        else
        {	
            pre_node = req;
            req = req->next_node;
        }
        		
    }
}

/*******************************************************************************
 *statistic_output()函数主要是输出处理完一条请求后的相关处理信息。
 *1，计算出每个plane的擦除次数即plane_erase和总的擦除次数即erase
 *2，打印min_lsn，max_lsn，read_count，program_count等统计信息到文件outputfile中。
 *3，打印相同的信息到文件statisticfile中
 *******************************************************************************/
void statistic_output(struct cxl_switch_device_info *ssd)
{
#ifdef DEBUG
    printf("enter statistic_output,  current time:%lld\n",ssd->current_time);
#endif

    fprintf(ssd->outputfile,"\n");
    fprintf(ssd->outputfile,"\n");
    fprintf(ssd->outputfile,"---------------------------statistic data---------------------------\n");	 
    // fprintf(ssd->outputfile,"min lsn: %13d\n",ssd->min_lsn);	
    // fprintf(ssd->outputfile,"max lsn: %13d\n",ssd->max_lsn);
    fprintf(ssd->outputfile,"read count: %13lu\n",ssd->read_count);	  
    fprintf(ssd->outputfile,"program count: %13lu",ssd->program_count);	
    fprintf(ssd->outputfile,"                        include the flash write count leaded by read requests\n");
    // fprintf(ssd->outputfile,"the read operation leaded by un-covered update count: %13d\n",ssd->update_read_count);
    // fprintf(ssd->outputfile,"erase count: %13lu\n",ssd->erase_count);
    // fprintf(ssd->outputfile,"direct erase count: %13lu\n",ssd->direct_erase_count);
    // fprintf(ssd->outputfile,"copy back count: %13lu\n",ssd->copy_back_count);
    // fprintf(ssd->outputfile,"multi-plane program count: %13lu\n",ssd->m_plane_prog_count);
    // fprintf(ssd->outputfile,"multi-plane read count: %13lu\n",ssd->m_plane_read_count);
    // fprintf(ssd->outputfile,"interleave write count: %13lu\n",ssd->interleave_count);
    // fprintf(ssd->outputfile,"interleave read count: %13lu\n",ssd->interleave_read_count);
    // fprintf(ssd->outputfile,"interleave two plane and one program count: %13lu\n",ssd->inter_mplane_prog_count);
    // fprintf(ssd->outputfile,"interleave two plane count: %13lu\n",ssd->inter_mplane_count);
    // fprintf(ssd->outputfile,"gc copy back count: %13lu\n",ssd->gc_copy_back);
    // fprintf(ssd->outputfile,"write flash count: %13lu\n",ssd->write_flash_count);
    // fprintf(ssd->outputfile,"interleave erase count: %13lu\n",ssd->interleave_erase_count);
    // fprintf(ssd->outputfile,"multiple plane erase count: %13lu\n",ssd->mplane_erase_conut);
    // fprintf(ssd->outputfile,"interleave multiple plane erase count: %13lu\n",ssd->interleave_mplane_erase_count);
    fprintf(ssd->outputfile,"read request count: %13u\n",ssd->read_request_count);
    fprintf(ssd->outputfile,"write request count: %13u\n",ssd->write_request_count);
    fprintf(ssd->outputfile,"read request average size: %13f\n",ssd->ave_read_size);
    fprintf(ssd->outputfile,"write request average size: %13f\n",ssd->ave_write_size);
    if (ssd->read_request_count != 0)
        fprintf(ssd->outputfile,"read request average response time: %lld\n",ssd->read_avg/ssd->read_request_count);
    if (ssd->write_request_count != 0)
        fprintf(ssd->outputfile,"write request average response time: %lld\n",ssd->write_avg/ssd->write_request_count);
    // fprintf(ssd->outputfile,"buffer read hits: %13lu\n",ssd->dram->buffer->read_hit);
    // fprintf(ssd->outputfile,"buffer read miss: %13lu\n",ssd->dram->buffer->read_miss_hit);
    // fprintf(ssd->outputfile,"buffer write hits: %13lu\n",ssd->dram->buffer->write_hit);
    // fprintf(ssd->outputfile,"buffer write miss: %13lu\n",ssd->dram->buffer->write_miss_hit);
    // fprintf(ssd->outputfile,"erase: %13u\n",erase);
    // fprintf(ssd->outputfile,"write amplification: %.2f\n",(double)ssd->program_count/(double)ssd->write_request_count);
    // fprintf(ssd->outputfile,"read amplification: %.2f\n",(double)ssd->read_count/(double)ssd->read_request_count);
    fflush(ssd->outputfile);


    fprintf(ssd->statisticfile,"\n");
    fprintf(ssd->statisticfile,"\n");
    fprintf(ssd->statisticfile,"---------------------------statistic data---------------------------\n");	
    // fprintf(ssd->statisticfile,"min lsn: %13u\n",ssd->min_lsn);	
    // fprintf(ssd->statisticfile,"max lsn: %13u\n",ssd->max_lsn);
    fprintf(ssd->statisticfile,"read count: %13lu\n",ssd->read_count);	  
    fprintf(ssd->statisticfile,"program count: %13lu",ssd->program_count);	  
    fprintf(ssd->statisticfile,"                        include the flash write count leaded by read requests\n");
    fprintf(ssd->statisticfile,"the read operation leaded by un-covered update count: %13u\n",ssd->update_read_count);
    // fprintf(ssd->statisticfile,"erase count: %13lu\n",ssd->erase_count);	  
    // fprintf(ssd->statisticfile,"direct erase count: %13lu\n",ssd->direct_erase_count);
    // fprintf(ssd->statisticfile,"copy back count: %13lu\n",ssd->copy_back_count);
    // fprintf(ssd->statisticfile,"multi-plane program count: %13lu\n",ssd->m_plane_prog_count);
    // fprintf(ssd->statisticfile,"multi-plane read count: %13lu\n",ssd->m_plane_read_count);
    // fprintf(ssd->statisticfile,"interleave count: %13lu\n",ssd->interleave_count);
    // fprintf(ssd->statisticfile,"interleave read count: %13lu\n",ssd->interleave_read_count);
    // fprintf(ssd->statisticfile,"interleave two plane and one program count: %13lu\n",ssd->inter_mplane_prog_count);
    // fprintf(ssd->statisticfile,"interleave two plane count: %13lu\n",ssd->inter_mplane_count);
    // fprintf(ssd->statisticfile,"gc copy back count: %13lu\n",ssd->gc_copy_back);
    // fprintf(ssd->statisticfile,"gc count: %13lu\n",ssd->num_gc);
    // fprintf(ssd->statisticfile,"write flash count: %13lu\n",ssd->write_flash_count);
    // fprintf(ssd->statisticfile,"waste page count: %13lu\n",ssd->waste_page_count);
    // fprintf(ssd->statisticfile,"interleave erase count: %13lu\n",ssd->interleave_erase_count);
    // fprintf(ssd->statisticfile,"multiple plane erase count: %13lu\n",ssd->mplane_erase_conut);
    // fprintf(ssd->statisticfile,"interleave multiple plane erase count: %13lu\n",ssd->interleave_mplane_erase_count);
    fprintf(ssd->statisticfile,"read request count: %13u\n",ssd->read_request_count);
    fprintf(ssd->statisticfile,"write request count: %13u\n",ssd->write_request_count);
    fprintf(ssd->statisticfile,"read request average size: %13f\n",ssd->ave_read_size);
    fprintf(ssd->statisticfile,"write request average size: %13f\n",ssd->ave_write_size);
    if(ssd->read_request_count != 0)
        fprintf(ssd->statisticfile,"read request average response time: %lld\n",ssd->read_avg/ssd->read_request_count);
    if(ssd->write_request_count != 0)
        fprintf(ssd->statisticfile,"write request average response time: %lld\n",ssd->write_avg/ssd->write_request_count);
    // fprintf(ssd->statisticfile,"buffer read hits: %13lu\n",ssd->dram->buffer->read_hit);
    // fprintf(ssd->statisticfile,"buffer read miss: %13lu\n",ssd->dram->buffer->read_miss_hit);
    // fprintf(ssd->statisticfile,"buffer write hits: %13lu\n",ssd->dram->buffer->write_hit);
    // fprintf(ssd->statisticfile,"buffer write miss: %13lu\n",ssd->dram->buffer->write_miss_hit);
    // fprintf(ssd->statisticfile,"erase: %13u\n",erase);
    fprintf(ssd->statisticfile,"write sub request count: %13u\n",ssd->write_subreq_count);
    fprintf(ssd->statisticfile,"read subr request count: %13u\n",ssd->read_subreq_count);
    // if(ssd->write_request_count != 0)
    //     fprintf(ssd->statisticfile,"write amplification: %.2f\n",(double)ssd->program_count/(double)ssd->write_subreq_count);
    // if(ssd->read_request_count != 0)
    //     fprintf(ssd->statisticfile,"read amplification: %.2f\n",(double)ssd->read_count/(double)ssd->read_subreq_count);
    // fprintf(ssd->statisticfile, "write amplification (size): %.2f\n", (double)ssd->in_program_size/(double)ssd->write_request_size);
    // fprintf(ssd->statisticfile, "read amplification (size): %.2f\n", (double)ssd->in_read_size/(double)ssd->read_request_size);
    // fprintf(ssd->statisticfile, "avg. gc page move: %.2f (%.2f%%)\n", (double)ssd->gc_move_page/(double)ssd->num_gc, (100*((double)ssd->gc_move_page/(double)ssd->num_gc)/ssd->parameter->page_block));
    // fprintf(ssd->statisticfile, "gc time window: %lld\n", ssd->gc_time_window);
    fprintf(ssd->statisticfile, "\n\n simulation duration: %lld ns\n", ssd->simulation_end_time - ssd->simulation_start_time);
    fprintf(ssd->statisticfile, " IOPS: %.3f\n", (double)(ssd->read_count+ssd->program_count)/((double)(ssd->simulation_end_time - ssd->simulation_start_time)/1000000000));
    fprintf(ssd->statisticfile, " read BW: %.3f MB/s\n", ((double)ssd->read_request_size * 1024)/ ((double)(ssd->simulation_end_time - ssd->simulation_start_time)/1000000000));
    fprintf(ssd->statisticfile, " write BW: %.3f MB/s\n", ((double)ssd->write_request_size/2000.0)/ ((double)(ssd->simulation_end_time - ssd->simulation_start_time)/1000000000));
    fflush(ssd->statisticfile);
    //统计信息
    double simulation_duration = ssd->simulation_end_time - ssd->simulation_start_time;
    double simulation_duration_ms = simulation_duration / 1e6;
    double data_through_channel = ssd->read_request_c_a_size + ssd->read_request_size + ssd->rc_request_c_a_size + ssd->rc_request_c_a_size + ssd->rc_request_input_size + ssd->rc_request_result_size;
    double full_channel_bw = ssd->parameter->cxl_channel_number / (ssd->parameter->time_characteristics.tRC * 1e-9) * (simulation_duration * 1e-9);
    double full_computing_power = ssd->parameter->chip_computing_power * ssd->parameter->chip_num * 1e9 * (simulation_duration / 1e9);  // OPS
    printf(" Simulation Duration: \t\t%.0f ns = %.2f ms\n", simulation_duration, simulation_duration_ms);
    // printf(" Simulation Duration: %d ns = %.2fus\n", simulation_duration, simulation_duration_ms);
    // printf(" IOPS: %.3f\n", (double)(ssd->read_count+ssd->program_count)/((double)(ssd->simulation_end_time - ssd->simulation_start_time)/1000000000));
    printf(" READ BW: \t\t\t%.3f MB/s\n", ((double)ssd->read_request_size / 1024 / 1024)/ ((double)(ssd->simulation_end_time - ssd->simulation_start_time)/1000000000));
    //printf(" READ CA BW: \t\t\t%.3f MB/s\n", ((double)ssd->read_request_c_a_size / 1024 / 1024)/ ((double)(ssd->simulation_end_time - ssd->simulation_start_time)/1000000000));
    printf(" READ COMPUTE INPUT BW: \t%.3f MB/s\n", ((double)ssd->rc_request_input_size / 1024 / 1024)/ ((double)(ssd->simulation_end_time - ssd->simulation_start_time)/1000000000));    
    //printf(" READ COMPUTE CA BW: \t\t%.3f MB/s\n", ((double)ssd->rc_request_c_a_size / 1024 / 1024)/ ((double)(ssd->simulation_end_time - ssd->simulation_start_time)/1000000000)); 
    printf(" READ COMPUTE RESULT BW: \t%.3f MB/s\n", ((double)ssd->rc_request_result_size / 1024 / 1024)/ ((double)(ssd->simulation_end_time - ssd->simulation_start_time)/1000000000)); 
    printf(" CHANNEL USAGE: \t\t%.3f %%\n", (data_through_channel / full_channel_bw * 100));
    printf("\n");
    printf(" SSD OPS: \t\t\t%.3f GOPS\n", ((double)ssd->rc_request_compute_ops / 1024 / 1024 / 1024));
    printf(" SSD PE USAGE: \t\t\t%.3f %%\n", ((double)ssd->rc_request_compute_ops) / full_computing_power * 100);    
    printf("\n");
    printf(" write BW: %.3f MB/s\n", ((double)ssd->write_request_size / 1024) / ((double)(ssd->simulation_end_time - ssd->simulation_start_time)/1000000000));
}


/***********************************************************************************
 *根据每一页的状态计算出每一需要处理的子页的数目，也就是一个子请求需要处理的子页的页数
 *Calculate the number of subpages that need to be processed according to the state of each page, that is, the number of pages of subpages that a subrequest needs to process.
 ************************************************************************************/
unsigned int size(unsigned int stored)
{
    unsigned int i,total=0,mask=0x80000000;

#ifdef DEBUG
    printf("enter size\n");
#endif
    for(i=1;i<=32;i++)
    {
        if(stored & mask) total++;
        stored<<=1;
    }
#ifdef DEBUG
    printf("leave size\n");
#endif
    return total;
}


/*********************************************************
 *transfer_size()函数的作用就是计算出子请求的需要处理的size
 *函数中单独处理了first_lpn，last_lpn这两个特别情况，因为这
 *两种情况下很有可能不是处理一整页而是处理一页的一部分，因
 *为lsn有可能不是一页的第一个子页。
 *********************************************************/
unsigned int transfer_size(struct cxl_switch_device_info *ssd, int need_distribute, unsigned int lpn, struct request *req)
{
    unsigned int first_lpn,last_lpn,state,trans_size;
    unsigned int mask=0,offset1=0,offset2=0;

    first_lpn=req->lsn/ssd->parameter->subpage_page;
    last_lpn=(req->lsn+req->size-1)/ssd->parameter->subpage_page;

    mask=~(0xffffffff<<(ssd->parameter->subpage_page));
    state=mask;
    if(lpn==first_lpn)
    {
        offset1=ssd->parameter->subpage_page-((lpn+1)*ssd->parameter->subpage_page-req->lsn);
        state=state&(0xffffffff<<offset1);
    }
    if(lpn==last_lpn)
    {
        offset2=ssd->parameter->subpage_page-((lpn+1)*ssd->parameter->subpage_page-(req->lsn+req->size));
        state=state&(~(0xffffffff<<offset2));
    }

    trans_size=size(state&need_distribute);

    return trans_size;
}


/**********************************************************************************************************  
 *int64_t find_nearest_event(struct cxl_switch_device_info *ssd)
 *寻找所有子请求的最早到达的下个状态时间,首先看请求的下一个状态时间，如果请求的下个状态时间小于等于当前时间，
 *说明请求被阻塞，需要查看channel或者对应die的下一状态时间。Int64是有符号 64 位整数数据类型，值类型表示值介于
 *-2^63 ( -9,223,372,036,854,775,808)到2^63-1(+9,223,372,036,854,775,807 )之间的整数。存储空间占 8 字节。
 *channel,die是事件向前推进的关键因素，三种情况可以使事件继续向前推进，channel，die分别回到idle状态，die中的
 *读数据准备好了
 *Find the next state time of the earliest arrival of all sub-requests, first look at the next state time of the request, if the next state time of the request is less than or equal to the current time,
 *Indicates that the request is blocked. You need to check the channel or the next status time of the corresponding die. Int64 is a signed 64-bit integer data type, and the value type indicates that the value is between
 *An integer between -2^63 (-9,223,372,036,854,775,808) to 2^63-1 (+9,223,372,036,854,775,807). The storage space is 8 bytes.
 *Channel, die is the key factor for the event to advance. Three situations can make the event continue to move forward. The channel and die return to the idle state respectively.
 *Read the data is ready
 ***********************************************************************************************************/
int64_t find_nearest_event_sys(struct cxl_switch_device_info *ssd)
{
    unsigned int i,j;
    int64_t nearest_event_time=MAX_INT64;
    int64_t time_ssd=MAX_INT64;
    int64_t time_npu=MAX_INT64;

    struct npu_request *npu_req = NULL;

    // find nearest event in ssd
    time_ssd = find_nearest_event(ssd);

    // find nearest event in npu
    npu_req = ssd->npu->npu_reqs_head;
    while (npu_req!=NULL){
        if (npu_req->end_time <= time_npu){
            if (npu_req->end_time >= ssd->current_time){
                time_npu = npu_req->end_time;
            }
        }
        npu_req = npu_req->next_node;
    }

    nearest_event_time = (time_ssd > time_npu) ? time_npu : time_ssd;

    // update sys time
    if (nearest_event_time != MAX_INT64)
    {
        ssd->current_time=nearest_event_time;       
        ssd->simulation_end_time = ssd->current_time;    
    }

    return 0;
}

/**********************************************************************************************************
 遍历所有的channel，找到最早的事件时间，包括Read事件、Write事件、ReadCompute事件
 ***********************************************************************************************************/
int64_t find_nearest_event(struct cxl_switch_device_info *ssd) {
    int64_t time=MAX_INT64;

    for (int i = 0; i < ssd->parameter->cxl_channel_number; i++) {
        struct sub_request *sub = ssd->bottom_channel_head[i].subs_r_head;
        while (sub != NULL) {
            if (sub->next_state_predict_time <= time && sub->next_state_predict_time > ssd->current_time) {
                time = sub->next_state_predict_time;
            }
            sub = sub->next_node;
        }
        sub = ssd->bottom_channel_head[i].subs_w_head;
        while (sub != NULL) {
            if (sub->next_state_predict_time <= time && sub->next_state_predict_time > ssd->current_time) {
                time = sub->next_state_predict_time;
            }
            sub = sub->next_node;
        }
        sub = ssd->bottom_channel_head[i].subs_rc_head;
        while (sub != NULL) {
            if (sub->next_state_predict_time <= time && sub->next_state_predict_time > ssd->current_time) {
                time = sub->next_state_predict_time;
            }
            sub = sub->next_node;
        }
    }
    return time;
}

/*********************************************************************************************
 *slice_request()函数是处理当ssd没有dram的时候，
 *这是读写请求就不必再需要在buffer里面寻找，直接利用creat_sub_request()函数创建子请求，再处理。
 *The slice_request() function is used when ssd has no dram. This is a read/write request. 
 *You don't need to look in the buffer. You can use the creat_sub_request() function to create 
 *a subrequest and then process it.
 *********************************************************************************************/
struct cxl_switch_device_info *slice_request(struct cxl_switch_device_info *ssd) {
    struct request *req=NULL;
    struct sub_request *sub=NULL;
    unsigned int sub_size=0;

    ssd->dram->current_time=ssd->current_time;
    req=ssd->request_tail;

    if(req->operation==READ)        
    {		
        // unsigned int round = req->size / (ssd->parameter->subpage_page * ssd->parameter->chip_num);
        sub_size = ssd->parameter->subpage_page;
        // 记录下不同的cxl channel对应的的读请求，建立一个map，key为cxl_channel，value为组成sub_request的idx的链表
        unsigned int used_idx[req->addr_num];
        memset(used_idx, 0, sizeof(used_idx));
        for (int i=0; i<req->addr_num; i++){
            if (used_idx[i] == 1) continue;
            used_idx[i] = 1;
            // 新建一个大小为req->addr_num的addr_info**数组，用于重新存储一个读请求中的所有的cxl_channel相同的addr_info
            struct addr_info ** temp_addr = (struct addr_info **)malloc(sizeof(struct addr_info *) * req->addr_num);
            int temp_addr_num = 0; // number of addr on same cxl device
            temp_addr[temp_addr_num++] = &req->addr[i];
            for (int j=i+1; j<req->addr_num; j++){
                if (used_idx[j] == 1) continue;
                if (req->addr[j].cxl_id == req->addr[i].cxl_id){
                    temp_addr[temp_addr_num++] = &req->addr[j];
                    used_idx[j] = 1;
                }
            }
            // 为每个cxl channel的读请求创建sub_request
            // sub = creat_sub_request_simple(ssd, temp_addr, temp_addr_num, 0, sub_size, req);

            // slice the request into sub_requests based on dram_channel_number
            unsigned int sub_req_num = (temp_addr_num + ssd->parameter->dram_channel_num - 1) / ssd->parameter->dram_channel_num;
            int temp_temp_addr_num = 0;
            for (int ii=0; ii<sub_req_num; ii++){
                if ((ii < sub_req_num -1) || (temp_addr_num % ssd->parameter->dram_channel_num == 0)){
                    temp_temp_addr_num = ssd->parameter->dram_channel_num;
                }else{
                    temp_temp_addr_num = temp_addr_num % ssd->parameter->dram_channel_num;
                }
                struct addr_info ** temp_temp_addr = (struct addr_info **)malloc(sizeof(struct addr_info *) * temp_temp_addr_num);
                for (int jj=0; jj<temp_temp_addr_num; jj++){
                    temp_temp_addr[jj] = temp_addr[jj + ii * ssd->parameter->dram_channel_num];
                }
                sub = creat_sub_request_simple(ssd, temp_temp_addr, temp_temp_addr_num, 0, sub_size, req);
            }
            
        }
    }
    else if(req->operation==READ_COMPUTE)
    {   
        // unsigned int round = req->size / (ssd->parameter->subpage_page * ssd->parameter->chip_num);
        sub_size = ssd->parameter->subpage_page;
        // 记录下不同的cxl channel对应的的读请求，建立一个map，key为cxl_channel，value为组成sub_request的idx的链表
        unsigned int used_idx[req->addr_num];
        memset(used_idx, 0, sizeof(used_idx));
        for (int i=0; i<req->addr_num; i++){
            if (used_idx[i] == 1) continue;
            used_idx[i] = 1;
            // 新建一个大小为req->addr_num的addr_info**数组，用于重新存储一个读请求中的所有的cxl_channel相同的addr_info
            struct addr_info ** temp_addr = (struct addr_info **)malloc(sizeof(struct addr_info *) * req->addr_num);
            int temp_addr_num = 0;
            temp_addr[temp_addr_num++] = &req->addr[i];
            for (int j=i+1; j<req->addr_num; j++){
                if (req->addr[j].cxl_id == req->addr[i].cxl_id){
                    temp_addr[temp_addr_num++] = &req->addr[j];
                    used_idx[j] = 1;
                }
            }
            // 为每个cxl channel的读请求创建sub_request
            // sub = creat_sub_request_simple(ssd, temp_addr, temp_addr_num, 0, sub_size, req);

            // slice the request based on dram_channel_number
            unsigned int sub_req_num = (temp_addr_num + ssd->parameter->dram_channel_num - 1) / ssd->parameter->dram_channel_num;
            int temp_temp_addr_num = 0;
            for (int ii=0; ii<sub_req_num; ii++){
                if ((ii < sub_req_num -1) || (temp_addr_num % ssd->parameter->dram_channel_num == 0)){
                    temp_temp_addr_num = ssd->parameter->dram_channel_num;
                } else{
                    temp_temp_addr_num = temp_addr_num % ssd->parameter->dram_channel_num;
                }
                struct addr_info ** temp_temp_addr = (struct addr_info **)malloc(sizeof(struct addr_info *) * temp_temp_addr_num);
                for (int jj=0; jj<temp_temp_addr_num; jj++){
                    temp_temp_addr[jj] = temp_addr[jj + ii * ssd->parameter->dram_channel_num];
                }
                sub = creat_sub_request_simple(ssd, temp_temp_addr, temp_temp_addr_num, 0, sub_size, req);
            }


        }
    }
    else if(req->operation==WRITE)
    {
       // unsigned int round = req->size / (ssd->parameter->subpage_page * ssd->parameter->chip_num);
        sub_size = ssd->parameter->subpage_page;
        // 记录下不同的cxl channel对应的的读请求，建立一个map，key为cxl_channel，value为组成sub_request的idx的链表
        unsigned int used_idx[req->addr_num];
        memset(used_idx, 0, sizeof(used_idx));
        for (int i=0; i<req->addr_num; i++){
            if (used_idx[i] == 1) continue;
            used_idx[i] = 1;
            // 新建一个大小为req->addr_num的addr_info**数组，用于重新存储一个读请求中的所有的cxl_channel相同的addr_info
            struct addr_info ** temp_addr = (struct addr_info **)malloc(sizeof(struct addr_info *) * req->addr_num);
            int temp_addr_num = 0;
            temp_addr[temp_addr_num++] = &req->addr[i];
            for (int j=i+1; j<req->addr_num; j++){
                if (req->addr[j].cxl_id == req->addr[i].cxl_id){
                    temp_addr[temp_addr_num++] = &req->addr[j];
                    used_idx[j] = 1;
                }
            }
            // 为每个cxl channel的读请求创建sub_request
            // sub = creat_sub_request_simple(ssd, temp_addr, temp_addr_num, 0, sub_size, req);

            // slice the request based on dram_channel_number
            unsigned int sub_req_num = (temp_addr_num + ssd->parameter->dram_channel_num - 1) / ssd->parameter->dram_channel_num;
            int temp_temp_addr_num = 0;
            for (int ii=0; ii<sub_req_num; ii++){
                if ((ii < sub_req_num -1) || (temp_addr_num % ssd->parameter->dram_channel_num == 0)){
                    temp_temp_addr_num = ssd->parameter->dram_channel_num;
                } else{
                    temp_temp_addr_num = temp_addr_num % ssd->parameter->dram_channel_num;
                }
                struct addr_info ** temp_temp_addr = (struct addr_info **)malloc(sizeof(struct addr_info *) * temp_temp_addr_num);
                for (int jj=0; jj<temp_temp_addr_num; jj++){
                    temp_temp_addr[jj] = temp_addr[jj + ii * ssd->parameter->dram_channel_num];
                }
                sub = creat_sub_request_simple(ssd, temp_temp_addr, temp_temp_addr_num, 0, sub_size, req);
            }
        
        }
    }

    return ssd;
}

void display_title() 
{
    printf("\n");
    printf("               _     _             \n");
    printf("              | |   (_)            \n");
    printf("   ___ ___  __| |___ _ _ __ ___    \n");
    printf("  / __/ __|/ _` / __| | '_ ` _ \\   \n");
    printf("  \\__ \\__ \\ (_| \\__ \\ | | | | | |  \n");
    printf("  |___/___/\\__,_|___/_|_| |_| |_|  \n");
    printf("                                   \n");
    printf("  SSD internal simulation tool,\n  created by Yang Hu, v.2.0. Modified by Fadhil Kurnia \n\n");
}

void display_help() 
{
    printf("  usage: ssd [options] trace_file\n");
    printf("    options:\n");
    printf("     --timestamp <time> \t 15 chars timestamp used for log directory name (e.g: 20190214_220000), the default is current time\n");
    printf("     --parameter <filename> \t parameter filename (default: page.parameter)\n");
    printf("     --raid0 \t\t\t run raid 0 simulation\n");
    printf("     --raid5 \t\t\t run raid 5 simulation\n");
    printf("     --ndisk <num_disk> \t number of disk for raid simulation\n\n");
    printf("     --file <filename> \t\t trace file name\n");
}

void display_simulation_intro(struct cxl_switch_device_info *ssd)
{
    printf("\n\nBegin simulating ... ... ... ...\n");
    printf("  -parameter file: %s\n",ssd->parameterfilename); 
    printf("  -trace file    : %s\n",ssd->tracefilename);
    printf("\n\n   ^o^    OK, please wait a moment, and enjoy music and coffee   ^o^    \n\n");
}

void close_file(struct cxl_switch_device_info *ssd)
{
    if (ssd->tracefile!=NULL) fclose(ssd->tracefile);
    if (ssd->outputfile) fclose(ssd->outputfile);
    if (ssd->statisticfile) fclose(ssd->statisticfile);
    if (ssd->statisticfile2) fclose(ssd->statisticfile2);
    if (ssd->outfile_gc) fclose(ssd->outfile_gc);
}

// Get current time in string for log directory name
void get_current_time(char *current_time) {
    time_t timer;
    struct tm* tm_info;

    time(&timer);
    tm_info = localtime(&timer);

    strftime(current_time, 26, "%Y%m%d_%H%M%S", tm_info);
}
