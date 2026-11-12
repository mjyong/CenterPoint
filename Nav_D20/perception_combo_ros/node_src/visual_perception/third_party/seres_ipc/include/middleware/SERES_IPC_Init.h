/*
 * Seres_IPC_Init.h
*/

#ifndef SERES_IPC_INIT_H_
#define SERES_IPC_INIT_H_

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/types.h>
#include <pthread.h>
#include <mqueue.h>

/************************************************************************/
/* Definitions                                                          */
/************************************************************************/

// #define DEBUG_LINE() printf("[%s:%s] line=%d\r\n",__FILE__, __func__, __LINE__)
// #define DEBUG_ERR(fmt, args...) printf("\033[47;31m[%s:%d]\033[0m   " fmt " \r\n", __func__, __LINE__,  ##args)
// #define DEBUG_INFO(fmt, args...) printf("\033[33m[%s:%d]\033[0m  " fmt "  \r\n", __func__, __LINE__, ##args)

#define DEBUG_LINE()
#define DEBUG_ERR(fmt, ...)
#define DEBUG_INFO(fmt, ...)
#define SERES_ERR(fmt, ...) printf("\033[47;31m[%s:%d]\033[0m   " fmt " \r\n", __func__, __LINE__,  ##__VA_ARGS__)

#define MAX_IPC_BUF_NUM 		150
#define MAX_IPC_PROCESS_NUM 	50
#define MAX_IPC_BUF_INDEX 		3
#define MAX_IPC_BUF_USERNUM 	10
#define SERES_IPC_MAX_MSG_NUM   10
#define SERES_IPC_BUF_CTL_SPACE	0x20

#define IPC_CREATE_NAME				"/usr/sbin/"

#define IPC_CONFIG_FILE_PATH 		"/opt/middle_ware/etc/com/SERES_IPC_Config.json"

/************************************************************************/
/* DataStructure Define                                                 */
/************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif

	typedef struct seres_msq_msg
	{
		unsigned int frame_id;
		unsigned int buf_id;
		unsigned int buf_idx;
		mqd_t send_mqd;
		unsigned int msg_type;		
	}seres_msq_msg;


	typedef struct SERES_IPC_PROCESS_Desc
	{
		char 				process_name[256];
		mqd_t 				msgq_id;
	}SERES_IPC_PROCESS_Desc;

	typedef struct SERES_IPC_BUF_Desc
	{
		char 				buf_name[256];
		unsigned int 		shm_key;
		unsigned int 		buf_size;
		unsigned int 		buf_num;
		unsigned int 		msg_type;
		unsigned int 		user_num;
		unsigned int 		version_idx;
		char 				user_name[MAX_IPC_BUF_USERNUM][256];
		mqd_t 				user_msgq[MAX_IPC_BUF_USERNUM];
		void				*buf_addr[MAX_IPC_BUF_INDEX];
		pthread_rwlock_t 	*buf_rwlock[MAX_IPC_BUF_INDEX];
	}SERES_IPC_BUF_Desc;

	extern SERES_IPC_BUF_Desc IPC_BUF_Sets[MAX_IPC_BUF_NUM]; 

	void app_info_init(void);
	
	int seres_get_buf_id_by_name(const char *buf_name);  // check return value 
	int seres_get_msq_id_by_name(const char *process_name);
	int seres_write_lock(unsigned int buf_id, unsigned int buf_idx);
	int seres_read_lock(unsigned int buf_id, unsigned int buf_idx);
	int seres_release_lock(unsigned int buf_id, unsigned int buf_idx);
	int seres_receive_ipc_msg(unsigned int* buf_id, unsigned int *frame_id, unsigned int *buf_idx);
	int seres_receive_ipc_msg_with_sender(unsigned int* buf_id, unsigned int *frame_id, unsigned int *buf_idx,int *send_msq,int *msg_type);
	int seres_send_ipc_msg_cpy(unsigned int buf_id, unsigned int frame_id, unsigned int buf_idx, void* buf_addr);
	int seres_send_ipc_msg(unsigned int buf_id, unsigned int frame_id, unsigned int buf_idx);
	int seres_send_ipc_msg_with_type(unsigned int buf_id, unsigned int frame_id, unsigned int buf_idx,int msg_type);	
	int seres_send_ipc_msg2special(int msq_idx,int msg_type);
	int seres_get_msg_num_in_my_msgq(unsigned int *msg_num);
	int seres_get_msg_num_in_process(const char *process_name, unsigned int *msg_num);
	int report_app_version(int version_major, int version_middle, int version_minor);
	int SERES_IPC_init(const char* process_name);
	int seres_Load_IPC_Config(unsigned char * fstream);
	int seres_check_ipc_version(int buf_idx,int buf_id_version);

	void SERES_IPC_release();
	int SERES_IPC_Init_down();

#ifdef __cplusplus
}
#endif

#endif /* SERES_IPC_INIT_H_ */
