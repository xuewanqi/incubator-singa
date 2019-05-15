
#ifndef SINGA_DIST_COMMUNICATOR_H_
#define SINGA_DIST_COMMUNICATOR_H_

#include <iostream>
#include <cstdint>
#include <unistd.h>

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include "singa/core/tensor.h"

namespace singa{

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


class Communicator {
public:
  int MPIRankInGlobal;
  int totalMPIRanksInGlobal;
  int MPIRankInLocal;
  int nDev;
  cudaStream_t* s;
  ncclComm_t* comms;

  Communicator(int nDev);
  ~Communicator();
  void allReduce(int size, void** sendbuff, void** recvbuff);
  void wait();
};


void synch(Tensor &t1, Tensor &t2);
// void synch(Tensor &t1, Tensor &t2){
//   Communicator c(2);
//   void* addr1=t1.block()->mutable_data();
//   void* addr2=t2.block()->mutable_data();
  
//   void** addr;
//   addr[0]=addr1;
//   addr[1]=addr2;

//   c.allReduce(1, addr, addr);
//   c.wait();
// }

}
#endif