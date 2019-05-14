%module dist_communicator

%{
#include "singa/dist/communicator.h"
%}

namespace singa{
  
void synch(Tensor &t1, Tensor &t2);

}