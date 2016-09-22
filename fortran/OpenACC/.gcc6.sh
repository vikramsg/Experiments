export CUDA_HOME=/usr/local/cuda
export GCC6ROOT=/home/vikram/Experiments/Nvidia/OpenACC/OLCFHack15/gcc6
export OMPROOT=/home/vikram/usr/local/openmpi-coarray
export PATH=$GCC6ROOT/install/bin:$CUDA_HOME/bin/:$OMPROOT/bin:$PATH
export LIBRARY_PATH=$GCC6ROOT/install/lib64:$OMPROOT/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$GCC6ROOT/install/lib64:$GCC6ROOT/depends/mpc/lib/:$GCC6ROOT/depends/mpfr/lib:$GCC6ROOT/depends/gmp/lib:$OMPROOT/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$GCC6ROOT/install/include/c++:$CPLUS_INCLUDE_PATH
export CPATH=$GCC6ROOT/install/include/c++/6.2.0/:$GCC6ROOT/install/include/c++/6.2.0/x86_64-pc-linux-gnu/:$CPATH
export MANPATH=$GCC6ROOT/software/share/man:$MANPATH

#CAF PATH
if [[ -z "$PATH" ]]; then                                         
  export PATH="/home/vikram/Downloads/Installs/src/OpenCoarrays-1.7.2/prerequisites/installations/cmake/3.4.0/bin"                            
else                                                                 
  export PATH="/home/vikram/Downloads/Installs/src/OpenCoarrays-1.7.2/prerequisites/installations/cmake/3.4.0/bin":$PATH                     
fi                                                                   
if [[ -z "$PATH" ]]; then                                         
  export PATH="/home/vikram/usr/local/opencoarray/bin"                     
else                                                                 
  export PATH="/home/vikram/usr/local/opencoarray/bin":$PATH              
fi                                                                   
