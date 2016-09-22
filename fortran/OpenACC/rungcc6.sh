#!/bin/bash
export CUDA_HOME=/usr/local/cuda
export GCC6ROOT=/home/vikram/Experiments/Nvidia/OpenACC/OLCFHack15/gcc6
export PATH=$GCC6ROOT/install/bin:$CUDA_HOME/bin/:$PATH
export LIBRARY_PATH=$GCC6ROOT/install/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$GCC6ROOT/install/lib64:$GCC6ROOT/depends/mpc/lib/:$GCC6ROOT/depends/mpfr/lib:$GCC6ROOT/depends/gmp/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$GCC6ROOT/install/include/c++:$CPLUS_INCLUDE_PATH
export MANPATH=$GCC6ROOT/software/share/man:$MANPATH
if [ "x$*" == "x" ]; then
echo "[GCC6 offload wrapper] rungcc6: <gcc command> <args....>"
echo "Following gcc commands in the path:"
cmds=`ls $GCC6ROOT/install/bin |grep -v "-"`
for file in $cmds
do echo -n "$file "
done
echo ";"
echo "Some examples of compilation:"
echo "a) using offload via openacc    -> rungcc6 gcc test-pi.c -fopenacc -foffload=nvptx-none -foffload="-O3" -O3 -o gpu.x"
echo "b) not using offload            -> rungcc6 gcc -O3 test-pi.c -o cpu.x"
exit;
else
exec $*
fi
