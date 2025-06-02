# for fucking conda environment
unset CONDA_PREFIX
unset LD_LIBRARY_PATH

export PATH=$(echo $PATH | tr ':' '\n' | grep -v anaconda | tr '\n' ':' | sed 's/:$//')


mkdir -p build
cd build
cmake ..
make
./I_render
mv I_render ../demo/I_render
cd ..


