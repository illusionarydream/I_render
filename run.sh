mkdir -p build
cd build
cmake ..
make
./I_render
mv I_render ../demo/I_render
cd ..


