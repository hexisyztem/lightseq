if [ ! -d 'build' ]; then
    mkdir build
fi

cd build && cmake -DDEBUG_MODE=ON -DFP16_MODE=OFF .. && make -j${nproc}
