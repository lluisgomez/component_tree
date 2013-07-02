#!/bin/bash
g++ -O3 -march='core2' -fpermissive `pkg-config opencv --cflags` -c component_tree.cpp -o component_tree.o

libtool --tag=CXX --mode=link g++ -O3 -march='core2' -o component_tree component_tree.o `pkg-config opencv --libs` 
