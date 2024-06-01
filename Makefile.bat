@echo off

cl /O2 /Ob3 /Ot /Ox /favor:blend /GL /Iinclude /std:c++17 /EHsc /LD NN.cpp /o output\NN.dll

@echo on