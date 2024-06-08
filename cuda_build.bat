@echo off
nvcc CudaCore_v2.cu -O3 -ICudaInclude --shared -o CudaCore.dll
pause