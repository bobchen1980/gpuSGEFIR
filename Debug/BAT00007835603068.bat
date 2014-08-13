@echo off
echo "d:\CUDA\bin\nvcc.exe"   -arch sm_10 -ccbin "C:\Program Files\Microsoft Visual Studio 8\VC\bin"    -Xcompiler "/EHsc /W3 /nologo /Od /Zi   /MTd  " -I"d:\CUDA\include" -I"../../common/inc" -maxrregcount=32  --compile -o "Debug\gpuSGEFIR.cu.obj" gpuSGEFIR.cu 
 "d:\CUDA\bin\nvcc.exe"   -arch sm_10 -ccbin "C:\Program Files\Microsoft Visual Studio 8\VC\bin"    -Xcompiler "/EHsc /W3 /nologo /Od /Zi   /MTd  " -I"d:\CUDA\include" -I"../../common/inc" -maxrregcount=32  --compile -o "Debug\gpuSGEFIR.cu.obj" "d:\Cuda code\gpuSGEFIR\gpuSGEFIR.cu"
if errorlevel 1 goto VCReportError
goto VCEnd
:VCReportError
echo Project : error PRJ0019: A tool returned an error code from "Compiling with CUDA Build Rule..."
exit 1
:VCEnd