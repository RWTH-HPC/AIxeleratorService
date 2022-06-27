Do not link with cuda because on CPU nodes there is no cuda and startup of the executable will crash.
The will try to load cuda runtime shared library with dlopen to avoid this crash.
```
icpc -o testCudaDevices.x testCudaDevices.cpp
```