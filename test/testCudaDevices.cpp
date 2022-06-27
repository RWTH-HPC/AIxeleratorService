#include <stdio.h>
#include <dlfcn.h>
//#include <cuda_runtime_api.h>

int main(int argc, char* argv[])
{
    int count = -1337;
    void* handle = dlopen("libcudart.so", RTLD_LAZY);
    if(handle == NULL)
    {
        count = 0;
        printf("Could not load cuda runtime library --> device count = 0.\n");
    }
    else{
        void (*getDeviceCount)(int*) = (void(*)(int*)) dlsym(handle, "cudaGetDeviceCount");
        //cudaGetDeviceCount(&count);  
        getDeviceCount(&count);  
    }
    
    printf("Number of GPUs: %d\n", count);   
}
