#include <dlfcn.h>
#include <iostream>

namespace aixelerator_service {
namespace utils {

inline int deviceCount()
{
    int count = -1;
    void* cuda_rt = dlopen("libcudart.so", RTLD_LAZY);
    void* veda_rt = dlopen("libveda.so.0", RTLD_LAZY);

    if(veda_rt != NULL)
    {  
        void (*getDeviceCount)(int*) = (void(*)(int*)) dlsym(veda_rt, "vedaDeviceGetCount");
        getDeviceCount(&count);

        return count;
    }

    if(cuda_rt != NULL)
    {
        void (*getDeviceCount)(int*) = (void(*)(int*)) dlsym(cuda_rt, "cudaGetDeviceCount");
        getDeviceCount(&count);  
        if ( count == -1)
        {
            count = 0;
        }

        return count;
    }

    std::cout << "Could not open LIBCUDA or LIBVEDA" << std::endl;
    count = 0;
    return count;
}

}
}