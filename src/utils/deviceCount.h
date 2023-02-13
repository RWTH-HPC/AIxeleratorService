#include <dlfcn.h>
#include <iostream>

namespace aixelerator_service {
namespace utils {

int deviceCount()
{
    int count = -1;
    void* cuda_rt = dlopen("libcudart.so", RTLD_LAZY);
    if(cuda_rt == NULL)
    {
        count = 0;
        std::cout << "Could not open LIBCUDA" << std::endl;
    }
    else{
        std::cout << "Successfully opened LIBCUDA" << std::endl;
        void (*getDeviceCount)(int*) = (void(*)(int*)) dlsym(cuda_rt, "cudaGetDeviceCount");
        getDeviceCount(&count);  
        if ( count == -1)
        {
            count = 0;
        }
    }
    return count;
}

}
}