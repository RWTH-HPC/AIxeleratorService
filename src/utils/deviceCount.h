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
        std::cout << "Could not open libcuda" << std::endl;
        count = 0;
    }
    else{
        std::cout << "Successfully opened libcuda" << std::endl;
        void (*getDeviceCount)(int*) = (void(*)(int*)) dlsym(cuda_rt, "cudaGetDeviceCount");
        getDeviceCount(&count);  
    }
    return count;
}

}
}