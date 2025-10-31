#include "distributionStrategy/gpuAffineRRDistribution.h"

#include "utils/deviceCount.h"

#include <iostream>
#include <numeric>
#include <dlfcn.h>

#include <vector>
#include <map>
#include <iostream>
#include <limits.h>
#include <unistd.h>
#include <sched.h>
#include <numa.h>
#include <numaif.h>
#include <fstream>
#include <string>

#include <sstream>
#include <array>
#include <cstdio>
#include <algorithm>

GPUAffineRRDistribution::GPUAffineRRDistribution(MPI_Comm app_comm) 
{
    app_comm_ = app_comm;
    work_group_comm_ = app_comm_;
    /*
     * if AIxeleratorService is used together with PhyDLL or MLLIB in an MPMD 
     * run (i.e. app_comm_ will be different from MPI_COMM_WORLD), then we do
     * not want to use the GPU but leave it for PhyDLL or MLLIB.
     */
    if (app_comm_ == MPI_COMM_WORLD) {
        createWorkgroups();
    }
    else{
        workgroup_size_ = 0;
        num_devices_total_ = 0;
        is_gpu_controller_ = false;
    }
}

GPUAffineRRDistribution::~GPUAffineRRDistribution()
{

}

void GPUAffineRRDistribution::createWorkgroups()
{
    int my_rank, num_procs;
    int err;
    MPI_Comm_rank(app_comm_, &my_rank);
    MPI_Comm_size(app_comm_, &num_procs);
    my_rank_ = my_rank;

    // figure out our local node rank
    MPI_Comm node_communicator;
    MPI_Comm_split_type(app_comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_communicator);
    int node_rank, node_size;
    MPI_Comm_rank(node_communicator, &node_rank);
    MPI_Comm_size(node_communicator, &node_size);
    MPI_Comm_free(&node_communicator);

    std::cout << "Rank " << my_rank << "/" << num_procs << " on its local machine is " << node_rank << "/" << node_size << std::endl; 

    int rank_numa_node = getRankNUMANode();
    std::vector<GPUInfo> gpus = discoverGPUs();

    std::cout << "Rank " << my_rank << " is bound to NUMA node " << rank_numa_node << std::endl;
    for (auto &gpu : gpus) {
        std::cout << "Rank " << my_rank << " GPU " << gpu.index << " is attached to NUMA node " << gpu.numa_node << std::endl;
    }


    // Default: not a GPU controller
    is_gpu_controller_ = false;
    my_gpu_device_ = -1;

    // Assign GPU controllers: lowest rank per NUMA node where a GPU lives
    for (auto &gpu : gpus) {
        // Find minimum MPI rank on this NUMA node
        int my_rank_if_same = (rank_numa_node == gpu.numa_node) ? my_rank : INT_MAX;
        int min_rank_on_node;
        MPI_Allreduce(&my_rank_if_same, &min_rank_on_node, 1,
                        MPI_INT, MPI_MIN, app_comm_);
        std::cout << my_rank << " " << min_rank_on_node << " " << gpu.numa_node << std::endl;
        if (my_rank == min_rank_on_node) {
            is_gpu_controller_ = true;
            my_gpu_device_ = gpu.index;
            std::cout << "Rank " << my_rank << " is GPU controller for GPU " << gpu.index << " on NUMA node " << gpu.numa_node << std::endl;
        }
    }
    
    // figure out the total number of GPU controllers
    num_devices_total_ = 0;
    int my_num_devices = is_gpu_controller_ ? 1 : 0;
    MPI_Allreduce(&my_num_devices, &num_devices_total_, 1, MPI_INT, MPI_SUM, app_comm_);
    std::cout << "Rank " << my_rank << "/" << num_procs << " knows that there is a total of " << num_devices_total_ << " GPUs across all systems" << std::endl;

    if ( num_devices_total_ > 0)
    {
        // combine controllers (and workers) into separate communicators, to enumerate them
        MPI_Comm work_type_comm;
        MPI_Comm_split(app_comm_, my_num_devices, my_rank, &work_type_comm);
        int work_type_rank, work_type_size;
        MPI_Comm_rank(work_type_comm, &work_type_rank);
        MPI_Comm_size(work_type_comm, &work_type_size);
        MPI_Comm_free(&work_type_comm);

        // round robin assignment of data to a gpu, making sure the gpu master is rank 0
        int color = work_type_rank % num_devices_total_;
        int order = is_gpu_controller_ ? 0 : my_rank + num_devices_total_;
        std::cout << "Rank " << my_rank << "/" << num_procs << " will be in group " << color << " order " << order << std::endl;

        // initialize the work group communicator
        err = MPI_Comm_split(app_comm_, color, order, &work_group_comm_);
        if ( err != MPI_SUCCESS )
        {
            std::cout << "Rank " << my_rank << ": Error when splitting workgroup communicator " << work_group_comm_  << std::endl;
        }
        int work_group_rank, work_group_size;
        MPI_Comm_rank(work_group_comm_, &work_group_rank);
        MPI_Comm_size(work_group_comm_, &work_group_size);
        workgroup_size_ = work_group_size;
        std::cout << "Rank " << my_rank << "/" << num_procs << " got work group rank id " << work_group_rank << "/" << work_group_size << std::endl;
    }
    else
    {
        workgroup_size_ = 0;
        work_group_comm_ = app_comm_;
    } 
}

int GPUAffineRRDistribution::getRankNUMANode() {
    // Get the current CPU the thread is running on
    int cpu = sched_getcpu();
    if (cpu < 0) {
        perror("sched_getcpu failed");
        return -1;
    }

    // Get the NUMA node of that CPU
    int node = numa_node_of_cpu(cpu);
    if (node < 0) {
        perror("numa_node_of_cpu failed");
        return -1;
    }

    return node;
}

std::vector<GPUInfo> GPUAffineRRDistribution::discoverGPUs() {
    std::vector<GPUInfo> gpus;

    int count = aixelerator_service::utils::deviceCount();
    std::cout << "Found " << count << " devices" << std::endl;
    if (count <= 0) {
        std::cerr << "No devices found.\n";
        return gpus;
    }

    void* cuda_rt = dlopen("libcudart.so", RTLD_LAZY);
    void* veda_rt = dlopen("libveda.so.0", RTLD_LAZY);

    if (veda_rt != nullptr) {
        // VEDA devices: currently not implemented
        std::cerr << "VEDA runtime detected; NUMA node detection not implemented.\n";
        dlclose(veda_rt);
        return gpus;
    }

    if (cuda_rt == nullptr) {
        std::cerr << "Could not open libcudart.so\n";
        return gpus;
    }

    using cudaDeviceGetPCIBusId_t = int(*)(char*, int, int);
    auto cudaDeviceGetPCIBusId = (cudaDeviceGetPCIBusId_t)dlsym(cuda_rt, "cudaDeviceGetPCIBusId");
    if (!cudaDeviceGetPCIBusId) {
        std::cerr << "Could not find cudaDeviceGetPCIBusId\n";
        dlclose(cuda_rt);
        return gpus;
    }

    // Run lstopo and capture its text output
    FILE* pipe = popen("lstopo", "r");
    if (!pipe) {
        std::cerr << "Failed to run lstopo\n";
        dlclose(cuda_rt);
        return gpus;
    }

    std::vector<std::string> topo_lines;
    char buffer[2048];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        topo_lines.emplace_back(buffer);
    }
    pclose(pipe);
    
    // helper: lowercase a string
    auto toLower = [](std::string s) -> std::string {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
        return s;
    };

    // helper: extract the 'tail' of a PCI id: "0000:65:00.0" -> "65:00.0", "65:00.0" -> "65:00.0"
    auto extract_pci_tail = [](const std::string &s) -> std::string {
        // trim
        size_t start = s.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\n\r");
        std::string t = s.substr(start, end - start + 1);

        size_t pos_dot = t.find_last_of('.');
        if (pos_dot == std::string::npos) {
            // no dot -> fallback to full trimmed string
            return t;
        }
        // find the ':' before the dot (device:function separator) and the previous ':' (domain separator)
        size_t pos_colon_before = t.rfind(':', pos_dot);
        if (pos_colon_before == std::string::npos) return t;
        size_t pos_colon_prev = (pos_colon_before == 0) ? std::string::npos : t.rfind(':', pos_colon_before - 1);
        size_t start_idx = (pos_colon_prev == std::string::npos) ? 0 : (pos_colon_prev + 1);
        return t.substr(start_idx);
    };

    // For each GPU (via CUDA), find its PCI ID and then find the nearest NUMA node ABOVE it in lstopo output
    for (int i = 0; i < count; ++i) {
        char pciBusId[64] = {0};
        if (cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), i) != 0) {
            std::cerr << "Failed to get PCI Bus ID for GPU " << i << "\n";
            gpus.push_back({i, -1});
            continue;
        }
        std::string pciIdOrig = pciBusId;
        std::string pciTail = extract_pci_tail(pciIdOrig);
        std::string pciTailLower = toLower(pciTail);
        std::string pciOrigLower = toLower(pciIdOrig);

        std::cout << "pciBusId " << pciIdOrig << " of gpu " << i << std::endl;

        int found_index = -1;
        // find the first line containing either the full pci id or the tail
        for (int l = 0; l < (int)topo_lines.size(); ++l) {
            std::string low = toLower(topo_lines[l]);
            if (low.find(pciTailLower) != std::string::npos || low.find(pciOrigLower) != std::string::npos) {
                found_index = l;
                break;
            }
        }

        int numa_node = -1;
        if (found_index != -1) {
            // NUMA node must be above: search upward (towards beginning of file)
            for (int j = found_index; j >= 0; --j) {
                std::string low = toLower(topo_lines[j]);
                size_t pos = low.find("numanode");
                if (pos != std::string::npos) {
                    int node = -1;
                    const char* c = low.c_str() + pos;
                    // try formats: "NUMANode L#%d" or "NUMANode %d"
                    if (sscanf(c, "numanode l#%d", &node) == 1 || sscanf(c, "numanode %d", &node) == 1) {
                        numa_node = node;
                        break;
                    }
                }
                // keep searching upward until start of the output (no artificial small range)
            }
        } else {
            std::cerr << "Could not find PCI ID " << pciIdOrig << " in lstopo output\n";
        }

        std::cout << "Normalized pciBusId " << pciTail << " of gpu " << i << " -> NUMA node " << numa_node << std::endl;
        gpus.push_back({i, numa_node});
    }
    
    dlclose(cuda_rt);
    return gpus;
}
