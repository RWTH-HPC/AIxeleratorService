#include "data_distributor.h"

namespace AIxelerator {

template <>
DataDistributor<>::DataDistributor()
    : workGroupComm_(MPI_COMM_WORLD)
{
    // auto test = mpi::COMM_WORLD;
}

template <>
void DataDistributor<>::scatter()
{
}

template <>
void DataDistributor<>::gather()
{
}

template <>
int DataDistributor<>::device_count()
{
    return this->device_count_;
}

} // namespace AIxelerator