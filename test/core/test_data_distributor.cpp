#include "data_distributor.h"
#include <gtest/gtest.h>

TEST(DataDistributor, Constructor)
{
    AIxelerator::DataDistributor dd {};
    int device_count = dd.device_count();
}