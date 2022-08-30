#include "adapter.h"
#include <gtest/gtest.h>

TEST(Adapter, Constructor)
{
    AIxelerator::Adapter<int, int> adapter_int {};

    std::vector<int> vec = { 1, 2, 3, 4, 5 };
    adapter_int.from(vec);

    AIxelerator::Adapter<float, float> adapter_float {};
    std::vector<float> vec2 = { 1, 2, 3, 4, 5 };
    adapter_float.from(vec2);
}
