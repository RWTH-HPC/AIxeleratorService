#pragma once
#include <iostream>
#include <type_traits>
#include <vector>

namespace AIxelerator {

void from_int(std::vector<float> vec)
{
    for (auto& e : vec) {
        std::cout << e << '\n';
    }
}

void from_int(std::vector<int> vec)
{
    for (auto& e : vec) {
        std::cout << e << '\n';
    }
}

// T - type from flow filed
// U - type to ML framework
template <typename T, typename U, typename Dtype = double>
class Adapter {
private:
    /* data */
public:
    Adapter(/* args */)
    {
    }
    ~Adapter()
    {
    }

    // Only accept containers as argument compatible with ranged-based for loops
    template <template <typename, typename...> class Container, typename... Args>
    auto from(const Container<T, Args...>& c) -> std::enable_if_t<std::is_same_v<decltype(std::begin(c)), decltype(std::end(c))>>
    {
        from_int(c);
    }
};

// template<typename T, typename U>
// Adapter::Adapter(/* args */)
// {
// }

// template<typename T, typename U>
// Adapter::~Adapter()
// {
// }

// Not correct syntax right now for function declaration
// template<typename T, typename U, typename Dtype>
// template <typename Container>
// auto Adapter<T, U, Dtype>::from(const Container& c) -> std::enable_if_t<std::is_same_v<decltype(std::begin(c)), decltype(std::end(c))>>
// {

//     ret = std::vector<Dtype*> = {};
//     return ret;
// }

} // namespace AIxelerator
