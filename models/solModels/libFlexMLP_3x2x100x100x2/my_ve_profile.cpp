// needs to be compiled with nc++ (.vcpp ending does not work for some reason)
#include <functional>
#include <cstdint>
uint64_t sol_ve_profile(std::function<void(void)> func, const int _runs, const int _notImproved){return 0;}
void sol_check(int, char const*, int) {}
