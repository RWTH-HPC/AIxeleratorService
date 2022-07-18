#pragma once

#include <map>
#include <string>

namespace AIxelerator
{

enum class AIFramework{pytorch, tensorflow};

template <typename T, typename U>
void print_map_content(std::map<T,U> map);

std::string aiframework_to_string(AIFramework framework);
std::string lookup_aiframework_by_filextension(const std::string& extension);

} // AIxelerator
