#include <iostream>
#include "AIModelUtils.h"

namespace AIxelerator
{

template <typename T, typename U>
void print_map_content(std::map<T,U> map)
{
    for(const auto& elem : map)
    {
        std::cout << elem.first << "\t" << elem.second << "\n";
    }
}

std::map<AIFramework, std::string> aiframework_string_map = {
    {AIFramework::pytorch, "pytorch"}, 
    {AIFramework::tensorflow, "tensorflow"}, 
};

std::string aiframework_to_string(AIFramework framework)
{
    auto search = aiframework_string_map.find(framework);
    if ( search == aiframework_string_map.end() )
    {   
        std::string message("Framework not available\n"); 
        throw std::runtime_error(message);
    }
    return search->second;
}

std::map<std::string, std::string> extension_aiframework_map = {
    {".pt", "pytorch"}, 
    {".pb", "tensorflow"}, 
};

std::string lookup_aiframework_by_filextension(const std::string& extension)
{
    auto search = extension_aiframework_map.find(extension);
    if ( search == extension_aiframework_map.end() )
    {   
        std::cerr << "Framework could not be detected with file-extension '" << extension << "'." << std::endl;
        std::cerr << "Available options are: \n" <<
                     "Extension\tFramework";
        //print_map_content(string_aiframework_map);             
    }
    return search->second;
}

} // AIxelerator
