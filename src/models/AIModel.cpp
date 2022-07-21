#include <filesystem> // C++17
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "AIModel.h"

namespace fs = std::filesystem;

namespace AIxelerator {

AIModel::AIModel(AIFramework framework, std::string filename)
{
    std::string framework_name = aiframework_to_string(framework);
    pimpl = AIModelRegistry::constructByName(framework_name, filename);
}

AIModel::AIModel(std::string framework_name, std::string filename)
{
    pimpl = AIModelRegistry::constructByName(framework_name, filename);
}

AIModel::AIModel(std::string filename)
{
    std::string extension(fs::path(filename).extension());
    std::string framework = lookup_aiframework_by_filextension(extension);
    pimpl = AIModelRegistry::constructByName(framework, filename);
}

void AIModel::forward() const
{
    pimpl->forward();
}

// Special member functions
AIModel::AIModel(AIModel const& other)
{
}

AIModel& AIModel::operator=(AIModel other)
{
    std::swap(pimpl, other.pimpl);
    return *this;
}

} // namespace AIxelerator