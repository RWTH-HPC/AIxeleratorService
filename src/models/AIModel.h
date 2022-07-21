#pragma once

#include <filesystem> // C++17
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "AIModelUtils.h"
#include "abstractRegistry.h"

namespace fs = std::filesystem;

namespace AIxelerator {

class AIModelBase {
public:
    AIModelBase() = default;
    virtual ~AIModelBase() = default;

    virtual void forward() = 0;
};

// define registry
using AIModelEntry = abstractEntry<AIModelBase, std::string>;
using AIModelRegistry = abstractRegistry<AIModelBase, AIModelEntry, std::string>;

// convenience macro to register derived AIModels
#define REGISTER_AIMODEL(DERIVED_CLASS, CLASS_NAME)                                                            \
    AIModelEntry entry##DERIVED_CLASS(&AIxelerator::abstractFactory<AIModelBase, DERIVED_CLASS, std::string>); \
    AIModelRegistry add##DERIVED_CLASS(CLASS_NAME, entry##DERIVED_CLASS);

class AIModel {
private:
    friend void forward(AIModel const& model)
    {
        model.pimpl->forward();
    }

    std::unique_ptr<AIModelBase> pimpl;

public:
    AIModel() = delete;
    ~AIModel() = default;
    AIModel(AIFramework framework, std::string filename);
    AIModel(std::string framework_name, std::string filename);
    AIModel(std::string filename);

    void forward() const;

    // Special member functions
    AIModel(AIModel const& other);
    AIModel& operator=(AIModel other);

    AIModel(AIModel&& other) = default;
    AIModel& operator=(AIModel&& other) = default;
};

} // namespace AIxelerator