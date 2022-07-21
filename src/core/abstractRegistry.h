#pragma once

/*
 *
 *  EXAMPLE #1 - AIModel
 *
 *      // usage of the abstract factory
 *
 *  AIModel.h
 *      // define specific AIModelEntry and AIModelRegistry
 *      using AIModelEntry = abstractEntry<AIModelBase, std::string>;
 *      using AIModelRegistry = abstractRegistry<AIModelBase, AIModelEntry, std::string>;
 *
 *  TorchModel.cpp
 *      // register a specific AIModel (here: TorchModel)
 *      AIModelEntry entryTorchModel(&abstractFactory<AIModelBase, TorchModel, std::string>);
 *      AIModelRegistry addTorchModel("TorchModel", entrytorchModel);
 *
 *      //...NOTE: in this example 'entryTorchModel' and 'addTorchModel' are arbitrary names
 *
 *  yourCode.cpp
 *      // construct model by name
 *      model_ = AIModelRegistry::constructByName("TorchModel", std::string);
 *
 */

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>

namespace AIxelerator {
/*! \brief abstract factory function
 *
 *  metafunction, designed after the abstract factory pattern: creates a
 *  specific instance, but returns it as base class
 *
 *  template arguments:
 *
 *  @param _base base class (e.g. AIModel)
 *  @param _special specialised derived class (e.g. TorchModel, which inherits from AIModelBase)
 *  @param _constructTypes argument types which are required by the _base class' constructor
 *
 *  returns unique pointer to base class instance
 */
template <class _base, class _special, typename... _constructTypes>
std::unique_ptr<_base> abstractFactory(_constructTypes... constructArgs)
{
    return std::make_unique<_special>(_special(constructArgs...));
}

/*! \class abstractRegistry
 *  \brief provides meta registry for derived classes and allows their runtime selection
 *
 *  Abstract registry class which provides an infrastructure for registering
 *  and constructing derived classes (potentially from dynamically linked
 *  libraries). Objects are runtime selectable by registering to a unique,
 *  central object (registry = singleton) and making them available by their name
 *  throughout the whole code.
 *
 *  We use a static map (registry) as trojan horse into our library. Other
 *  modules (e.g. a specific AIModel) can register itself to this registry and
 *  will be constructable by name (arbitrary string) through this class. The
 *  returned type always is the base class (every derived AIModel IS A AIModel!)
 *
 *  More about the singleton design pattern:
 *      https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
 *
 *  Example: a model is dynamically linked to the code, after registration it
 *           is readily available by selecting it in a template by setting the
 *           modelType correspondingly
 *
 *
 *  template arguments:
 *
 *  @param _base base class (e.g. AIModelBase)
 *  @param _entry entry type, specific to base class (e.g. TorchModel)
 *  @param _constructTypes argument types which are required by the _base class' constructor
 *                        (which neccessarily also applies to all derived classes)
 */
template <class _base, class _entry, typename... constructTypes>
class abstractRegistry {
public:
    // \typedef entryList
    //          manages entries, which contain at least an abstract factory
    //          function which creates a derived class, but returns its base
    //          class (here: _base)
    using entryList = std::unordered_map<std::string, _entry>;
    // avoid unwanted copy-constructs etc. -> singleton design pattern
    abstractRegistry() = delete; // no default constructor
    abstractRegistry(abstractRegistry const&) = delete; // cannot copy
    void operator=(abstractRegistry const&) = delete; // cannot copy

    //!< constructor
    abstractRegistry(const std::string name, _entry e)
    {
        _entry temp_e(e);
        objects().insert({ name, temp_e });
    }
    static entryList l_; // guaranteed to be destroyed, instantiated on first use

    //!< returns list of entries, singleton pattern: entryList is created once
    static entryList& objects()
    {
        static entryList l; // guaranteed to be destroyed, instantiated on first use
        return l;
    }

    /*! \brief print contents of registry (all registered identifiers)
     *
     *  - for debugging and for user information if wrong identifier has been entered
     *  - list all identifiers for the user to choose
     */
    static void printContents()
    {
        std::cout << std::endl
                  << " REGISTRY INFO: " << std::endl
                  << std::endl;
        std::cout << "available options are:" << std::endl;
        std::cout << "(" << std::endl;
        for (auto& obj : objects())
            std::cout << "   " << obj.first.c_str() << std::endl;
        std::cout << ")" << std::endl;
    }

    /*! \brief construct instance of a registered class
     *
     *  - constructs specific base object (identified by its name)
     *  - use the factory function managed by its entry
     *
     *  returns shared_ptr to _base class, but constructs derived class instance (abstract factory pattern)
     */
    static std::unique_ptr<_base> constructByName(const std::string name, constructTypes... constructArgs)
    {
        auto obj = objects().find(name); // entry is a const_iterator
        if (obj != objects().end()) {
            // found entry, construct and return instance of derived class
            return obj->second.factory_(constructArgs...);
        }
        // did not find entry - list all available identifiers in registry, then throw
        printContents();
        std::string message = "registry::constructByName: object '" + name + "' does not exist";
        throw std::runtime_error(message);
    }
};

/*! \brief abstract entry type
 *
 *  Entry type, designed to be registered to a registry. Manages a factory
 *  function for instances of base class and all of its derived types.
 *
 *  template arguments:
 *
 *  @param _base base class (e.g. solver)
 *  @param _constructTypes argument types which are required by the _base class' constructor
 *
 *  example usage:
 *
 *  // define specific entry type
 *  using solverEntry = abstractEntry<solver, parser*, database*, solution*>;
 */
template <class _base, typename... constructTypes>
struct abstractEntry {
    // definition of the factory function type
    using factoryFunction = std::unique_ptr<_base> (*)(constructTypes...);
    // (public) class members
    factoryFunction factory_;
    // default constructors
    abstractEntry(factoryFunction f)
        : factory_(f)
    {
    }
    abstractEntry(const abstractEntry& e)
        : factory_(e.factory_)
    {
    }
};

} // namespace AIxelerator
