#ifndef AIXELERATORSERVICE_DATACONVERSIONSTRATEGY_H_
#define AIXELERATORSERVICE_DATACONVERSIONSTRATEGY_H_

#include <vector>

class DataConversionStrategy
{
    public:
        ~DataConversionStrategy() = default;

        // the concrete strategy should know what the actual type of field is
        virtual void fieldsToTensor(std::vector<void*> fields, double* tensor);
        virtual void tensorToFields(double* tensor, std::vector<void*> fields);

    protected:
};

#endif