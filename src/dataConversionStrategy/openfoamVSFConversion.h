#ifndef AIXELERATORSERVICE_DATACONVERSIONSTRATEGY_OPENFOAMVSFCONVERSION_H_
#define AIXELERATORSERVICE_DATACONVERSIONSTRATEGY_OPENFOAMVSFCONVERSION_H_

#include "dataConversionStrategy/dataConversionStrategy.h"

class OpenfoamVSFConversion : public DataConversionStrategy
{
    public:
        void fieldsToTensor(std::vector<void*> fields, double* tensor) override;
        void tensorToFields(double* tensor, std::vector<void*> fields) override;

    private:
};

#endif