#ifndef AIXELERATORSERVICE_DATACONVERSIONSTRATEGY_H_
#define AIXELERATORSERVICE_DATACONVERSIONSTRATEGY_H_

class DataConversionStrategy
{
    public:
        ~DataConversionStrategy() = default;

        // the concrete strategy should know what the actual type of field is
        virtual void fieldToTensor(void* field, double* tensor);
        virtual void tensorToField(double* tensor, void* field);

    protected:
}

#endif