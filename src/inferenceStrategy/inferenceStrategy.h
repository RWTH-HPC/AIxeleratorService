#ifndef AIXELERATORSERVICE_INFERENCESTRATEGY_H_
#define AIXELERATORSERVICE_INFERENCESTRATEGY_H_

class InferenceStrategy
{
    public:
        virtual ~InferenceStrategy() = default;

        virtual void inference() = 0;
};

#endif