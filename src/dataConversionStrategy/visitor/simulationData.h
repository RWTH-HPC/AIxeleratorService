#include "visitor.h"

class SimulationData
{
    public:
        virtual ~SimulationData() = default;
        virtual accept(SimulationDataVisitor visitor){}
}