class SimulationDataVisitor
{
    public: 
        virtual ~SimulationDataVisitor() = default; 
        virtual void visit(SimulationData& obj){ obj.accept(*this)}
}