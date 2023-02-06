#include "dataConversionStrategy/openfoamVSFConversion.h"

#include <volFieldsFwd.H> // OpenFOAM header for volScalarField

// field is actually an OpenFOAM volScalarField
void OpenfoamVSFConversion::fieldsToTensor(std::vector<void*> fields, double* tensor)
{
    unsigned int num_fields = fields.size();
    for(unsigned int j = 0; j < num_fields; j++)
    {
        volScalarField* field_ptr = (volScalarField*) fields[j];
        const volScalarField &field = *field_ptr;
        unsigned int num_cells = field.size();
        for(unsigned int k = 0; k < num_cells; k++)
        {
            tensor[j+k*num_fields] = field[k];
        }
    }
}

// field is actually an OpenFOAM volScalarField
void OpenfoamVSFConversion::tensorToFields(double* tensor, std::vector<void*> fields)
{
    unsigned int num_fields = fields.size();
    for(unsigned int j = 0; j < num_fields; j++)
    {
        volScalarField* field_ptr = (volScalarField*) fields[j]; 
        volScalarField &field = *field_ptr; 
        unsigned int num_cells = field.size(); 
        for(unsigned int k = 0; k < num_cells; k++)
        {
            field[k] = tensor[j+k*num_fields];
        }   
    }
}