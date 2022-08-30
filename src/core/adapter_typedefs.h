#pragma once

#include "adapter.h"

// forward declaration
namespace foam {
class volScalarField;
}

namespace torch {
class Tensor;
}

namespace AIxelerator {

class foam::volScalarField;
class torch::Tensor;

using FoamTorchAdapter = Adapter<foam::volScalarField, torch::Tensor>;

} // namespace AIxelerator