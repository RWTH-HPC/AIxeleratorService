
#include <sol/dfp/ncc/api.h>
#include "ve_9BC88669.h"
struct ve_9BC88669_FI_dfp_ncc_6D_B8 {
	double*  F15_Output;
	const double*  F13_Output;
	const double*  F5_Output;
	inline ve_9BC88669_FI_dfp_ncc_6D_B8(double*  _F15_Output, const double*  _F13_Output, const double*  _F5_Output) : F15_Output(_F15_Output), F13_Output(_F13_Output), F5_Output(_F5_Output){}
	void operator()(const int32_t min, const int32_t max) const {

		#pragma _NEC vector
		for(int32_t IB8 = min; IB8 < max; IB8++) {
			const int32_t L2 = (IB8) % 2;
			double F14_Output_s = (F13_Output[IB8] + F5_Output[L2]);
			F15_Output[IB8] = F14_Output_s;
		}

	}
};

extern "C" void ve_9BC88669_FI_dfp_ncc_6D(sol_ctx* ctx, sol_tensor* __F13_Output) {
	Outputs* outputs = (Outputs*)sol_ctx_outputs(ctx);
	Params* params = (Params*)sol_ctx_params(ctx);
	double* F15_Output = sol_internal_ptr_double(ctx, &outputs->F15_Output);
	double* F13_Output = sol_internal_ptr_double(ctx, __F13_Output);
	double* F5_Output = sol_internal_ptr_double(ctx, &params->F5_Output);
	veda_omp_simd(6, ve_9BC88669_FI_dfp_ncc_6D_B8(F15_Output, F13_Output, F5_Output));
}
