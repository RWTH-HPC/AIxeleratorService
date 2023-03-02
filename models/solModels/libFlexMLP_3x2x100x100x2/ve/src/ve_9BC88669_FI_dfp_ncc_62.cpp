#include <sol/dfp/ncc/api.h>
#include "ve_9BC88669.h"
struct ve_9BC88669_FI_dfp_ncc_62_86 {
	const double*  F7_Output;
	const double*  F1_Output;
	double*  F9_Output;
	inline ve_9BC88669_FI_dfp_ncc_62_86(const double*  _F7_Output, const double*  _F1_Output, double*  _F9_Output) : F7_Output(_F7_Output), F1_Output(_F1_Output), F9_Output(_F9_Output){}
	void operator()(const int32_t min, const int32_t max) const {

		#pragma _NEC vector
		for(int32_t I86 = min; I86 < max; I86++) {
			const int32_t L2 = (I86) % 100;
			double F8_Output_s = (F7_Output[I86] + F1_Output[L2]);
			double F9_Output_s = sol_dfp_ncc_max(F8_Output_s, 0x0p+0);
			F9_Output[I86] = F9_Output_s;
		}

	}
};

extern "C" void ve_9BC88669_FI_dfp_ncc_62(sol_ctx* ctx, sol_tensor* __F7_Output, sol_tensor* __F9_Output) {
	Params* params = (Params*)sol_ctx_params(ctx);
	double* F7_Output = sol_internal_ptr_double(ctx, __F7_Output);
	double* F1_Output = sol_internal_ptr_double(ctx, &params->F1_Output);
	double* F9_Output = sol_internal_ptr_double(ctx, __F9_Output);
	veda_omp_simd(300, ve_9BC88669_FI_dfp_ncc_62_86(F7_Output, F1_Output, F9_Output));
}
