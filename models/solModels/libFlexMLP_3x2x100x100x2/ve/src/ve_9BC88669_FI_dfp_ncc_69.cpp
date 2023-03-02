
#include <sol/dfp/ncc/api.h>
#include "ve_9BC88669.h"
struct ve_9BC88669_FI_dfp_ncc_69_A2 {
	const double*  F10_Output;
	const double*  F3_Output;
	double*  F12_Output;
	inline ve_9BC88669_FI_dfp_ncc_69_A2(const double*  _F10_Output, const double*  _F3_Output, double*  _F12_Output) : F10_Output(_F10_Output), F3_Output(_F3_Output), F12_Output(_F12_Output){}
	void operator()(const int32_t min, const int32_t max) const {

		#pragma _NEC vector
		for(int32_t IA2 = min; IA2 < max; IA2++) {
			const int32_t L2 = (IA2) % 100;
			double F11_Output_s = (F10_Output[IA2] + F3_Output[L2]);
			double F12_Output_s = sol_dfp_ncc_max(F11_Output_s, 0x0p+0);
			F12_Output[IA2] = F12_Output_s;
		}

	}
};

extern "C" void ve_9BC88669_FI_dfp_ncc_69(sol_ctx* ctx, sol_tensor* __F10_Output, sol_tensor* __F12_Output) {
	Params* params = (Params*)sol_ctx_params(ctx);
	double* F10_Output = sol_internal_ptr_double(ctx, __F10_Output);
	double* F3_Output = sol_internal_ptr_double(ctx, &params->F3_Output);
	double* F12_Output = sol_internal_ptr_double(ctx, __F12_Output);
	veda_omp_simd(300, ve_9BC88669_FI_dfp_ncc_69_A2(F10_Output, F3_Output, F12_Output));
}
