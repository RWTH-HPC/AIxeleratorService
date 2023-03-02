#include <sol/dnn/veblas/api.h>
#include "ve_9BC88669.h"
extern "C" void ve_9BC88669_dnn_veblas_create(sol_ctx* ctx) {
}

extern "C" void ve_9BC88669_dnn_veblas_destroy() {
}

extern "C" void ve_9BC88669_FI_dnn_veblas_5E(sol_ctx* ctx, sol_tensor* __F7_Output) {
	Inputs* inputs = (Inputs*)sol_ctx_inputs(ctx);
	Params* params = (Params*)sol_ctx_params(ctx);
	const double* F6_Output = sol_internal_ptr_double(ctx, &inputs->F6_Output);
	const double* F0_Output = sol_internal_ptr_double(ctx, &params->F0_Output);
	double* F7_Output = sol_internal_ptr_double(ctx, __F7_Output);
	sol_dnn_veblas_gemm_c<SOL_LAYOUT_NOI, double>(ctx, 2ll, 100ll, 3ll, 1ll, 1.0, 0.0, F6_Output, F0_Output, F7_Output);
}

extern "C" void ve_9BC88669_FI_dnn_veblas_65(sol_ctx* ctx, sol_tensor* __F9_Output, sol_tensor* __F10_Output) {
	Params* params = (Params*)sol_ctx_params(ctx);
	const double* F9_Output = sol_internal_ptr_double(ctx, __F9_Output);
	const double* F2_Output = sol_internal_ptr_double(ctx, &params->F2_Output);
	double* F10_Output = sol_internal_ptr_double(ctx, __F10_Output);
	sol_dnn_veblas_gemm_c<SOL_LAYOUT_NOI, double>(ctx, 100ll, 100ll, 3ll, 1ll, 1.0, 0.0, F9_Output, F2_Output, F10_Output);
}

extern "C" void ve_9BC88669_FI_dnn_veblas_6C(sol_ctx* ctx, sol_tensor* __F12_Output, sol_tensor* __F13_Output) {
	Params* params = (Params*)sol_ctx_params(ctx);
	const double* F12_Output = sol_internal_ptr_double(ctx, __F12_Output);
	const double* F4_Output = sol_internal_ptr_double(ctx, &params->F4_Output);
	double* F13_Output = sol_internal_ptr_double(ctx, __F13_Output);
	sol_dnn_veblas_gemm_c<SOL_LAYOUT_NOI, double>(ctx, 100ll, 2ll, 3ll, 1ll, 1.0, 0.0, F12_Output, F4_Output, F13_Output);
}
