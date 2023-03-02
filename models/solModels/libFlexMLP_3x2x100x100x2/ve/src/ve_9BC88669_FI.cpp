#include <sol/shared/types.h>
#include "ve_9BC88669.h"
#include <sol/ve/api.h>

extern "C" void ve_9BC88669_FI_dnn_veblas_5E(sol_ctx* ctx, sol_tensor* F7_Output);
extern "C" void ve_9BC88669_FI_dfp_ncc_62(sol_ctx* ctx, sol_tensor* F7_Output, sol_tensor* F9_Output);
extern "C" void ve_9BC88669_FI_dnn_veblas_65(sol_ctx* ctx, sol_tensor* F9_Output, sol_tensor* F10_Output);
extern "C" void ve_9BC88669_FI_dfp_ncc_69(sol_ctx* ctx, sol_tensor* F10_Output, sol_tensor* F12_Output);
extern "C" void ve_9BC88669_FI_dnn_veblas_6C(sol_ctx* ctx, sol_tensor* F12_Output, sol_tensor* F13_Output);
extern "C" void ve_9BC88669_FI_dfp_ncc_6D(sol_ctx* ctx, sol_tensor* F13_Output);

extern "C" void ve_9BC88669_FI(sol_ctx* ctx) {
	sol_tensor __F7_Output = SOL_TENSOR(0, SOL_DTYPE_F64, {300ll});
	sol_internal_malloc_double(ctx, &__F7_Output, 0);
	ve_9BC88669_FI_dnn_veblas_5E(ctx, &__F7_Output);
	sol_tensor __F9_Output = SOL_TENSOR(0, SOL_DTYPE_F64, {300ll});
	sol_internal_malloc_double(ctx, &__F9_Output, 0);
	ve_9BC88669_FI_dfp_ncc_62(ctx, &__F7_Output, &__F9_Output);
	sol_internal_free(ctx, &__F7_Output);
	sol_tensor __F10_Output = SOL_TENSOR(0, SOL_DTYPE_F64, {300ll});
	sol_internal_malloc_double(ctx, &__F10_Output, 0);
	ve_9BC88669_FI_dnn_veblas_65(ctx, &__F9_Output, &__F10_Output);
	sol_internal_free(ctx, &__F9_Output);
	sol_tensor __F12_Output = SOL_TENSOR(0, SOL_DTYPE_F64, {300ll});
	sol_internal_malloc_double(ctx, &__F12_Output, 0);
	ve_9BC88669_FI_dfp_ncc_69(ctx, &__F10_Output, &__F12_Output);
	sol_internal_free(ctx, &__F10_Output);
	sol_tensor __F13_Output = SOL_TENSOR(0, SOL_DTYPE_F64, {6ll});
	sol_internal_malloc_double(ctx, &__F13_Output, 0);
	ve_9BC88669_FI_dnn_veblas_6C(ctx, &__F12_Output, &__F13_Output);
	sol_internal_free(ctx, &__F12_Output);
	Outputs* outputs = (Outputs*)sol_ctx_outputs(ctx);
//	sol_external_malloc_double(ctx, &outputs->F15_Output, 0);
	ve_9BC88669_FI_dfp_ncc_6D(ctx, &__F13_Output);
	sol_internal_free(ctx, &__F13_Output);
}

extern "C" void ve_9BC88669_FI_offload(const sol_dim batch_size, const int device_idx, sol_tensor* inputs, sol_tensor* params, sol_tensor* buffers, sol_tensor* outputs) {
	sol_ctx ctx      = {0};
	ctx.batch_size   = batch_size;
	ctx.device_idx   = device_idx;
	ctx.inputs       = inputs;
	ctx.params       = params;
	ctx.buffers      = buffers;
	ctx.outputs      = outputs;

	ve_9BC88669_FI(&ctx);
}
