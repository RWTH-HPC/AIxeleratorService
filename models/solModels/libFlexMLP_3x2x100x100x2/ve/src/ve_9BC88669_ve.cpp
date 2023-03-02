#include <sol/ve/api.h>

extern "C" void ve_9BC88669_dnn_veblas_create(sol_ctx* ctx);
extern "C" void ve_9BC88669_dnn_veblas_update(sol_ctx* ctx);
extern "C" void ve_9BC88669_dnn_veblas_destroy(void);
extern "C" int  ve_9BC88669_dnn_veblas_seed(int seed);
extern "C" void ve_9BC88669_create(sol_ctx* ctx) {
	ve_9BC88669_dnn_veblas_create(ctx);
}

extern "C" void ve_9BC88669_create_offload(const sol_dim batch_size, const int device_idx) {
	sol_ctx ctx = {0};
	ctx.batch_size = batch_size;
	ctx.device_idx = device_idx;
	ve_9BC88669_create(&ctx);
}

extern "C" void ve_9BC88669_update(sol_ctx* ctx) {
}

extern "C" void ve_9BC88669_update_offload(const sol_dim batch_size, const int device_idx) {
	static sol_ctx s_ctx = {0};
	if(s_ctx.batch_size != batch_size) {
		s_ctx.batch_size = batch_size;
		s_ctx.device_idx = device_idx;
		ve_9BC88669_update(&s_ctx);
	}
}

extern "C" void ve_9BC88669_destroy(void) {
	ve_9BC88669_dnn_veblas_destroy();
}
