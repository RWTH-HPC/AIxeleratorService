#pragma once
typedef struct Inputs_struct {
	sol_tensor F6_Output;
} Inputs;

typedef struct Outputs_struct {
	sol_tensor F15_Output;
} Outputs;

typedef struct Params_struct {
	sol_tensor F0_Output;
	sol_tensor F1_Output;
	sol_tensor F2_Output;
	sol_tensor F3_Output;
	sol_tensor F4_Output;
	sol_tensor F5_Output;
} Params;

typedef struct Copies_struct {
	sol_tensor F8_Output;
	sol_tensor F9_Output;
	sol_tensor F11_Output;
	sol_tensor F12_Output;
} Copies;

typedef struct GradInputs_struct {
} GradInputs;

typedef struct GradOutputs_struct {
	sol_tensor B15_Output;
} GradOutputs;

typedef struct GradParams_struct {
	sol_tensor B0_Output;
	sol_tensor B1_Output;
	sol_tensor B2_Output;
	sol_tensor B3_Output;
	sol_tensor B4_Output;
	sol_tensor B5_Output;
} GradParams;

typedef struct Buffers_struct {
} Buffers;

