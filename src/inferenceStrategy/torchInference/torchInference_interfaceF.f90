module torch_inference_mod

use, intrinsic :: ISO_C_Binding, only: C_int, C_int64_t, C_float, C_double, C_char, C_size_t, C_null_char
use, intrinsic :: ISO_C_Binding, only: C_ptr, C_NULL_ptr

implicit none

interface

function C_createTorchInferenceDouble () result(torch_obj) bind(C, name="createTorchInferenceDouble")
    import
    type(C_ptr) :: torch_obj
end function

function C_createTorchInferenceFloat () result(torch_obj) bind(C, name="createTorchInferenceFloat")
    import
    type(C_ptr) :: torch_obj
end function

subroutine C_deleteTorchInferenceDouble (torch_obj) bind(C, name="deleteTorchInferenceDouble")
    import
    type(C_ptr), value :: torch_obj
end subroutine

subroutine C_deleteTorchInferenceFloat (torch_obj) bind(C, name="deleteTorchInferenceFloat")
    import
    type(C_ptr), value :: torch_obj
end subroutine

subroutine C_initTorchInferenceDouble (torch_obj, batchsize, device_id, model_file, &
        input_shape, num_input_dims, input_data, &
        output_shape, num_output_dims, output_data) bind(C, name="initTorchInferenceDouble")
    import
    type(C_ptr), value :: torch_obj
    integer(C_int), value :: batchsize
    integer(C_int), value :: device_id
    character(kind=C_char) :: model_file(*)
    integer(C_int64_t) :: input_shape(*)
    integer(C_int), value :: num_input_dims
    REAL(C_double) :: input_data(*)
    integer(C_int64_t) :: output_shape(*)
    integer(C_int), value :: num_output_dims
    REAL(C_double) :: output_data(*)
end subroutine

subroutine C_initTorchInferenceFloat (torch_obj, batchsize, device_id, model_file, &
    input_shape, num_input_dims, input_data, &
    output_shape, num_output_dims, output_data) bind(C, name="initTorchInferenceFloat")
    import
    type(C_ptr), value :: torch_obj
    integer(C_int), value :: batchsize
    integer(C_int), value :: device_id
    character(kind=C_char) :: model_file(*)
    integer(C_int64_t) :: input_shape(*)
    integer(C_int), value :: num_input_dims
    REAL(C_float) :: input_data(*)
    integer(C_int64_t) :: output_shape(*)
    integer(C_int), value :: num_output_dims
    REAL(C_float) :: output_data(*)
end subroutine

subroutine C_forwardTorchInferenceDouble (torch_obj) bind(C, name="forwardTorchInferenceDouble")
    import
    type(C_ptr), value :: torch_obj 
end subroutine

subroutine C_forwardTorchInferenceFloat (torch_obj) bind(C, name="forwardTorchInferenceFloat")
    import
    type(C_ptr), value :: torch_obj 
end subroutine

end interface

end module torch_inference_mod