module torch_inference_mod

use, intrinsic :: ISO_C_Binding, only: C_int, C_int64_t, C_double, C_char, C_size_t, C_null_char
use, intrinsic :: ISO_C_Binding, only: C_ptr, C_NULL_ptr

implicit none

interface

function C_createTorchInference () result(torch_obj) bind(C, name="createTorchInference")
    import
    type(C_ptr) :: torch_obj
end function

subroutine C_deleteTorchInference (torch_obj) bind(C, name="deleteTorchInference")
    import
    type(C_ptr), value :: torch_obj
end subroutine

subroutine C_initTorchInference (torch_obj, batchsize, device_id, model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data) bind(C, name="initTorchInference")
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

subroutine C_forwardTorchInference (torch_obj) bind(C, name="forwardTorchInference")
    import
    type(C_ptr), value :: torch_obj 
end subroutine

end interface

end module torch_inference_mod