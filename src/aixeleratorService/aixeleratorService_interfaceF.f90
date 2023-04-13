module aixelerator_service_mod

use, intrinsic :: ISO_C_Binding, only: C_int, C_int64_t, C_double, C_float, C_char, C_size_t, C_null_char
use, intrinsic :: ISO_C_Binding, only: C_ptr, C_NULL_ptr

implicit none

interface

    function createAIxeleratorServiceDouble_C(model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data, batchsize) result(aixelerator) bind(C, name="createAIxeleratorServiceDouble")
        import 
        character(kind=C_char) :: model_file(*)
        integer(C_int64_t) :: input_shape(*)
        integer(C_int), value :: num_input_dims
        REAL(C_double) :: input_data(*)
        integer(C_int64_t) :: output_shape(*)
        integer(C_int), value :: num_output_dims
        REAL(C_double) :: output_data(*)
        integer(C_int), value :: batchsize

        type(C_ptr) :: aixelerator
    end function

    function createAIxeleratorServiceFloat_C(model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data, batchsize) result(aixelerator) bind(C, name="createAIxeleratorServiceFloat")
        import 
        character(kind=C_char) :: model_file(*)
        integer(C_int64_t) :: input_shape(*)
        integer(C_int), value :: num_input_dims
        REAL(C_float) :: input_data(*)
        integer(C_int64_t) :: output_shape(*)
        integer(C_int), value :: num_output_dims
        REAL(C_float) :: output_data(*)
        integer(C_int), value :: batchsize

        type(C_ptr) :: aixelerator
    end function

    subroutine deleteAIxeleratorServiceDouble_C(aixelerator) bind(C, name="deleteAIxeleratorServiceDouble")
        import
        type(C_ptr), value :: aixelerator
    end subroutine 

    subroutine deleteAIxeleratorServiceFloat_C(aixelerator) bind(C, name="deleteAIxeleratorServiceFloat")
        import
        type(C_ptr), value :: aixelerator
    end subroutine

    subroutine inferenceAIxeleratorServiceDouble_C(aixelerator) bind(C, name="inferenceAIxeleratorServiceDouble")
        import
        type(C_ptr), value :: aixelerator
    end subroutine

    subroutine inferenceAIxeleratorServiceFloat_C(aixelerator) bind(C, name="inferenceAIxeleratorServiceFloat")
        import
        type(C_ptr), value :: aixelerator
    end subroutine


end interface

end module aixelerator_service_mod