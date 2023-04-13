program test_torch_inference

use torch_inference_mod
use iso_c_binding, only: C_ptr

    implicit none
    type(C_ptr) :: torch_obj

    integer(kind=4) :: num_input_dims
    integer(kind=8), dimension(2) :: input_shape
    real(kind=8), dimension(8) :: input_data

    integer(kind=4) :: num_output_dims
    integer(kind=8), dimension(2) :: output_shape
    real(kind=8), dimension(8) :: output_data

    integer(kind=4) :: batchsize
    integer(kind=4) :: device_id

    character(len=44) :: model_file
    character(kind=c_char), dimension(45) :: model_file_c

    character(len=4) :: tensor_val_fmt
    character(len=99) :: tensor_format

    integer :: i

    

    model_file = "../models/torchModels/flexMLP-2x100x100x2.pt"

    num_input_dims = 2
    input_shape = (/4, 2/)
    input_data = (/0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0/)

    num_output_dims = 2
    output_shape = (/4, 2/)
    output_data = (/-13.37, -13.37, -13.37, -13.37, -13.37, -13.37, -13.37, -13.37/)
    
    batchsize = 3
    device_id = 0

    write(*,*) "Creating Torch Inference object in Fortran now!"
    torch_obj = C_createTorchInferenceDouble()

    write(*,*) "Init Torch Inference object in Fortran now!"
    call C_initTorchInferenceDouble(torch_obj, batchsize, device_id, model_file, input_shape, num_input_dims, input_data, output_shape, num_output_dims, output_data)

    write(*,*) "Torch Inference Test inference in Fortran now!"
    call C_forwardTorchInferenceDouble(torch_obj)

    tensor_val_fmt = 'F6.3'
    tensor_format = '(A1,' // tensor_val_fmt // ',A2,' // tensor_val_fmt // ',A7,' // tensor_val_fmt // ',A2,' // tensor_val_fmt // ',A1)'

    write(*,tensor_format) "(", input_data(1), ", ", input_data(2), ") --> (", output_data(1), ", ", output_data(2), ")"
    write(*,tensor_format) "(", input_data(3), ", ", input_data(4), ") --> (", output_data(3), ", ", output_data(4), ")"
    write(*,tensor_format) "(", input_data(5), ", ", input_data(6), ") --> (", output_data(5), ", ", output_data(6), ")"
    write(*,tensor_format) "(", input_data(7), ", ", input_data(8), ") --> (", output_data(7), ", ", output_data(8), ")"

    write(*,*) "Deleting Torch Inference object in Fortran now!"
    call C_deleteTorchInferenceDouble(torch_obj)

    write(*,*) "Torch Inference Test in Fortran completed!"

end program test_torch_inference