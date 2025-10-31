program test_aixelerator_service

use aixelerator_service_mod
use iso_c_binding, only: C_ptr
use mpi

    implicit none
    type(C_ptr) :: aixelerator

    integer(kind=4) :: num_input_dims
    integer(kind=8), dimension(2) :: input_shape
    real(kind=8), dimension(2) :: input_data

    integer(kind=4) :: num_output_dims
    integer(kind=8), dimension(2) :: output_shape
    real(kind=8), dimension(2) :: output_data

    integer(kind=4) :: batchsize

    character(len=45) :: model_file
    character(kind=c_char), dimension(45) :: model_file_c
    character(len=50) :: model_file_tf

    character(len=4) :: tensor_val_fmt
    character(len=99) :: tensor_format

    integer :: i
    integer rank, ierr

    

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

    write(*,*) "Test AIxeleratorService from Fortran starting!"

    model_file      = "../models/torchModels/flexMLP-2x100x100x2.pt" // c_null_char
    model_file_tf   = "../models/tensorflowModels/flexMLP-2x100x100x2.tf" // c_null_char

    num_input_dims = 2
    input_shape = (/1, 2/)
    input_data = (/real(rank), real(rank)/)

    num_output_dims = 2
    output_shape = (/1, 2/)
    output_data = (/-13.37, -13.37/)
    
    batchsize = 3

    write(*,*) "Creating AIxeleratorService object from Fortran now!"
    write(*,'(A,I0.0,A,F3.1,A,F3.1,A)') "MPI rank ", rank, ": registering input tensor for AIxeleratorService = (", &
    input_data(1), ", ", input_data(2), ")"
    aixelerator = createAIxeleratorServiceDouble_C(model_file_tf, input_shape, num_input_dims, input_data, &
    output_shape, num_output_dims, output_data, batchsize, MPI_COMM_WORLD)

    write(*,'(A,I0.0,A)') "MPI rank ", rank, ": calling AIxeleratorService inference from Fortran now!"
    call inferenceAIxeleratorServiceDouble_C(aixelerator)

    tensor_val_fmt = 'F6.3'
    tensor_format = '(A,I0.0,A,' // tensor_val_fmt // ',A2,' // tensor_val_fmt // &
    ',A7,' // tensor_val_fmt // ',A2,' // tensor_val_fmt // ',A1)'

    write(*,tensor_format) "MPI rank ", rank, ": received output from AIxeleratorService from Fortran = (", &
    output_data(1), ", ", output_data(2), ")"

    write(*,'(A,I0.0,A)') "MPI rank ", rank, ": deleting AIxeleratorService object from Fortran now!"
    call deleteAIxeleratorServiceDouble_C(aixelerator)

    call MPI_Finalize(ierr)

    write(*,*) "Test AIxeleratorService from Fortran completed!"

end program test_aixelerator_service