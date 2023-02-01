```
/*---------------------------------------------------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v2006                                 |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
Build  : _b45f8f6f58-20200629 OPENFOAM=2006
Arch   : "LSB;label=32;scalar=64"
Exec   : /rwthfs/rz/cluster/home/fo014819/OpenFOAM/fo014819-v2006/platforms/linux64IccDPInt32Opt/bin/rhoPimpleFoamTorch -parallel
Date   : Feb 01 2023
Time   : 17:07:14
Host   : login18-g-2.hpc.itc.rwth-aachen.de
PID    : 87220
I/O    : uncollated
Case   : /work/fo014819/nhr4ces/data/verificationCase/00002-transferCase
nProcs : 4
Hosts  :
(
    (login18-g-2.hpc.itc.rwth-aachen.de 4)
)
Pstream initialized with:
    floatTransfer      : 0
    nProcsSimpleSum    : 0
    commsType          : nonBlocking
    polling iterations : 0
trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).
fileModificationChecking : Monitoring run-time modified files using timeStampMaster (fileModificationSkew 5, maxFileModificationPolls 20)
allowSystemOperations : Allowing user-supplied system call operations

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
Create time

Create mesh for time = 0


PIMPLE: no residual control data found. Calculations will employ 2 corrector loops

Reading thermophysical properties

Selecting thermodynamics package 
{
    type            hePsiThermo;
    mixture         reactingMixture;
    transport       sutherland;
    thermo          janaf;
    energy          sensibleEnthalpy;
    equationOfState perfectGas;
    specie          specie;
}

Selecting chemistryReader chemkinReader
Reading field U

Reading/calculating face flux field phi

Creating turbulence model

Selecting turbulence model type laminar
Selecting laminar stress model Stokes
Creating field dpdt

Creating field kinetic energy K

No MRF models present

No finite volume options present
Courant Number mean: 0.0005153904244 max: 0.0009059595567

Initialize torch model

Rank 1/4 on its local machine is 1/4
Rank 2/4 on its local machine is 2/4
Rank 0/4 on its local machine is 0/4
Rank 3/4 on its local machine is 3/4
Rank 0/4 on its local machine is 0/4 and has access to 1 devices!
Rank 0/4 on its local machine is 0/4 is the GPU Master!
Rank 1/4 on its local machine is 1/4 and has access to 1 devices!
Rank 2/4 on its local machine is 2/4 and has access to 1 devices!
Rank 0/4 knows that there is a total of 1 gpus across all systems. 
Rank 0/4 will be in group 0 order 0
Rank 0/4 got id of 0/4 within the workGroup
Rank 2/4 knows that there is a total of 1 gpus across all systems. 
Rank 2/4 will be in group 0 order 3
Rank 2/4 got id of 2/4 within the workGroup
Rank 1/4 knows that there is a total of 1 gpus across all systems. 
Rank 1/4 will be in group 0 order 2
Rank 1/4 got id of 1/4 within the workGroup
Rank 3/4 on its local machine is 3/4 and has access to 1 devices!
Rank 3/4 knows that there is a total of 1 gpus across all systems. 
Rank 3/4 will be in group 0 order 4
Rank 3/4 got id of 3/4 within the workGroup
Input 0: scalar_1
Input 1: scalar_2

Starting time loop

Courant Number mean: 0.0005153904244 max: 0.0009059595567
deltaT = 5.99880024e-07
Time = 5.9988e-07

PIMPLE: iteration 1
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
smoothSolver:  Solving for Ux, Initial residual = 1.054934288e-05, Final residual = 1.768638375e-10, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 0.0001905966278, Final residual = 7.341437153e-09, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 0.0001377661984, Final residual = 4.906494865e-09, No Iterations 1
smoothSolver:  Solving for h, Initial residual = 1.192937797e-05, Final residual = 2.882413226e-09, No Iterations 1
DICPCG:  Solving for p, Initial residual = 1, Final residual = 1.901439417e-09, No Iterations 3
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 6.038033299e-14, global = 1.100279853e-14, cumulative = 1.100279853e-14
DICPCG:  Solving for p, Initial residual = 2.429137556e-05, Final residual = 1.470154871e-10, No Iterations 2
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 7.293180692e-15, global = -7.005055035e-15, cumulative = 3.997743491e-15
PIMPLE: iteration 2
MPI rank: 1 Inputfield: 0 = 0
MPI rank: 1 Inputfield: 1 = 0
MPI rank: 1 input tensor = 
smoothSolver:  Solving for Ux, Initial residual = 4.546206937e-06, Final residual = 1.205667907e-10, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 9.741156263e-06, Final residual = 1.883112475e-10, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 7.544133719e-06, Final residual = 1.344111506e-10, No Iterations 1
smoothSolver:  Solving for h, Initial residual = 4.547951666e-06, Final residual = 9.045743504e-10, No Iterations 1
DICPCG:  Solving for p, Initial residual = 0.1558505883, Final residual = 6.405675959e-10, No Iterations 3
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 3.756438285e-14, global = -1.235338765e-14, cumulative = -8.355644157e-15
DICPCG:  Solving for p, Initial residual = 7.51343516e-06, Final residual = 4.599636453e-11, No Iterations 2
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 2.881256594e-15, global = -2.881256594e-15, cumulative = -1.123690075e-14
MPI rank: 0 Inputfield: 0 = 0
MPI rank: 0 Inputfield: 1 = 0
MPI rank: 0 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 3 Inputfield: 0 = 0
MPI rank: 3 Inputfield: 1 = 0
MPI rank: 3 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 2 Inputfield: 0 = 0
MPI rank: 2 Inputfield: 1 = 0
MPI rank: 2 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
batchedforward complete
scatter starting
MPI rank: 0 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 0 field: 0 = 0.07728102139
MPI rank: 0 field: 1 = -0.08813373617
scatter complete
MPI rank: 2 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 2 field: 0 = 0.07728102139
MPI rank: 2 field: 1 = -0.08813373617
MPI rank: 3 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 3 field: 0 = 0.07728102139
MPI rank: 3 field: 1 = -0.08813373617
MPI rank: 1 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 1 field: 0 = 0.07728102139
MPI rank: 1 field: 1 = -0.08813373617
Max. FOAM to Torch time: 0.000427206s
Foward time       : 0.055642782s
Max. toCUDA time   : 1.8e-08s
Max. IValue time   : 2.3e-08s
Max. Actual forward time   : 0.05564271s
Output to CPU time   : 3.1e-08s
Max. Torch to FOAM     : 9.0248e-05s
Max. Total ML time     : 0.056166125s
ExecutionTime = 5.63 s  ClockTime = 4 s

scalarTransport execute: scalar_1
smoothSolver:  Solving for scalar_1, Initial residual = 0, Final residual = 0, No Iterations 0

scalarTransport execute: scalar_2
smoothSolver:  Solving for scalar_2, Initial residual = 0, Final residual = 0, No Iterations 0

Courant Number mean: 0.0006237406293 max: 0.001077885799
deltaT = 7.200289049e-07
Time = 1.319909e-06

PIMPLE: iteration 1
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
smoothSolver:  Solving for Ux, Initial residual = 1.866219242e-05, Final residual = 3.13868674e-10, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 0.0005355083615, Final residual = 1.242102179e-08, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 2.11667079e-05, Final residual = 4.007758792e-10, No Iterations 1
smoothSolver:  Solving for h, Initial residual = 1.929856786e-05, Final residual = 3.469695678e-09, No Iterations 1
DICPCG:  Solving for p, Initial residual = 0.4623717101, Final residual = 5.858866089e-11, No Iterations 3
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 6.896984076e-15, global = -6.896984076e-15, cumulative = -1.813388483e-14
DICPCG:  Solving for p, Initial residual = 1.234866808e-05, Final residual = 2.7490094e-11, No Iterations 2
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 4.141792004e-15, global = -4.141792004e-15, cumulative = -2.227567683e-14
PIMPLE: iteration 2
MPI rank: 1 Inputfield: 0 = 0
MPI rank: 1 Inputfield: 1 = 0
MPI rank: 1 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 2 Inputfield: 0 = 0
MPI rank: 2 Inputfield: 1 = 0
MPI rank: 2 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 3 Inputfield: 0 = 0
MPI rank: 3 Inputfield: 1 = 0
MPI rank: 3 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 1 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 1 field: 0 = 0.07728102139
MPI rank: 1 field: 1 = -0.08813373617
MPI rank: 2 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 2 field: 0 = 0.07728102139
MPI rank: 2 field: 1 = -0.08813373617
MPI rank: 3 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 3 field: 0 = 0.07728102139
MPI rank: 3 field: 1 = -0.08813373617
smoothSolver:  Solving for Ux, Initial residual = 3.158572344e-06, Final residual = 8.664208922e-11, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 4.850270577e-06, Final residual = 1.406974713e-10, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 4.168196987e-06, Final residual = 1.180903262e-10, No Iterations 1
smoothSolver:  Solving for h, Initial residual = 3.020305645e-06, Final residual = 4.66250892e-10, No Iterations 1
DICPCG:  Solving for p, Initial residual = 0.03941726921, Final residual = 3.558216528e-11, No Iterations 3
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 5.582415328e-15, global = -1.908825886e-15, cumulative = -2.418450272e-14
DICPCG:  Solving for p, Initial residual = 1.815293084e-06, Final residual = 3.624202824e-09, No Iterations 1
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 5.803010772e-13, global = -3.974859791e-13, cumulative = -4.216704818e-13
MPI rank: 0 Inputfield: 0 = 0
MPI rank: 0 Inputfield: 1 = 0
MPI rank: 0 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
batchedforward complete
scatter starting
MPI rank: 0 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 0 field: 0 = 0.07728102139
MPI rank: 0 field: 1 = -0.08813373617
scatter complete
Max. FOAM to Torch time: 5.4985e-05s
Foward time       : 0.00037859s
Max. toCUDA time   : 1.8e-08s
Max. IValue time   : 1.8e-08s
Max. Actual forward time   : 0.000378518s
Output to CPU time   : 3.6e-08s
Max. Torch to FOAM     : 8.7489e-05s
Max. Total ML time     : 0.000522001s
ExecutionTime = 5.64 s  ClockTime = 4 s

scalarTransport execute: scalar_1
smoothSolver:  Solving for scalar_1, Initial residual = 0, Final residual = 0, No Iterations 0

scalarTransport execute: scalar_2
smoothSolver:  Solving for scalar_2, Initial residual = 0, Final residual = 0, No Iterations 0

Courant Number mean: 0.0007538933096 max: 0.00131038878
deltaT = 8.639101134e-07
Time = 2.183819e-06

PIMPLE: iteration 1
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
smoothSolver:  Solving for Ux, Initial residual = 2.686121604e-05, Final residual = 7.535174912e-10, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 0.001326617716, Final residual = 4.867268441e-08, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 3.182648078e-05, Final residual = 9.874664912e-10, No Iterations 1
smoothSolver:  Solving for h, Initial residual = 2.736656251e-05, Final residual = 5.812374486e-09, No Iterations 1
DICPCG:  Solving for p, Initial residual = 0.3210924501, Final residual = 1.341043054e-10, No Iterations 3
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 3.11353453e-14, global = -3.11353453e-14, cumulative = -4.528058271e-13
DICPCG:  Solving for p, Initial residual = 1.104622495e-05, Final residual = 5.087474219e-11, No Iterations 2
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 1.462226743e-14, global = -1.462226743e-14, cumulative = -4.674280946e-13
PIMPLE: iteration 2
smoothSolver:  Solving for Ux, Initial residual = 6.933669383e-07, Final residual = 1.871524478e-11, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 6.20453184e-07, Final residual = 1.9947377e-11, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 7.552366788e-07, Final residual = 2.378272089e-11, No Iterations 1
MPI rank: 1 Inputfield: 0 = 0
MPI rank: 1 Inputfield: 1 = 0
MPI rank: 1 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 2 Inputfield: 0 = 0
MPI rank: 2 Inputfield: 1 = 0
MPI rank: 2 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 3 Inputfield: 0 = 0
MPI rank: 3 Inputfield: 1 = 0
MPI rank: 3 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 1 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 1 field: 0 = 0.07728102139
MPI rank: 1 field: 1 = -0.08813373617
MPI rank: 2 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 2 field: 0 = 0.07728102139
MPI rank: 2 field: 1 = -0.08813373617
MPI rank: 3 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 3 field: 0 = 0.07728102139
MPI rank: 3 field: 1 = -0.08813373617
smoothSolver:  Solving for h, Initial residual = 6.798924463e-07, Final residual = 1.241941486e-10, No Iterations 1
DICPCG:  Solving for p, Initial residual = 0.004698292588, Final residual = 5.275365922e-09, No Iterations 2
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 1.530584043e-12, global = 1.530584043e-12, cumulative = 1.063155948e-12
DICPCG:  Solving for p, Initial residual = 2.197702064e-07, Final residual = 4.974864062e-10, No Iterations 1
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 1.449801417e-13, global = -8.476233103e-14, cumulative = 9.78393617e-13
MPI rank: 0 Inputfield: 0 = 0
MPI rank: 0 Inputfield: 1 = 0
MPI rank: 0 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
batchedforward complete
scatter starting
MPI rank: 0 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 0 field: 0 = 0.07728102139
MPI rank: 0 field: 1 = -0.08813373617
scatter complete
Max. FOAM to Torch time: 5.136e-05s
Foward time       : 0.000431466s
Max. toCUDA time   : 1.9e-08s
Max. IValue time   : 1.8e-08s
Max. Actual forward time   : 0.000431399s
Output to CPU time   : 3e-08s
Max. Torch to FOAM     : 9.0209e-05s
Max. Total ML time     : 0.000573969s
ExecutionTime = 5.65 s  ClockTime = 4 s

scalarTransport execute: scalar_1
smoothSolver:  Solving for scalar_1, Initial residual = 0, Final residual = 0, No Iterations 0

scalarTransport execute: scalar_2
smoothSolver:  Solving for scalar_2, Initial residual = 0, Final residual = 0, No Iterations 0

Courant Number mean: 0.0009049982797 max: 0.001574874097
deltaT = 1.036153874e-06
Time = 3.219973e-06

PIMPLE: iteration 1
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
smoothSolver:  Solving for Ux, Initial residual = 3.306933765e-05, Final residual = 1.168806573e-09, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 0.002663651043, Final residual = 1.217819855e-07, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 3.97234774e-05, Final residual = 1.540392746e-09, No Iterations 1
smoothSolver:  Solving for h, Initial residual = 3.359029802e-05, Final residual = 8.539483831e-09, No Iterations 1
DICPCG:  Solving for p, Initial residual = 0.2403256877, Final residual = 1.520669087e-10, No Iterations 3
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 5.715623495e-14, global = 5.715623495e-14, cumulative = 1.035549852e-12
DICPCG:  Solving for p, Initial residual = 1.015247917e-05, Final residual = 9.617338807e-11, No Iterations 2
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 4.314629456e-14, global = -4.314629456e-14, cumulative = 9.924035574e-13
PIMPLE: iteration 2
smoothSolver:  Solving for Ux, Initial residual = 9.025068638e-07, Final residual = 3.058314527e-11, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 1.567750581e-06, Final residual = 6.890950002e-11, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 1.044253953e-06, Final residual = 4.1056495e-11, No Iterations 1
smoothSolver:  Solving for h, Initial residual = 8.68517575e-07, Final residual = 2.080394439e-10, No Iterations 1
MPI rank: 2 Inputfield: 0 = 0
MPI rank: 2 Inputfield: 1 = 0
MPI rank: 2 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 1 Inputfield: 0 = 0
MPI rank: 1 Inputfield: 1 = 0
MPI rank: 1 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 3 Inputfield: 0 = 0
MPI rank: 3 Inputfield: 1 = 0
MPI rank: 3 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 1 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 1 field: 0 = 0.07728102139
MPI rank: 1 field: 1 = -0.08813373617
MPI rank: 2 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 2 field: 0 = 0.07728102139
MPI rank: 2 field: 1 = -0.08813373617
MPI rank: 3 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 3 field: 0 = 0.07728102139
MPI rank: 3 field: 1 = -0.08813373617
DICPCG:  Solving for p, Initial residual = 0.003859854746, Final residual = 1.972167908e-11, No Iterations 3
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 8.949794803e-15, global = 1.49463374e-15, cumulative = 9.938981912e-13
DICPCG:  Solving for p, Initial residual = 2.342895744e-07, Final residual = 8.765158525e-10, No Iterations 1
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 3.941331164e-13, global = 3.105776881e-13, cumulative = 1.304475879e-12
MPI rank: 0 Inputfield: 0 = 0
MPI rank: 0 Inputfield: 1 = 0
MPI rank: 0 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
batchedforward complete
scatter starting
MPI rank: 0 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 0 field: 0 = 0.07728102139
MPI rank: 0 field: 1 = -0.08813373617
scatter complete
Max. FOAM to Torch time: 4.9355e-05s
Foward time       : 0.000335392s
Max. toCUDA time   : 1.7e-08s
Max. IValue time   : 2.3e-08s
Max. Actual forward time   : 0.00033533s
Output to CPU time   : 2.2e-08s
Max. Torch to FOAM     : 6.6638e-05s
Max. Total ML time     : 0.000452255s
ExecutionTime = 5.66 s  ClockTime = 4 s

scalarTransport execute: scalar_1
smoothSolver:  Solving for scalar_1, Initial residual = 0, Final residual = 0, No Iterations 0

scalarTransport execute: scalar_2
smoothSolver:  Solving for scalar_2, Initial residual = 0, Final residual = 0, No Iterations 0

Courant Number mean: 0.001083812418 max: 0.001884963595
deltaT = 1.242867864e-06
Time = 4.462841e-06

PIMPLE: iteration 1
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
smoothSolver:  Solving for Ux, Initial residual = 3.844783395e-05, Final residual = 1.574503133e-09, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 0.004787944324, Final residual = 2.550811244e-07, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 4.634242406e-05, Final residual = 2.076621568e-09, No Iterations 1
smoothSolver:  Solving for h, Initial residual = 3.909996184e-05, Final residual = 1.19551415e-08, No Iterations 1
DICPCG:  Solving for p, Initial residual = 0.1919055061, Final residual = 1.017422478e-09, No Iterations 3
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 5.504906382e-13, global = 4.535380181e-13, cumulative = 1.758013897e-12
DICPCG:  Solving for p, Initial residual = 9.830205516e-06, Final residual = 1.898022592e-10, No Iterations 2
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 1.205244959e-13, global = -1.205244959e-13, cumulative = 1.637489401e-12
PIMPLE: iteration 2
smoothSolver:  Solving for Ux, Initial residual = 7.723783495e-07, Final residual = 3.243700523e-11, No Iterations 1
smoothSolver:  Solving for Uy, Initial residual = 1.960657828e-06, Final residual = 1.023135074e-10, No Iterations 1
smoothSolver:  Solving for Uz, Initial residual = 9.309178086e-07, Final residual = 4.502231825e-11, No Iterations 1
smoothSolver:  Solving for h, Initial residual = 7.276745031e-07, Final residual = 2.092110148e-10, No Iterations 1
DICPCG:  Solving for p, Initial residual = 0.002300806971, Final residual = 4.238743001e-11, No Iterations 3
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 2.686725646e-14, global = 5.582338809e-15, cumulative = 1.64307174e-12
DICPCG:  Solving for p, Initial residual = 1.72695973e-07, Final residual = 9.509275815e-10, No Iterations 1
diagonal:  Solving for rho, Initial residual = 0, Final residual = 0, No Iterations 0
time step continuity errors : sum local = 6.010918369e-13, global = 4.597686273e-13, cumulative = 2.102840367e-12
MPI rank: 0 Inputfield: 0 = 0
MPI rank: 0 Inputfield: 1 = 0
MPI rank: 0 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]MPI rank: 2 Inputfield: 0 = 0
MPI rank: 2 Inputfield: 1 = 0
MPI rank: 2 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 3 Inputfield: 0 = 0
MPI rank: 3 Inputfield: 1 = 0
MPI rank: 3 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 1 Inputfield: 0 = 0
MPI rank: 1 Inputfield: 1 = 0
MPI rank: 1 input tensor = 
 0  0
[ CPUDoubleType{1,2} ]
MPI rank: 1 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 1 field: 0 = 0.07728102139
MPI rank: 1 field: 1 = -0.08813373617
MPI rank: 2 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 2 field: 0 = 0.07728102139
MPI rank: 2 field: 1 = -0.08813373617
MPI rank: 3 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 3 field: 0 = 0.07728102139
MPI rank: 3 field: 1 = -0.08813373617

batchedforward complete
scatter starting
MPI rank: 0 output tensor = 
0.01 *
 7.7281 -8.8134
[ CPUDoubleType{1,2} ]
MPI rank: 0 field: 0 = 0.07728102139
MPI rank: 0 field: 1 = -0.08813373617
scatter complete
Max. FOAM to Torch time: 9.7906e-05s
Foward time       : 0.000362813s
Max. toCUDA time   : 1.9e-08s
Max. IValue time   : 1.9e-08s
Max. Actual forward time   : 0.000362741s
Output to CPU time   : 3.4e-08s
Max. Torch to FOAM     : 7.6905e-05s
Max. Total ML time     : 0.000539827s
ExecutionTime = 5.67 s  ClockTime = 4 s

scalarTransport execute: scalar_1
smoothSolver:  Solving for scalar_1, Initial residual = 0, Final residual = 0, No Iterations 0

scalarTransport execute: scalar_2
smoothSolver:  Solving for scalar_2, Initial residual = 0, Final residual = 0, No Iterations 0

End

Exception was thrown: basic_ios::clear: iostream error

```