# Copyright (c) 2016, Technische Universität Dresden, Germany
# All rights reserved.
#
# Copyright (c) 2023, RWTH Aachen University, Germany
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the
#    distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
#    or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#[================================================================================[.rst
FindScorep
----------

Find the Score-P performance measurement infrastructure.

Components
^^^^^^^^^^

The Score-P Module recognizes the following components.

``IO``
  System-level IO interception.

``KOKKOS``
  Kokkos measurement.

``OPENMP``
  General OpenMP measurement support.

``OMPT``
  OpenMP measurement via the OMPT interface.

``OPARI2``
  OpenMP measurement via OPARI2 source-to-source instrumentation.
  (this is not yet supported by this module)

``POSIXIO``
  Posix I/O measurement.

``PTHREAD``
  Pthread measurement.

``SAMPLING``
  Sampling support.

``MPI``
  MPI measurement.

``SHMEM``
  OpenSHMEM measurement.

Imported Targets
^^^^^^^^^^^^^^^^

This module does not provide imported targets.

Usage
^^^^^

Dependencies to Score-P libraries, Compile Options, and Compile definitions
have to be set by calling

```
scorep_instrument_target(targetname
    [COMPILER (ON|OFF)]
    [USER (ON|OFF)]
    [KOKKOS (ON|OFF)]
    [IO (none|posix)]
    [THREADING (none|pthread|omp:ompt)]
    [MPP (none|mpi|shmem)]
    )
```

All arguments but the targetname are optional and can be used to
override an existing global default value.

**NOTE**: Each executable using Score-P instrumented libraries or
source files also needs to be marked for instrumentation via a call
to ``scorep_instrument_target``. 

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables
``SCOREP_FOUND``
  True if the system has found Score-P.
``SCOREP_VERSION``
  The version of the Score-P infrastructure found.

If found, a ``SCOREP_<component>_FOUND`` is set for each component found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``SCOREP_ROOT_DIR``
  The full path to the Score-P installation.
``SCOREP_CONFIG_EXECUTABLE``
  The full path to ``scorep-config`` executable.
``SCOREP_INFO_EXECUTABLE``
  The full path to ``scorep-info`` executable.
``SCOREP_ENABLE_COMPILER``
  Default setting for automatic compiler-based instrumentation of targets.
``SCOREP_ENABLE_KOKKOS``
  Default setting for KOKKOS instrumentation of targets.
``SCOREP_ENABLE_USER``
  Default setting for manual user insttrumentation of targets.
``SCOREP_IO_SYSTEM``
  Default value for system-level IO interception of targets.
``SCOREP_MEMORY_SYSTEM``
  Default list of memory allocation tracking of targets.
``SCOREP_MPP_SYSTEM
  Default value for MPP system of targets.
``SCOREP_THREADING_SYSTEM``
  Default value for threading system of targets.
   

#]================================================================================]

# _scorep_check_support_yesno
#
# checks the yes/no value in Score-P's config summary for a given key string.
#
# _string   The string to search for (has to be regex escaped)
#
# _comp     The corresponding component.
#
# Sets ``Scorep_${comp}_FOUND`` if entry has a value of 'yes' (i.e. is
# supported.
#
macro(_scorep_check_support_yesno _string _comp)
    string(TOUPPER "${_comp}" _upper_comp)
    string(REGEX MATCH " ${_string}:  *([^, \r\n]*)" _match ${SCOREP_CONFIG_SUMMARY})
    if (NOT "${_match}" STREQUAL "")
        string(FIND "${_match}" ":" _cloc)
        math(EXPR _cloc "${_cloc}+1")
        string(SUBSTRING ${_match} ${_cloc} -1 _value)
        string(STRIP ${_value} _value)
        if(${_value} STREQUAL "yes")
            set(Scorep_${_upper_comp}_FOUND True)
        else()
            set(Scorep_${_upper_comp}_FOUND False)
        endif()
    else()
        set(Scorep_${_upper_comp}_FOUND False)
    endif()
endmacro()

macro(_scorep_check_support_backend _string _comp)
    string(FIND "${SCOREP_CONFIG_SUMMARY}" "(${_string} backend)" _match)
    if (NOT "${_match}" EQUAL -1)
        set(Scorep_${_comp}_FOUND True)
    else()
        set(Scorep_${_comp}_FOUND False)
    endif()
endmacro()

macro(_scorep_configure_subsystem _target _prop _arg_if_yes _arg_if_no)
    get_target_property(_prop_value ${_target} ${_prop})
    message(DEBUG "Evaluating property ${_prop} on ${_target}: ${_prop_value}")
    if (${_prop_value} STREQUAL "no" OR ${_prop_value} STREQUAL "none" OR ${_prop_value} STREQUAL "_prop_value-NOTFOUND" OR ${_prop_value} STREQUAL FALSE OR ${_prop_value} STREQUAL OFF)
        set(SCOREP_CONFIG_TARGET_FLAGS ${SCOREP_CONFIG_TARGET_FLAGS} ${_arg_if_no})
    else()
        string(REPLACE "@arg@" "${_prop_value}" _arg "${_arg_if_yes}")
        set(SCOREP_CONFIG_TARGET_FLAGS ${SCOREP_CONFIG_TARGET_FLAGS} ${_arg})
    endif()
endmacro()

macro(_scorep_test_and_append_system_support _list _var _option)
    if (${_var})
        list(APPEND ${_list} "${_option}")
    endif()
endmacro()

macro(_scorep_set_enable_property _key)
    if (DEFINED SCOREP_ARGS_${_key} AND NOT "${SCOREP_ARGS_${_key}}" STREQUAL "none")
        set_property(TARGET ${_target} PROPERTY SCOREP_ENABLE_${_key} ${SCOREP_ARGS_${_key}})
    endif()
endmacro()

macro(_scorep_set_system_property _key)
    set(options UNIQUE)
    cmake_parse_arguments(ARGS "${options}" "" "" ${ARGN})

    if (DEFINED SCOREP_ARGS_${_key})
        set_property(TARGET ${_target} PROPERTY SCOREP_${_key}_SYSTEM ${SCOREP_ARGS_${_key}})
        if (ARGS_UNIQUE)
            set_property(TARGET ${_target} PROPERTY INTERFACE_SCOREP_${_key}_SYSTEM ${SCOREP_ARGS_${_key}})
            set_property(TARGET ${_target} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING SCOREP_${_key}_SYSTEM)
        endif()
    endif()
endmacro()

function(scorep_instrument_target _target)

    # NOTE: COMPILER, KOKKOS, and USER have to be value-args to be able to test
    # wether the argument is defined (to be able to override the default) 
    set(singleValueArgs COMPILER KOKKOS USER CUDA IO MEMORY MPP THREADING)
    cmake_parse_arguments(SCOREP_ARGS "" "${singleValueArgs}" "" ${ARGN})

    # Override target property if explicitly requested
    _scorep_set_enable_property(COMPILER)
    _scorep_set_enable_property(KOKKOS)
    _scorep_set_enable_property(USER)
    _scorep_set_system_property(MEMORY PROPAGATE)
    _scorep_set_system_property(MPP PROPAGATE UNIQUE)
    _scorep_set_system_property(THREADING PROPAGATE UNIQUE)

    if (DEFINED SCOREP_ARGS_MEMORY)
        set_property(TARGET ${_target} PROPERTY SCOREP_MEMORY_SYSTEM ${SCOREP_ARGS_MEMORY})
    endif()
    if (DEFINED SCOREP_ARGS_MPP)
        set_property(TARGET ${_target} PROPERTY SCOREP_MPP_SYSTEM ${SCOREP_ARGS_MPP})
    endif()
    if (DEFINED SCOREP_ARGS_THREADING)
        if (NOT Scorep_OMPT_FOUND AND "${SCOREP_ARGS_THREADING}" STREQUAL "omp:ompt")
            set(SCOREP_ARGS_THREADING "pthread")
            message(WARNING "Overriding unavailable selected threading system 'omp:ompt' to 'pthread'")
        endif()
        set_property(TARGET ${_target} PROPERTY SCOREP_THREADING_SYSTEM ${SCOREP_ARGS_THREADING})
    endif()

    _scorep_configure_subsystem(${_target} SCOREP_ENABLE_COMPILER "--compiler" "--nocompiler")
    _scorep_configure_subsystem(${_target} SCOREP_ENABLE_KOKKOS "--kokkos" "--nokokkos")
    _scorep_configure_subsystem(${_target} SCOREP_ENABLE_USER "--user" "--nouser")
    _scorep_configure_subsystem(${_target} SCOREP_ENABLE_CUDA "--cuda" "--nocuda")
    _scorep_configure_subsystem(${_target} SCOREP_IO_SYSTEM "--io=@arg@" "--io=none")
    _scorep_configure_subsystem(${_target} SCOREP_MEMORY_SYSTEM "--io=@arg@" "--io=none")
    _scorep_configure_subsystem(${_target} SCOREP_MPP_SYSTEM "--mpp=@arg@" "--mpp=none")
    _scorep_configure_subsystem(${_target} SCOREP_THREADING_SYSTEM "--thread=@arg@" "--thread=none")

    message(DEBUG "Instrumenting target '${_target}' using options ${SCOREP_CONFIG_TARGET_FLAGS}")

    # Create adapter initialization
    set(outfile ${CMAKE_CURRENT_BINARY_DIR}/${_target}_scorep_adapter_init.c)
    add_custom_command(
        OUTPUT ${outfile}
        COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--adapter-init" ${SCOREP_CONFIG_TARGET_FLAGS}
            "--user" "--compiler" > ${outfile}
        COMMENT "Generating Score-P adapter initialization for target '${_target}'"
        )
    target_sources(${_target} PRIVATE $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${outfile}>)

    # get reference to measurement constructor object
    execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--constructor" OUTPUT_VARIABLE _SCOREP_CONSTRUCTOR)
        string(STRIP ${_SCOREP_CONSTRUCTOR} _SCOREP_CONSTRUCTOR)
        set_source_files_properties(${_SCOREP_CONSTRUCTOR} PROPERTIES
            EXTERNAL_OBJECT TRUE
            GENERATED TRUE)
        target_sources(${_target} PRIVATE $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:${_SCOREP_CONSTRUCTOR}>)
    
    # get library directories for linker
    execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--ldflags" ${SCOREP_CONFIG_TARGET_FLAGS}
        OUTPUT_VARIABLE _LINK_LD_ARGS)
    # Combine Score-P's rpath statements into a single argument
    string(REGEX REPLACE "-Wl,-rpath -Wl,([^ ]*)" "-Wl,-rpath,\\1" _LINK_LD_ARGS ${_LINK_LD_ARGS})
    list(LENGTH "${_LINK_LD_ARGS}" _NUM_LINK_LD_ARGS)
    string(REPLACE " " ";" _LINK_LD_ARGS "${_LINK_LD_ARGS}")
    foreach( _ARG ${_LINK_LD_ARGS} )
        if(${_ARG} MATCHES "^-L")
            string(REGEX REPLACE "^-L" "" _ARG ${_ARG})
            set(SCOREP_LINK_DIRS ${SCOREP_LINK_DIRS} ${_ARG})
        else()
            set(SCOREP_LINK_OPTIONS ${SCOREP_LINK_OPTIONS} "${_ARG}")
        endif()
    endforeach()
    set_property(TARGET ${_target} APPEND
        PROPERTY LINK_DIRECTORIES ${SCOREP_LINK_DIRS})
    set_property(TARGET ${_target} APPEND
        PROPERTY LINK_OPTIONS ${SCOREP_LINK_OPTIONS})

    # determine link libraries for target
    execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--event-libs" ${SCOREP_CONFIG_TARGET_FLAGS}
        --user --compiler
        OUTPUT_VARIABLE _LINK_LD_ARGS)
    string(REPLACE " " ";" _LINK_LD_ARGS "${_LINK_LD_ARGS}")
    set_property(TARGET ${_target} APPEND
        PROPERTY LINK_LIBRARIES ${_LINK_LD_ARGS})
    
    # determine link libraries for target
    execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--mgmt-libs" ${SCOREP_CONFIG_TARGET_FLAGS}
        --user --compiler
        OUTPUT_VARIABLE _LINK_LD_ARGS)
    string(REPLACE " " ";" _LINK_LD_ARGS "${_LINK_LD_ARGS}")
    set_property(TARGET ${_target} APPEND
        PROPERTY LINK_LIBRARIES ${_LINK_LD_ARGS})
    
    # Determine compile options for compiler-based instrumentation
    execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--cflags" ${SCOREP_CONFIG_TARGET_FLAGS}
        OUTPUT_VARIABLE _SCOREP_C_FLAGS)
    string(REPLACE " " ";" _SCOREP_C_FLAGS "${_SCOREP_C_FLAGS}")
    foreach(flag ${_SCOREP_C_FLAGS})
        string(STRIP ${flag} flag)
        if(NOT ${flag} MATCHES "^-I")
            list(APPEND SCOREP_C_COMPILER_INSTRUMENTATION ${flag})
        endif()
    endforeach()

    execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--cxxflags" ${SCOREP_CONFIG_TARGET_FLAGS}
        OUTPUT_VARIABLE _SCOREP_CXX_FLAGS)
    string(REPLACE " " ";" _SCOREP_CXX_FLAGS "${_SCOREP_CXX_FLAGS}")
    foreach(flag ${_SCOREP_CXX_FLAGS})
        string(STRIP ${flag} flag)
        if(NOT ${flag} MATCHES "^-I")
            list(APPEND SCOREP_CXX_COMPILER_INSTRUMENTATION ${flag})
        endif()
    endforeach()

    execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--fflags" ${SCOREP_CONFIG_TARGET_FLAGS}
        OUTPUT_VARIABLE _SCOREP_Fortran_FLAGS)
    string(REPLACE " " ";" _SCOREP_Fortran_FLAGS "${_SCOREP_Fortran_FLAGS}")
    foreach(flag ${_SCOREP_Fortran_FLAGS})
        string(STRIP ${flag} flag)
        if(NOT ${flag} MATCHES "^-I")
            list(APPEND SCOREP_Fortran_COMPILER_INSTRUMENTATION ${flag})
        endif()
    endforeach()

    set_property(TARGET ${_target} APPEND
        PROPERTY COMPILE_OPTIONS
            $<$<COMPILE_LANGUAGE:C>:${SCOREP_C_COMPILER_INSTRUMENTATION}>
            $<$<COMPILE_LANGUAGE:CXX>:${SCOREP_CXX_COMPILER_INSTRUMENTATION}>
            $<$<COMPILE_LANGUAGE:Fortran>:${SCOREP_Fortran_COMPILER_INSTRUMENTATION}>
    )

    # Determine compile definitions for target
    execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--cppflags" ${SCOREP_CONFIG_TARGET_FLAGS}
        OUTPUT_VARIABLE _SCOREP_CPP_FLAGS)
    string(REPLACE " " ";" _SCOREP_CPP_FLAGS "${_SCOREP_CPP_FLAGS}")
    foreach(flag ${_SCOREP_CPP_FLAGS})
        string(STRIP ${flag} flag)
        if(${flag} MATCHES "^-D")
            list(APPEND SCOREP_COMPILE_DEFINITIONS ${flag})
        endif()
    endforeach()
    set_property(TARGET ${_target} APPEND
        PROPERTY COMPILE_DEFINITIONS "${SCOREP_COMPILE_DEFINITIONS_DIRS}"
    )

    # Determine include paths
    execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--cppflags" OUTPUT_VARIABLE SCOREP_CONFIG_FLAGS)
    string(REGEX MATCHALL "-I[^ ]*" SCOREP_CONFIG_INCLUDES "${SCOREP_CONFIG_FLAGS}")
    foreach(inc ${SCOREP_CONFIG_INCLUDES})
        string(SUBSTRING ${inc} 2 -1 inc)
        list(APPEND SCOREP_INCLUDE_DIRS ${inc})
    endforeach()
    set_property(TARGET ${_target} APPEND
        PROPERTY INCLUDE_DIRECTORIES ${SCOREP_INCLUDE_DIRS}
    )
    
endfunction(scorep_instrument_target)

#[=========================================================================[
    Find scorep-config and scorep-info
#]=========================================================================]

# Check for different locations to search for
if (SCOREP_ROOT_DIR)
    set(SCOREP_SEARCH_PATHS ${SCOREP_SEARCH_PATH} ${SCOREP_ROOT_DIR}/bin)
endif()
if ($ENV{EBROOTSCOREMINP})
    set(SCOREP_SEARCH_PATHS ${SCOREP_SEARCH_PATH} $ENV{RBROOTSCOREMINP}/bin)
endif()
set (SCOREP_SEARCH_PATHS ${SCOREP_SEARCH_PATHS} /opt/scorep/bin)

find_program(SCOREP_INFO_EXECUTABLE NAMES scorep-info
    PATHS
    ${SCOREP_SEARCH_PATHS}
)
mark_as_advanced(SCOREP_INFO_EXECUTABLE)

find_program(SCOREP_CONFIG_EXECUTABLE NAMES scorep-config
    PATHS
    ${SCOREP_SEARCH_PATHS}
)
mark_as_advanced(SCOREP_CONFIG_EXECUTABLE)

if(NOT SCOREP_CONFIG_EXECUTABLE OR NOT SCOREP_INFO_EXECUTABLE)
    set(SCOREP_FOUND false)
else(NOT SCOREP_CONFIG_EXECUTABLE OR NOT SCOREP_INFO_EXECUTABLE)

    execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--version" OUTPUT_VARIABLE SCOREP_VERSION)
    execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} "--prefix" OUTPUT_VARIABLE _SCOREP_ROOT_DIR)
    set(SCOREP_ROOT_DIR "${_SCOREP_ROOT_DIR}" CACHE PATH "Root directory of Score-P installation.")
    mark_as_advanced(SCOREP_ROOT_DIR)

    #[=========================================================================[
        Check components supported by Score-P installation
    #]=========================================================================]

    # Get config-summary from scorep-info
    execute_process(
        COMMAND ${SCOREP_INFO_EXECUTABLE} config-summary
        OUTPUT_VARIABLE SCOREP_CONFIG_SUMMARY)

    # Query components
    _scorep_check_support_yesno("Pthread support" PTHREAD)
    _scorep_check_support_yesno("OpenMP support" OPENMP)
    _scorep_check_support_yesno("OMPT support" OMPT)
    _scorep_check_support_yesno("opari2 support" OPARI2)
    _scorep_check_support_yesno("Sampling support" SAMPLING)
    _scorep_check_support_yesno("CUDA support" CUDA)
    _scorep_check_support_yesno("Memory tracking support" MEMORY)
    _scorep_check_support_yesno("OpenCL support" OPENCL)
    _scorep_check_support_yesno("OpenACC support" OPENACC)
    _scorep_check_support_yesno("Library wrapper support" LIBWRAP)
    _scorep_check_support_yesno("POSIX I/O support" POSIXIO)
    _scorep_check_support_yesno("Kokkos support" KOKKOS)
    _scorep_check_support_backend("SHMEM" SHMEM)
    _scorep_check_support_backend("MPI" MPI)
    
    unset(SCOREP_CONFIG_FLAGS)
    unset(SCOREP_CONFIG_INCLUDES)
    unset(SCOREP_CONFIG_CXXFLAGS)

    #[=========================================================================[
        Defining Target Properties
    #]=========================================================================]
    macro(_scorep_define_enable_property _keyword _default _doc)
        define_property(TARGET
            PROPERTY SCOREP_ENABLE_${_keyword} INHERITED
            BRIEF_DOCS "${_doc}"
            INITIALIZE_FROM_VARIABLE SCOREP_ENABLE_${_keyword})
        # setting no threading as default
        set(SCOREP_ENABLE_${_keyword} ${_default} CACHE BOOL "${_doc}")
        if (NOT "${_strings}" STREQUAL "")
            set_property(CACHE SCOREP_ENABLE_${_keyword} PROPERTY STRINGS ${_strings})
        endif()
        mark_as_advanced(SCOREP_ENABLE_${_keyword})
    endmacro()

    macro(_scorep_define_system_property _keyword _default _strings _doc)
        define_property(TARGET
            PROPERTY SCOREP_${_keyword}_SYSTEM INHERITED
            BRIEF_DOCS "${_doc}"
            INITIALIZE_FROM_VARIABLE SCOREP_${_keyword}_SYSTEM)
        # setting no threading as default
        set(SCOREP_${_keyword}_SYSTEM ${_default} CACHE STRING "${_doc}")
        if (NOT "${_strings}" STREQUAL "")
            set_property(CACHE SCOREP_${_keyword}_SYSTEM PROPERTY STRINGS ${_strings})
        endif()
        mark_as_advanced(SCOREP_${_keyword}_SYSTEM)
    endmacro()

    _scorep_define_enable_property(COMPILER ON "Default automatic compiler-based instrumentation for targets.")
    _scorep_define_enable_property(KOKKOS OFF "Default Kokkos instrumentation for targets.")
    _scorep_define_enable_property(USER OFF "Default manual user-instrumentation for targets.")
    _scorep_define_enable_property(CUDA OFF "Default CUDA instrumentation for targets.")
    list(APPEND SUPPORTED_IO_SYSTEMS "none")
    _scorep_test_and_append_system_support(SUPPORTED_IO_SYSTEMS Scorep_POSIXIO_FOUND "posix")
    _scorep_define_system_property(IO none "${SUPPORTED_IO_SYSTEMS}" "I/O subsystem to intercept by default.")
    unset(SUPPORTED_IO_SYSTEMS)
    _scorep_define_system_property(MEMORY none "" "List of memory systems to track by default.")
    list(APPEND SUPPORTED_MPP_SYSTEMS "none")
    _scorep_test_and_append_system_support(SUPPORTED_MPP_SYSTEMS Scorep_MPI_FOUND "mpi")
    _scorep_test_and_append_system_support(SUPPORTED_MPP_SYSTEMS Scorep_SHMEM_FOUND "shmem")
    _scorep_define_system_property(MPP none "${SUPPORTED_MPP_SYSTEMS}" "Default MPP system for targets.")
    unset(SUPPORTED_MPP_SYSTEMS)

    list(APPEND SUPPORTED_THREADING_SYSTEMS "none")
    _scorep_test_and_append_system_support(SUPPORTED_THREADING_SYSTEMS Scorep_PTHREAD_FOUND "pthread")
    _scorep_test_and_append_system_support(SUPPORTED_THREADING_SYSTEMS Scorep_OMPT_FOUND "omp:ompt")
    _scorep_define_system_property(THREADING none "${SUPPORTED_THREADING_SYSTEMS}" "Default threading system for targets.")
    unset(SUPPORTED_THREADING_SYSTEMS)

    set(SCOREP_FOUND true)
endif()

################################################################################
# Exported Targets
################################################################################

add_library(Scorep::Plugin INTERFACE IMPORTED)
set_target_properties(Scorep::Plugin PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SCOREP_INCLUDE_DIRS}"
)

include (FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Scorep
    FOUND_VAR SCOREP_FOUND 
    VERSION_VAR SCOREP_VERSION
    REQUIRED_VARS SCOREP_ROOT_DIR SCOREP_CONFIG_EXECUTABLE SCOREP_INFO_EXECUTABLE
    HANDLE_COMPONENTS
)