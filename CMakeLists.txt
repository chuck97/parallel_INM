# required version of cmake
cmake_minimum_required (VERSION 3.17.0)

# project name
project(MVM)

# enabled languages
enable_language(C)

# options for MPI and OpenMP support
option(USE_MPI "Compile with MPI support" OFF)
option(USE_OMP "Compile with OpenMP support" OFF)

# option for intel compiler (default is gnu)
option(INTEL_COMPILER "Intel compiler" OFF)

# manage otion for intel compiler
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    MESSAGE(STATUS "Intel compiler")
    add_definitions(-DINTEL_COMPILER)
endif()

# manage option for mpi
if (USE_MPI)
    MESSAGE(STATUS "MPI enabled")
    add_definitions(-DUSE_MPI)
endif()

# manage option for omp
if (USE_OMP)
    MESSAGE(STATUS "OpenMP enabled")
    add_definitions(-DUSE_OMP)
endif()

# add compiler -O2 option
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")

# add compiler openmp option if OpenMP is enabled
if (USE_OMP)
    if (INTEL_COMPILER)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    endif()
endif()

# create executable from the source code 
add_executable(mvm mvm.c)

# link executable with math library
target_link_libraries(mvm PRIVATE m)