cmake_minimum_required(VERSION 2.8)

project(slow_flow)

set(CMAKE_C_COMPILER_INIT g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -g -O3 -msse4") 

set(MIDDLEBURY_DEVKIT "[SPECIFY PATH TO MIDDLEBURY DEVKIT]")

# opencv library
find_package(OpenCV REQUIRED)

# boost library
find_package(Boost REQUIRED COMPONENTS system filesystem chrono regex)

# eigen library
find_package(Eigen3 REQUIRED )

# PNG library
find_package(PNG REQUIRED)

# JPEG library
find_package(JPEG REQUIRED)

# Img library
find_path(IMG_INCLUDE_DIRS Image.h ${MIDDLEBURY_DEVKIT}/cpp/imageLib)
find_library(IMG_LIBRARY NAMES Img PATHS ${MIDDLEBURY_DEVKIT}/cpp/imageLib)
add_library(Img STATIC IMPORTED)

# LAPACK library
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

# GSL library
FIND_PACKAGE(GSL REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PNG_INCLUDE_DIR})
include_directories(${JPEG_INCLUDE_DIR})
include_directories(${IMG_INCLUDE_DIRS})
include_directories(${BLAS_INCLUDE_DIR})
include_directories(${LAPACK_INCLUDE_DIR})
include_directories(${GSLCBLAS_INCLUDE_DIR})
include_directories(${GSL_INCLUDE_DIR})

link_directories(${Boost_LIBRARY_DIR})

# parallel computing
find_package(OpenMP)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(slow_flow slow_flow.cpp
    epic_flow_extended/epic_aux.cpp
    epic_flow_extended/epic.cpp
    epic_flow_extended/image.c
    epic_flow_extended/io.c
    epic_flow_extended/solver.c
    epic_flow_extended/variational_aux_mt.cpp
    epic_flow_extended/variational_mt.cpp
   	utils/utils.cpp
    utils/parameter_list.cpp
    ../libs/middlebury_devkit/cpp/colorcode.cpp
    ../libs/middlebury_devkit/cpp/flowIO.cpp
    ../libs/gco/GCoptimization.cpp
    ../libs/gco/maxflow.cpp
    ../libs/gco/graph.cpp
    ../libs/gco/LinkedBlockList.cpp
    ../libs/dmgunturk/dmha.c
    ../libs/dmgunturk/dmbilinear.c
    ../libs/dmgunturk/basic.c
    ../libs/dmgunturk/imageio.c)

add_executable(adaptiveFR adaptiveFR.cpp
    epic_flow_extended/epic_aux.cpp
    epic_flow_extended/epic.cpp
    epic_flow_extended/image.c
    epic_flow_extended/io.c
    epic_flow_extended/solver.c
    epic_flow_extended/variational_aux.c
    epic_flow_extended/variational.c
   	utils/utils.cpp
    utils/parameter_list.cpp
    ../libs/middlebury_devkit/cpp/colorcode.cpp
    ../libs/middlebury_devkit/cpp/flowIO.cpp
    ../libs/gco/GCoptimization.cpp
    ../libs/gco/maxflow.cpp
    ../libs/gco/graph.cpp
    ../libs/gco/LinkedBlockList.cpp
    ../libs/dmgunturk/dmha.c
    ../libs/dmgunturk/dmbilinear.c
    ../libs/dmgunturk/basic.c
    ../libs/dmgunturk/imageio.c)

add_executable(dense_tracking dense_tracking.cpp
    epic_flow_extended/epic_aux.cpp
    epic_flow_extended/epic.cpp
    epic_flow_extended/image.c
    epic_flow_extended/io.c
    epic_flow_extended/solver.c
    epic_flow_extended/variational_aux_mt.cpp
    epic_flow_extended/variational_mt.cpp
   	utils/hypothesis.cpp
   	utils/utils.cpp
    utils/parameter_list.cpp
   	penalty_functions/penalty_function.h
   	penalty_functions/lorentzian.h
   	penalty_functions/modified_l1_norm.h
   	penalty_functions/quadratic_function.h
    ../libs/middlebury_devkit/cpp/colorcode.cpp
    ../libs/middlebury_devkit/cpp/flowIO.cpp
    ../libs/maxflow/maxflow.cpp
    ../libs/maxflow/graph.cpp
    ../libs/gco/GCoptimization.cpp
    ../libs/gco/maxflow.cpp
    ../libs/gco/graph.cpp
    ../libs/gco/LinkedBlockList.cpp
    ../libs/dmgunturk/dmha.c
    ../libs/dmgunturk/dmbilinear.c
    ../libs/dmgunturk/basic.c
    ../libs/dmgunturk/imageio.c)

target_link_libraries(slow_flow ${OpenCV_LIBS} ${Boost_LIBRARIES} ${IMG_LIBRARY} ${PNG_LIBRARY} ${JPEG_LIBRARY} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
target_link_libraries(adaptiveFR ${OpenCV_LIBS} ${Boost_LIBRARIES} ${IMG_LIBRARY} ${PNG_LIBRARY} ${JPEG_LIBRARY} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
target_link_libraries(dense_tracking ${OpenCV_LIBS} ${Boost_LIBRARIES} ${IMG_LIBRARY} ${PNG_LIBRARY} ${JPEG_LIBRARY} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})