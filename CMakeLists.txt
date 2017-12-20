cmake_minimum_required(VERSION 2.8)

project(slow_flow)

set(CMAKE_C_COMPILER_INIT g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -g -O3 -msse4") 


############################# PLEASE SPECIFY ############################# 
set(MIDDLEBURY_DEVKIT_PATH "[SPECIFY PATH TO MIDDLEBURY DEVKIT]")					# Middlebury Devkit
set(GCO_PATH "[SPECIFY PATH TO GRAPH CUT]")  										# Graph cut library
set(TRWS_PATH "[SPECIFY PATH TO TRWS]")												# Tree-Reweigthed Message Passing
#set(DMGUNTURK_PATH "[SPECIFY PATH TO HAMILTON ADAMS DEMOSAICING]")					# Gunturk-Altunbasak-Mersereau Alternating Projections Image Demosaicking 
############################# PLEASE SPECIFY ############################# 

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
find_path(IMG_INCLUDE_DIRS Image.h PATHS ${MIDDLEBURY_DEVKIT_PATH}/imageLib/ NO_DEFAULT_PATH) 
find_library(IMG_LIBRARY NAMES Img PATHS ${MIDDLEBURY_DEVKIT_PATH}/imageLib/)
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
include_directories(${TRWS_PATH})

link_directories(${Boost_LIBRARY_DIR})

# INCLUDES
set(EPIC_FLOW_EXTENDED_INCLUDES epic_flow_extended/epic_aux.cpp
    epic_flow_extended/epic.cpp
    epic_flow_extended/image.c
    epic_flow_extended/io.c
    epic_flow_extended/solver.c
    epic_flow_extended/variational_aux_mt.cpp
    epic_flow_extended/variational_mt.cpp)
set(EPIC_FLOW_INCLUDES epic_flow_extended/epic_aux.cpp
    epic_flow_extended/epic.cpp
    epic_flow_extended/image.c
    epic_flow_extended/io.c
    epic_flow_extended/solver.c
    epic_flow_extended/variational_aux.c
    epic_flow_extended/variational.c)
set(UTILS_INCLUDES utils/utils.cpp utils/parameter_list.cpp)
set(MIDDLEBURY_INCLUDES ${MIDDLEBURY_DEVKIT_PATH}/colorcode.cpp ${MIDDLEBURY_DEVKIT_PATH}/flowIO.cpp)
set(GCO_INCLUDES ${GCO_PATH}/GCoptimization.cpp
    ${GCO_PATH}/maxflow.cpp
    ${GCO_PATH}/graph.cpp
    ${GCO_PATH}/LinkedBlockList.cpp)
if(DMGUNTURK_PATH)
	set(DMGUNTURK_INCLUDES ${DMGUNTURK_PATH}/dmha.c
	    ${DMGUNTURK_PATH}/dmbilinear.c
	    ${DMGUNTURK_PATH}/basic.c
	    ${DMGUNTURK_PATH}/imageio.c)
else()
	set(DMGUNTURK_INCLUDES )
endif()

# parallel computing
find_package(OpenMP)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(slow_flow slow_flow.cpp
    ${EPIC_FLOW_EXTENDED_INCLUDES}
    ${UTILS_INCLUDES}
    ${MIDDLEBURY_INCLUDES}
    ${GCO_INCLUDES}
    ${DMGUNTURK_INCLUDES})

add_executable(adaptiveFR adaptiveFR.cpp
    ${EPIC_FLOW_INCLUDES}
    ${UTILS_INCLUDES}
    ${MIDDLEBURY_INCLUDES}
    ${GCO_INCLUDES}
    ${DMGUNTURK_INCLUDES})

add_executable(dense_tracking dense_tracking.cpp
    ${EPIC_FLOW_EXTENDED_INCLUDES}
    ${UTILS_INCLUDES}
   	utils/hypothesis.cpp
   	penalty_functions/penalty_function.h
   	penalty_functions/lorentzian.h
   	penalty_functions/modified_l1_norm.h
   	penalty_functions/quadratic_function.h
    ${MIDDLEBURY_INCLUDES}
    ${TRWS_PATH}/MRFEnergy.cpp
    ${TRWS_PATH}/minimize.cpp
    ${TRWS_PATH}/ordering.cpp
    ${TRWS_PATH}/treeProbabilities.cpp
    ${GCO_INCLUDES}
    ${DMGUNTURK_INCLUDES})

target_link_libraries(slow_flow ${OpenCV_LIBS} ${Boost_LIBRARIES} ${IMG_LIBRARY} ${PNG_LIBRARY} ${JPEG_LIBRARY} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
target_link_libraries(adaptiveFR ${OpenCV_LIBS} ${Boost_LIBRARIES} ${IMG_LIBRARY} ${PNG_LIBRARY} ${JPEG_LIBRARY} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
target_link_libraries(dense_tracking ${OpenCV_LIBS} ${Boost_LIBRARIES} ${IMG_LIBRARY} ${PNG_LIBRARY} ${JPEG_LIBRARY} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})