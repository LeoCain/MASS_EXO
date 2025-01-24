cmake_minimum_required(VERSION 2.8.6)
project(pymss)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(DART REQUIRED COMPONENTS collision-bullet CONFIG)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

link_directories(../core/)
include_directories(../core/)
link_directories(../render/)
include_directories(../render/)

file(GLOB srcs "*.h" "*.cpp")

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${DART_INCLUDE_DIRS})

add_library(pymss SHARED ${srcs})
target_link_libraries(pymss ${DART_LIBRARIES} ${PYTHON_LIBRARIES} GL GLU glut mss pybind11::module pybind11::embed)
set_target_properties(pymss PROPERTIES PREFIX "" )

add_executable(display_model ../data/load_model.cpp)
target_link_libraries(display_model ${DART_LIBRARIES} ${PYTHON_LIBRARIES} mss pybind11::module)
set_target_properties(display_model PROPERTIES PREFIX "" )

install(TARGETS display_model DESTINATION build/)