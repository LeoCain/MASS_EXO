cmake_minimum_required(VERSION 2.8.6)
project(exo_render)

link_directories(../core/)
include_directories(../core/)

find_package(DART REQUIRED COMPONENTS gui collision-bullet CONFIG)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${DART_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(/home/medrobotics/MASSExo/MASSMerge/MASS_EXO/render)

file(GLOB srcs "*.h" "*.cpp"
    "${CMAKE_HOME_DIRECTORY}/render/Window.h"
    "${CMAKE_HOME_DIRECTORY}/render/Window.cpp"
)
add_executable(exo_render ${srcs})
target_link_libraries(exo_render ${DART_LIBRARIES} ${PYTHON_LIBRARIES} GL GLU glut mss pybind11::module pybind11::embed)