cmake_minimum_required(VERSION 2.8.6)
project(mss)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

find_package(DART 6.8 REQUIRED COMPONENTS collision-bullet gui utils utils-urdf)
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${DART_INCLUDE_DIRS})

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations -g)

add_compile_options(-DHAVE_CSTDDEF)
add_definitions(-DMASS_ROOT_DIR="${CMAKE_HOME_DIRECTORY}")
set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake")

add_subdirectory( core )
add_subdirectory( render )
add_subdirectory( python )

add_executable(load_model data/load_model.cpp)
target_link_libraries(load_model ${PYTHON_LIBRARIES} mss dart dart-collision-bullet dart-gui dart-utils dart-utils-urdf pybind11::module)

install(TARGETS load_model DESTINATION build/)