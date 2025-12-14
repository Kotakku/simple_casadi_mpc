cmake_minimum_required(VERSION 3.14)

option(BUILD_EXAMPLES "Build example executables" OFF)
option(BUILD_BENCHMARKS "Build benchmark executables" OFF)

project(simple_casadi_mpc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

set(HEADER_FILES
    ${CMAKE_SOURCE_DIR}/include/simple_casadi_mpc/simple_casadi_mpc.hpp
    ${CMAKE_SOURCE_DIR}/include/simple_casadi_mpc/casadi_utils.hpp)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                            $<INSTALL_INTERFACE:include>)

include(GNUInstallDirs)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# build examples

if(BUILD_EXAMPLES)
  find_package(PkgConfig REQUIRED)
  find_package(casadi REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(
    Python3
    COMPONENTS Interpreter Development NumPy
    REQUIRED)
  find_package(pybind11 REQUIRED)
  find_package(matplotlibcpp17 REQUIRED)

  pkg_check_modules(IPOPT REQUIRED ipopt)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(simple_casadi_mpc_codegen)

  add_executable(double_integrator_mpc_example
                 example/double_integrator_mpc_example.cpp)
  target_include_directories(double_integrator_mpc_example
                             PRIVATE ${CMAKE_SOURCE_DIR}/example)
  target_link_libraries(
    double_integrator_mpc_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME}
                                          Python3::Python Python3::NumPy
                                          pybind11::embed
                                          matplotlibcpp17::matplotlibcpp17)

  add_executable(cartpole_mpc_example example/cartpole_mpc_example.cpp)
  target_link_libraries(
    cartpole_mpc_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME}
                                 Python3::Python Python3::NumPy pybind11::embed
                                 matplotlibcpp17::matplotlibcpp17)

  add_executable(inverted_pendulum_mpc_example
                 example/inverted_pendulum_mpc_example.cpp)
  target_link_libraries(
    inverted_pendulum_mpc_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME}
                                          Python3::Python Python3::NumPy pybind11::embed
                                          matplotlibcpp17::matplotlibcpp17)

  add_executable(diff_drive_mpc_example example/diff_drive_mpc_example.cpp)
  target_link_libraries(
    diff_drive_mpc_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME}
                                   Python3::Python Python3::NumPy pybind11::embed
                                   matplotlibcpp17::matplotlibcpp17)

  add_executable(cartpole_mpc_jit_example example/cartpole_mpc_jit_example.cpp)
  target_link_libraries(
    cartpole_mpc_jit_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME}
                                     Python3::Python Python3::NumPy pybind11::embed
                                     matplotlibcpp17::matplotlibcpp17)

  add_simple_casadi_mpc_codegen(
    double_integrator_problem
    example/double_integrator_mpc_codegen.cpp
    EXPORT_SOLVER_NAME
    double_integrator_solver
    INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/example
    LINK_LIBS
    fatrop
    blasfeo)

  add_executable(double_integrator_mpc_compiled_example
                 example/double_integrator_mpc_compiled_example.cpp)
  target_sources(
    double_integrator_mpc_compiled_example
    PRIVATE ${double_integrator_problem_COMPILED_SOLVER_CONFIG_SOURCE})
  add_dependencies(double_integrator_mpc_compiled_example
                   double_integrator_problem_compiled_solver)
  target_include_directories(
    double_integrator_mpc_compiled_example
    PRIVATE ${CMAKE_SOURCE_DIR}/example
            ${double_integrator_problem_CODEGEN_DIR})
  target_link_libraries(
    double_integrator_mpc_compiled_example
    PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME}
            double_integrator_problem_compiled_solver)

  add_simple_casadi_mpc_codegen(
    cartpole_problem
    example/cartpole_mpc_codegen.cpp
    EXPORT_SOLVER_NAME
    cartpole_solver
    INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/example
    LINK_LIBS
    fatrop
    blasfeo)

  add_executable(cartpole_mpc_codegen_example
                 example/cartpole_mpc_codegen_example.cpp)
  target_sources(cartpole_mpc_codegen_example
                 PRIVATE ${cartpole_problem_COMPILED_SOLVER_CONFIG_SOURCE})
  add_dependencies(cartpole_mpc_codegen_example
                   cartpole_problem_compiled_solver)
  target_include_directories(
    cartpole_mpc_codegen_example PRIVATE ${CMAKE_SOURCE_DIR}/example
                                         ${cartpole_problem_CODEGEN_DIR})
  target_link_libraries(
    cartpole_mpc_codegen_example
    PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME}
            cartpole_problem_compiled_solver pybind11::embed
            matplotlibcpp17::matplotlibcpp17)

endif()

# build benchmarks
if(BUILD_BENCHMARKS)
  find_package(PkgConfig REQUIRED)
  find_package(casadi REQUIRED)
  find_package(Eigen3 REQUIRED)
  find_package(
    Python3
    COMPONENTS Interpreter Development NumPy
    REQUIRED)
  find_package(pybind11 REQUIRED)
  find_package(matplotlibcpp17 REQUIRED)

  pkg_check_modules(IPOPT REQUIRED ipopt)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(simple_casadi_mpc_codegen)

  add_simple_casadi_mpc_codegen(
    cartpole_solver_bench
    benchmark/cartpole_mpc_codegen.cpp
    EXPORT_SOLVER_NAME
    cartpole_solver_bench
    INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/benchmark
    LINK_LIBS
    fatrop
    blasfeo)

  add_executable(cartpole_mpc_solver_bench
                 benchmark/cartpole_mpc_solver_bench.cpp)
  target_sources(cartpole_mpc_solver_bench
                 PRIVATE ${cartpole_solver_bench_COMPILED_SOLVER_CONFIG_SOURCE})
  add_dependencies(cartpole_mpc_solver_bench
                   cartpole_solver_bench_compiled_solver)
  target_include_directories(
    cartpole_mpc_solver_bench PRIVATE ${CMAKE_SOURCE_DIR}/benchmark
                                      ${cartpole_solver_bench_CODEGEN_DIR})
  target_link_libraries(
    cartpole_mpc_solver_bench
    PRIVATE casadi
            Eigen3::Eigen
            ${PROJECT_NAME}
            cartpole_solver_bench_compiled_solver
            Python3::Python
            Python3::NumPy
            pybind11::embed
            matplotlibcpp17::matplotlibcpp17)

endif()
