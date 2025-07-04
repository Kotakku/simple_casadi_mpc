cmake_minimum_required(VERSION 3.10)

project(simple_casadi_mpc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")

set(HEADER_FILES
    ${CMAKE_SOURCE_DIR}/include/simple_casadi_mpc/simple_casadi_mpc.hpp
    ${CMAKE_SOURCE_DIR}/include/simple_casadi_mpc/casadi_utils.hpp
)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(GNUInstallDirs)
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# build examples
option(BUILD_EXAMPLES "Build example executables" ON)
if(BUILD_EXAMPLES)
    find_package(casadi REQUIRED)
    find_package(Eigen3 REQUIRED)
    find_package(Python3 COMPONENTS Development NumPy REQUIRED)

    add_executable(double_integrator_mpc_example example/double_integrator_mpc_example.cpp)
    target_link_libraries(double_integrator_mpc_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME} Python3::Python Python3::NumPy)

    add_executable(cartpole_mpc_example example/cartpole_mpc_example.cpp)
    target_link_libraries(cartpole_mpc_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME} Python3::Python Python3::NumPy)

    add_executable(inverted_pendulum_mpc_example example/inverted_pendulum_mpc_example.cpp)
    target_link_libraries(inverted_pendulum_mpc_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME} Python3::Python Python3::NumPy)

    add_executable(diff_drive_mpc_example example/diff_drive_mpc_example.cpp)
    target_link_libraries(diff_drive_mpc_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME} Python3::Python Python3::NumPy)
endif()