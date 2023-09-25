cmake_minimum_required(VERSION 3.10)

project(simple_casadi_mpc CXX)

set(CMAKE_CXX_STANDARD 17)

set(HEADER_FILES
    ${CMAKE_SOURCE_DIR}/include/simple_casadi_mpc.hpp
)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(GNUInstallDirs)
install(
    FILES ${HEADER_FILES}
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
find_package(casadi REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Development NumPy)

add_executable(double_integrator_mpc_example example/double_integrator_mpc_example.cpp)
target_link_libraries(double_integrator_mpc_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME} Python3::Python Python3::NumPy)

add_executable(cartpole_mpc_example example/cartpole_mpc_example.cpp)
target_link_libraries(cartpole_mpc_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME} Python3::Python Python3::NumPy)

add_executable(inverted_pendulum_mpc_example example/inverted_pendulum_mpc_example.cpp)
target_link_libraries(inverted_pendulum_mpc_example PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME} Python3::Python Python3::NumPy)