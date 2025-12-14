function(add_simple_casadi_mpc_codegen name codegen_cpp)
  set(options)
  set(oneValueArgs EXPORT_SOLVER_NAME)
  set(multiValueArgs LINK_LIBS INCLUDE_DIRS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  if(NOT ARG_EXPORT_SOLVER_NAME)
    set(ARG_EXPORT_SOLVER_NAME ${name}_compiled_solver)
  endif()

  # TODO: ipoptソルバ用のtempolaryな対応なので汎用性をもたせる
  set(CODEGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/${name}_codegen)
  file(MAKE_DIRECTORY ${CODEGEN_DIR}/coin-or)
  file(
    WRITE ${CODEGEN_DIR}/coin-or/IpStdCInterface.h
    "#include <coin/IpStdCInterface.h>\nusing ipindex = Index;\nusing ipnumber = Number;\n"
  )

  add_executable(${name}_codegen_bin ${codegen_cpp})
  if(ARG_INCLUDE_DIRS)
    target_include_directories(${name}_codegen_bin PRIVATE ${ARG_INCLUDE_DIRS})
  endif()

  # TODO: ソルバーごとのライブラリリンクをユーザーに任せてるので隠蔽・自動化する
  target_link_libraries(
    ${name}_codegen_bin PRIVATE casadi Eigen3::Eigen ${PROJECT_NAME}
                                ${ARG_LINK_LIBS})

  add_custom_command(
    OUTPUT ${CODEGEN_DIR}/${ARG_EXPORT_SOLVER_NAME}.c
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CODEGEN_DIR}
    COMMAND ${name}_codegen_bin
    WORKING_DIRECTORY ${CODEGEN_DIR}
    DEPENDS ${name}_codegen_bin
    COMMENT "Generate ${name} compiled solver source"
    VERBATIM)
  add_custom_target(${name}_codegen_target
                    DEPENDS ${CODEGEN_DIR}/${ARG_EXPORT_SOLVER_NAME}.c)

  add_library(${name}_compiled_solver SHARED
              ${CODEGEN_DIR}/${ARG_EXPORT_SOLVER_NAME}.c)
  add_dependencies(${name}_compiled_solver ${name}_codegen_target)
  set_source_files_properties(${CODEGEN_DIR}/${ARG_EXPORT_SOLVER_NAME}.c
                              PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-w")
  target_include_directories(${name}_compiled_solver
                             PRIVATE ${CODEGEN_DIR} ${IPOPT_INCLUDE_DIRS})
  target_link_libraries(${name}_compiled_solver
                        PRIVATE casadi ${IPOPT_LIBRARIES} ${ARG_LINK_LIBS})
  target_compile_options(${name}_compiled_solver PRIVATE -fpermissive)
  set_target_properties(${name}_compiled_solver
                        PROPERTIES OUTPUT_NAME ${ARG_EXPORT_SOLVER_NAME})

  set(compiled_solver_config_header
      ${CODEGEN_DIR}/${ARG_EXPORT_SOLVER_NAME}_config.hpp)
  set(compiled_solver_config_source
      ${CODEGEN_DIR}/${ARG_EXPORT_SOLVER_NAME}_config.cpp)
  file(
    GENERATE
    OUTPUT ${compiled_solver_config_header}
    CONTENT
      "#pragma once\n#include <simple_casadi_mpc/simple_casadi_mpc.hpp>\n\nsimple_casadi_mpc::CompiledMPC::CompiledLibraryConfig get_${ARG_EXPORT_SOLVER_NAME}_compiled_library_options();\n"
  )
  file(
    GENERATE
    OUTPUT ${compiled_solver_config_source}
    CONTENT
      "#include \"${ARG_EXPORT_SOLVER_NAME}_config.hpp\"\n\nsimple_casadi_mpc::CompiledMPC::CompiledLibraryConfig get_${ARG_EXPORT_SOLVER_NAME}_compiled_library_options()\n{\n    simple_casadi_mpc::CompiledMPC::CompiledLibraryConfig config;\n    config.export_solver_name = \"${ARG_EXPORT_SOLVER_NAME}\";\n    config.shared_library_path = \"$<TARGET_FILE:${name}_compiled_solver>\";\n    return config;\n}\n"
  )

  set(${name}_CODEGEN_DIR
      ${CODEGEN_DIR}
      PARENT_SCOPE)
  set(${name}_COMPILED_SOLVER
      ${name}_compiled_solver
      PARENT_SCOPE)
  set(${name}_COMPILED_SOLVER_CONFIG_HEADER
      ${compiled_solver_config_header}
      PARENT_SCOPE)
  set(${name}_COMPILED_SOLVER_CONFIG_SOURCE
      ${compiled_solver_config_source}
      PARENT_SCOPE)
endfunction()
