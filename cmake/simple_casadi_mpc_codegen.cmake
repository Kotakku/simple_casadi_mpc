function(add_simple_casadi_mpc_codegen name codegen_cpp)
  set(options)
  set(oneValueArgs EXPORT_SOLVER_NAME LIB_TARGET SOLVER_NAME)
  set(multiValueArgs LINK_LIBS INCLUDE_DIRS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  if(NOT ARG_EXPORT_SOLVER_NAME)
    set(ARG_EXPORT_SOLVER_NAME ${name}_compiled_solver)
  endif()

  if(NOT ARG_LIB_TARGET)
    if(TARGET simple_casadi_mpc::simple_casadi_mpc)
      set(ARG_LIB_TARGET simple_casadi_mpc::simple_casadi_mpc)
    else()
      set(ARG_LIB_TARGET simple_casadi_mpc)
    endif()
  endif()

  # Allow users to pass additional solver-specific libraries if needed.
  set(link_libs ${ARG_LINK_LIBS})

  # Try to automatically link the casadi plugin library (brings solver deps).
  if(NOT link_libs)
    if(ARG_SOLVER_NAME)
      set(solver_name ${ARG_SOLVER_NAME})
    else()
      # Keep default consistent with example codegen
      set(solver_name fatrop)
    endif()

    # Collect possible casadi library directories to hint plugin lookup
    set(casadi_lib_dirs)
    foreach(casadi_target IN ITEMS casadi casadi::casadi)
      if(TARGET ${casadi_target})
        foreach(prop IN ITEMS IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION)
          get_target_property(_casadi_loc ${casadi_target} ${prop})
          if(_casadi_loc)
            get_filename_component(_casadi_dir ${_casadi_loc} DIRECTORY)
            list(APPEND casadi_lib_dirs ${_casadi_dir})
          endif()
        endforeach()
      endif()
    endforeach()
    list(REMOVE_DUPLICATES casadi_lib_dirs)

    find_library(
      casadi_nlpsol_${solver_name}_LIBRARY
      NAMES casadi_nlpsol_${solver_name}
      PATHS ${casadi_lib_dirs})
    if(casadi_nlpsol_${solver_name}_LIBRARY)
      # Ensure plugin is kept even with --as-needed to bring its dependencies.
      list(APPEND link_libs "-Wl,--no-as-needed"
           ${casadi_nlpsol_${solver_name}_LIBRARY} "-Wl,--as-needed")
    endif()
  endif()

  set(CODEGEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/${name}_codegen)

  # Ipopt codegen requires the coin-or header; provide a stub automatically.
  set(extra_include_dirs)
  if(solver_name STREQUAL "ipopt")
    set(ipopt_stub_dir ${CODEGEN_DIR}/coin-or)
    file(MAKE_DIRECTORY ${ipopt_stub_dir})
    file(
      WRITE ${ipopt_stub_dir}/IpStdCInterface.h
      "#include <coin/IpStdCInterface.h>\nusing ipindex = Index;\nusing ipnumber = Number;\n"
    )
    list(APPEND extra_include_dirs ${ipopt_stub_dir})
  endif()

  add_executable(${name}_codegen_bin ${codegen_cpp})
  if(ARG_INCLUDE_DIRS)
    target_include_directories(${name}_codegen_bin PRIVATE ${ARG_INCLUDE_DIRS})
  endif()

  target_link_libraries(
    ${name}_codegen_bin PRIVATE casadi Eigen3::Eigen ${ARG_LIB_TARGET}
                                ${link_libs})

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
                             PRIVATE ${CODEGEN_DIR} ${extra_include_dirs})
  target_link_libraries(${name}_compiled_solver PRIVATE casadi ${link_libs})
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
