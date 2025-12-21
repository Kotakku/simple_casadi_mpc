#!/bin/sh

# Install dependencies
sudo apt install -y git cmake gcc g++ gfortran pkg-config liblapack-dev pkg-config coinor-libipopt-dev --install-recommends

# Clone CasADi repository
git clone https://github.com/casadi/casadi.git casadi

cd casadi
mkdir -p build
cd build

ARGS=""

#==============================================================================
# Build Type
#==============================================================================
ARGS="$ARGS -DCMAKE_BUILD_TYPE=Release"
# ARGS="$ARGS -DCMAKE_BUILD_TYPE=Debug"

# Install prefix (default: /usr/local)
# ARGS="$ARGS -DCMAKE_INSTALL_PREFIX=$HOME/local"

#==============================================================================
# Language Bindings (Front-ends)
#==============================================================================
# Python bindings
# ARGS="$ARGS -DWITH_PYTHON=ON"
# ARGS="$ARGS -DWITH_PYTHON3=ON"
# ARGS="$ARGS -DWITH_PYTHON_GIL_RELEASE=ON"

# MATLAB bindings
# ARGS="$ARGS -DWITH_MATLAB=ON"

# Octave bindings (experimental)
# ARGS="$ARGS -DWITH_OCTAVE=ON"

# JSON front-end
# ARGS="$ARGS -DWITH_JSON=ON"

#==============================================================================
# Parallelization
#==============================================================================
# OpenMP support
# ARGS="$ARGS -DWITH_OPENMP=ON"

# POSIX Threads support
# ARGS="$ARGS -DWITH_THREAD=ON"

# Thread-safe symbolics (may impact performance)
# ARGS="$ARGS -DWITH_THREADSAFE_SYMBOLICS=ON"

# OpenCL support (experimental)
# ARGS="$ARGS -DWITH_OPENCL=ON"

#==============================================================================
# NLP Solvers (Nonlinear Programming)
#==============================================================================
# IPOPT - Interior Point Optimizer (requires coinor-libipopt-dev)
ARGS="$ARGS -DWITH_IPOPT=ON"
# ARGS="$ARGS -DWITH_BUILD_IPOPT=ON"

# BONMIN - Basic Open-source Nonlinear Mixed INteger programming
# ARGS="$ARGS -DWITH_BONMIN=ON"
# ARGS="$ARGS -DWITH_BUILD_BONMIN=ON"

# KNITRO - commercial solver
# ARGS="$ARGS -DWITH_KNITRO=ON"

# SNOPT - commercial solver
# ARGS="$ARGS -DWITH_SNOPT=ON"

# WORHP - commercial solver
# ARGS="$ARGS -DWITH_WORHP=ON"

# MadNLP - Nonlinear solver in Julia
# ARGS="$ARGS -DWITH_MADNLP=ON"

# SLEQP - Sequential Linearization Equality-constrained QP
# ARGS="$ARGS -DWITH_SLEQP=ON"
# ARGS="$ARGS -DWITH_BUILD_SLEQP=ON"

# Alpaqa - proximal gradient-based solver
# ARGS="$ARGS -DWITH_ALPAQA=ON"
# ARGS="$ARGS -DWITH_BUILD_ALPAQA=ON"

#==============================================================================
# QP Solvers (Quadratic Programming)
#==============================================================================
# qpOASES - online active set strategy QP solver (included)
# ARGS="$ARGS -DWITH_QPOASES=ON"
# ARGS="$ARGS -DWITH_NO_QPOASES_BANNER=ON"

# OSQP - Operator Splitting QP solver
# ARGS="$ARGS -DWITH_OSQP=ON"
# ARGS="$ARGS -DWITH_BUILD_OSQP=ON"

# PROXQP - Proximal QP solver
# ARGS="$ARGS -DWITH_PROXQP=ON"
# ARGS="$ARGS -DWITH_BUILD_PROXQP=ON"

# SuperSCS - Splitting Conic Solver (included)
# ARGS="$ARGS -DWITH_SUPERSCS=ON"
# ARGS="$ARGS -DWITH_BUILD_SUPERSCS=ON"

# DAQP - Dual Active-set QP solver
# ARGS="$ARGS -DWITH_DAQP=ON"
# ARGS="$ARGS -DWITH_BUILD_DAQP=ON"

# Clarabel - Interior point conic solver
# ARGS="$ARGS -DWITH_CLARABEL=ON"
# ARGS="$ARGS -DWITH_BUILD_CLARABEL=ON"

# blockSQP - Block-structured SQP solver (included)
# ARGS="$ARGS -DWITH_BLOCKSQP=ON"

#==============================================================================
# LP Solvers (Linear Programming)
#==============================================================================
# HiGHS - High performance LP/MIP solver
# ARGS="$ARGS -DWITH_HIGHS=ON"
# ARGS="$ARGS -DWITH_BUILD_HIGHS=ON"

# CLP - COIN-OR LP solver
# ARGS="$ARGS -DWITH_CLP=ON"
# ARGS="$ARGS -DWITH_BUILD_CLP=ON"

# CBC - COIN-OR Branch and Cut (MIP solver)
# ARGS="$ARGS -DWITH_CBC=ON"
# ARGS="$ARGS -DWITH_BUILD_CBC=ON"

# CPLEX - IBM commercial solver
# ARGS="$ARGS -DWITH_CPLEX=ON"

# GUROBI - commercial solver
# ARGS="$ARGS -DWITH_GUROBI=ON"

# DSDP - Semidefinite programming solver
# ARGS="$ARGS -DWITH_DSDP=ON"
# ARGS="$ARGS -DWITH_BUILD_DSDP=ON"

#==============================================================================
# MPC-related Solvers (Model Predictive Control)
#==============================================================================
# FATROP - Fast Trajectory Optimization (for OCP)
ARGS="$ARGS -DWITH_FATROP=ON"
ARGS="$ARGS -DWITH_BUILD_FATROP=ON"

# HPIPM - High-Performance Interior Point Method
# ARGS="$ARGS -DWITH_HPIPM=ON"
# ARGS="$ARGS -DWITH_BUILD_HPIPM=ON"

#==============================================================================
# Linear Algebra Libraries
#==============================================================================
# LAPACK
ARGS="$ARGS -DWITH_LAPACK=ON"
# ARGS="$ARGS -DWITH_BUILD_LAPACK=ON"

# BLASFEO - Basic Linear Algebra Subroutines For Embedded Optimization
ARGS="$ARGS -DWITH_BLASFEO=ON"
ARGS="$ARGS -DWITH_BUILD_BLASFEO=ON"

# CSparse - Concise Sparse matrix package (included, default ON)
# ARGS="$ARGS -DWITH_CSPARSE=ON"
# ARGS="$ARGS -DWITH_BUILD_CSPARSE=ON"

# Sundials - CVODES/IDAS integrators (included, default ON)
# Not needed for MPC code generation, disable to reduce build time
ARGS="$ARGS -DWITH_SUNDIALS=OFF"

# MUMPS - MUltifrontal Massively Parallel Sparse direct Solver
# ARGS="$ARGS -DWITH_MUMPS=ON"
# ARGS="$ARGS -DWITH_BUILD_MUMPS=ON"

# HSL - Harwell Subroutine Library (requires license)
# ARGS="$ARGS -DWITH_HSL=ON"

# SPRAL - Sparse Parallel Robust Algorithms Library
# ARGS="$ARGS -DWITH_SPRAL=ON"
# ARGS="$ARGS -DWITH_BUILD_SPRAL=ON"

# Metis - Graph partitioning (used by some solvers)
# ARGS="$ARGS -DWITH_BUILD_METIS=ON"

#==============================================================================
# Code Generation & Interfaces
#==============================================================================
# Dynamic loading of functions (default ON)
# ARGS="$ARGS -DWITH_DL=ON"

# FMI 2.0 binary import support (default ON)
# Not needed for MPC, disable to reduce build time
ARGS="$ARGS -DWITH_FMI2=OFF"

# FMI 3.0 binary import support (default ON)
ARGS="$ARGS -DWITH_FMI3=OFF"

# TinyXML (included, default ON)
# ARGS="$ARGS -DWITH_TINYXML=ON"
# ARGS="$ARGS -DWITH_BUILD_TINYXML=ON"

# Clang JIT compilation
# ARGS="$ARGS -DWITH_CLANG=ON"

# AMPL interface
# ARGS="$ARGS -DWITH_AMPL=ON"

# MATLAB IPC interface
# ARGS="$ARGS -DWITH_MATLAB_IPC=ON"

# Rumoca - Modelica compiler
# ARGS="$ARGS -DWITH_RUMOCA=ON"
# ARGS="$ARGS -DWITH_BUILD_RUMOCA=ON"

#==============================================================================
# Additional Libraries
#==============================================================================
# Eigen3 - Linear algebra library
# ARGS="$ARGS -DWITH_BUILD_EIGEN3=ON"

# SLICOT - Control and systems library
# ARGS="$ARGS -DWITH_SLICOT=ON"

# ZLIB - Compression library
# ARGS="$ARGS -DWITH_ZLIB=ON"
# ARGS="$ARGS -DWITH_BUILD_ZLIB=ON"

# LIBZIP - ZIP archive library
# ARGS="$ARGS -DWITH_LIBZIP=ON"
# ARGS="$ARGS -DWITH_BUILD_LIBZIP=ON"

# GHC Filesystem - std::filesystem alternative
# ARGS="$ARGS -DWITH_GHC_FILESYSTEM=ON"
# ARGS="$ARGS -DWITH_BUILD_GHC_FILESYSTEM=ON"

#==============================================================================
# Build Options
#==============================================================================
# Self-contained install directory
# ARGS="$ARGS -DWITH_SELFCONTAINED=ON"

# SO version for shared library (default ON)
# ARGS="$ARGS -DWITH_SO_VERSION=ON"

# Build examples (default ON)
ARGS="$ARGS -DWITH_EXAMPLES=OFF"

# DEEPBIND for plugin loading (default ON, useful for MATLAB)
# Not needed without MATLAB
ARGS="$ARGS -DWITH_DEEPBIND=OFF"

# Deprecated features support (default ON)
# Not needed for new development
ARGS="$ARGS -DWITH_DEPRECATED_FEATURES=OFF"

#==============================================================================
# Development & Debugging
#==============================================================================
# Extra warnings (-Wall -Wextra)
# ARGS="$ARGS -DWITH_EXTRA_WARNINGS=ON"

# Treat warnings as errors (-Werror)
# ARGS="$ARGS -DWITH_WERROR=ON"

# Extra runtime checks (for developers)
# ARGS="$ARGS -DWITH_EXTRA_CHECKS=ON"

# Coverage report
# ARGS="$ARGS -DWITH_COVERAGE=ON"

# Reference counting warnings
# ARGS="$ARGS -DWITH_REFCOUNT_WARNINGS=ON"

# Linting support
# ARGS="$ARGS -DWITH_LINT=ON"

# Spell-checking support
# ARGS="$ARGS -DWITH_SPELL=ON"

# clang-tidy support
# ARGS="$ARGS -DWITH_CLANG_TIDY=ON"

# Documentation generation
# ARGS="$ARGS -DWITH_DOC=ON"

#==============================================================================
# Build and Install
#==============================================================================
cmake .. $ARGS
make -j$(nproc)
sudo make install
