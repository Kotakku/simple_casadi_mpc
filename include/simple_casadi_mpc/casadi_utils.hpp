#pragma once

#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <memory>
#include <vector>

/// Small adapters between CasADi DM matrices and Eigen dense matrices.
namespace casadi_utils {

/// @brief Convert a CasADi DM (column-major) into an Eigen MatrixXd.
/// @param dm CasADi DM with the same shape as the returned matrix.
/// @return Eigen MatrixXd holding a deep copy of `dm`.
inline Eigen::MatrixXd to_eigen(const casadi::DM &dm) {
  Eigen::MatrixXd mat(dm.size1(), dm.size2());
  std::memcpy(mat.data(), dm.ptr(), sizeof(double) * dm.size1() * dm.size2());
  return mat;
}

/// @brief Convert an Eigen MatrixXd into a CasADi DM.
/// @param mat Eigen matrix to copy.
/// @return CasADi DM holding a deep copy of `mat`.
inline casadi::DM to_casadi(const Eigen::MatrixXd &mat) {
  casadi::DM dm = casadi::DM::zeros(mat.rows(), mat.cols());
  std::memcpy(dm.ptr(), mat.data(), sizeof(double) * mat.rows() * mat.cols());
  return dm;
}

} // namespace casadi_utils
