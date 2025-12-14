#pragma once

#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <memory>
#include <vector>

namespace casadi_utils {

inline Eigen::MatrixXd to_eigen(const casadi::DM &dm) {
  Eigen::MatrixXd mat(dm.size1(), dm.size2());
  std::memcpy(mat.data(), dm.ptr(), sizeof(double) * dm.size1() * dm.size2());
  return mat;
}

inline casadi::DM to_casadi(const Eigen::MatrixXd &mat) {
  casadi::DM dm = casadi::DM::zeros(mat.rows(), mat.cols());
  std::memcpy(dm.ptr(), mat.data(), sizeof(double) * mat.rows() * mat.cols());
  return dm;
}

} // namespace casadi_utils