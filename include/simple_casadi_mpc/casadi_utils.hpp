#pragma once

#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <memory>
#include <vector>

namespace casadi_utils {

static Eigen::MatrixXd to_eigen(const casadi::DM &cas_mat) {
  Eigen::MatrixXd eig_mat(cas_mat.size1(), cas_mat.size2());
  for (casadi_int i = 0; i < cas_mat.size1(); ++i) {
    for (casadi_int j = 0; j < cas_mat.size2(); ++j) {
      eig_mat(i, j) = static_cast<double>(cas_mat(i, j));
    }
  }
  return eig_mat;
}

static casadi::DM to_casadi(const Eigen::MatrixXd &eig_mat) {
  casadi::DM cas_mat = casadi::DM::zeros(eig_mat.rows(), eig_mat.cols());
  for (int i = 0; i < eig_mat.rows(); ++i) {
    for (int j = 0; j < eig_mat.cols(); ++j) {
      cas_mat(i, j) = eig_mat(i, j);
    }
  }
  return cas_mat;
}

template <class Sym = casadi::SX>
static Eigen::Matrix<Sym, Eigen::Dynamic, 1> pack_vector(const Sym &sym) {
  Eigen::Matrix<Sym, Eigen::Dynamic, 1> vec(sym.size1());
  for (int i = 0; i < sym.size1(); ++i) {
    vec(i) = sym(i);
  }
  return vec;
}

template <class Sym = casadi::SX>
static Sym unpack_vector(const Eigen::Matrix<Sym, Eigen::Dynamic, 1> &vec) {
  Sym sym = Sym::zeros(vec.size());
  for (int i = 0; i < vec.size(); ++i) {
    sym(i) = vec(i);
  }
  return sym;
}

} // namespace casadi_utils
