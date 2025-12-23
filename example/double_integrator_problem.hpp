#pragma once

#include "simple_casadi_mpc/simple_casadi_mpc.hpp"
#include <casadi/casadi.hpp>

class DoubleIntegratorProb : public simple_casadi_mpc::Problem<casadi::MX> {
public:
  DoubleIntegratorProb() : Problem(DynamicsType::ContinuesRK4, 2, 1, 20, 0.05) {
    set_input_bound(Eigen::VectorXd::Constant(1, -1.0), Eigen::VectorXd::Constant(1, 1.0));
  }

  casadi::MX dynamics(casadi::MX x, casadi::MX u) override {
    return casadi::MX::vertcat({x(1), u});
  }

  casadi::MX stage_cost(casadi::MX x, casadi::MX u, size_t k) override {
    (void)u;
    (void)k;
    return casadi::MX::mtimes(x.T(), x);
  }

  casadi::MX terminal_cost(casadi::MX x) override { return casadi::MX::mtimes(x.T(), x); }
};
