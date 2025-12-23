#pragma once

#include "simple_casadi_mpc/simple_casadi_mpc.hpp"

class CartpoleProb : public simple_casadi_mpc::Problem<casadi::MX> {
public:
  CartpoleProb() : Problem(DynamicsType::ContinuesRK4, 4, 1, 30, 0.05) {
    using namespace casadi;
    x_ref = parameter("x_ref", 4, 1);
    Q = DM::diag({5, 10.0, 0.01, 0.01});
    R = DM::diag({0.1});
    Qf = DM::diag({10, 10.0, 0.01, 0.01});

    set_input_bound(Eigen::VectorXd::Constant(1, -15.0), Eigen::VectorXd::Constant(1, 15.0));
  }

  virtual casadi::MX dynamics(casadi::MX x, casadi::MX u) override {
    using namespace casadi;

    // auto y = x(0);     // cart の水平位置[m]
    auto th = x(1);  // pole の傾き角[rad]
    auto dy = x(2);  // cart の水平速度[m/s]
    auto dth = x(3); // pole の傾き角速度[rad/s]
    auto f = u(0);   // cart を押す力[N]（制御入力）
    // cart の水平加速度
    casadi::MX ddy =
        (f + mp * sin(th) * (l * dth * dth + g * cos(th))) / (mc + mp * sin(th) * sin(th));
    // pole の傾き角加速度
    casadi::MX ddth =
        (-f * cos(th) - mp * l * dth * dth * cos(th) * sin(th) - (mc + mp) * g * sin(th)) /
        (l * (mc + mp * sin(th) * sin(th)));

    return vertcat(dy, dth, ddy, ddth);
  }

  virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u, size_t k) override {
    (void)k;
    using namespace casadi;
    MX L = 0;
    auto e = x - x_ref;
    L += 0.5 * mtimes(e.T(), mtimes(MX(Q), e));
    L += 0.5 * mtimes(u.T(), mtimes(MX(R), u));
    return dt() * L;
  }

  virtual casadi::MX terminal_cost(casadi::MX x) override {
    using namespace casadi;
    auto e = x - x_ref;
    return 0.5 * mtimes(e.T(), mtimes(MX(Qf), e));
  }

  const double mc = 2.0;
  const double mp = 0.2;
  const double l = 0.5;
  const double g = 9.8;

  casadi::MX x_ref;
  casadi::DM Q, R, Qf;
};
