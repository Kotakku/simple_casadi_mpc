#include "simple_casadi_mpc/simple_casadi_mpc.hpp"
#include <casadi/casadi.hpp>
#include <chrono>
#include <iostream>
#include <matplotlibcpp17/pyplot.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

using namespace pybind11::literals;

class InvertedPendulumProb : public simple_casadi_mpc::Problem<casadi::MX> {
public:
  InvertedPendulumProb() : Problem(DynamicsType::ContinuesRK4, 2, 1, 60, 0.05) {
    using namespace casadi;
    x_ref = {M_PI, 0};
    Q = DM::diag({5.0, 0.01});
    R = DM::diag({0.01});
    Qf = DM::diag({5.0, 0.1});

    set_input_bound(Eigen::VectorXd::Constant(1, -2.0), Eigen::VectorXd::Constant(1, 2.0));
  }

  virtual casadi::MX dynamics(casadi::MX x, casadi::MX u) override {
    using namespace casadi;

    auto th = x(0);
    auto dth = x(1);
    return vertcat(dth, (u(0) - m * g * sin(th)) / (m * l));
  }

  virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u, size_t k) override {
    (void)k;
    using namespace casadi;
    MX L = 0;
    auto e = x - MX(x_ref);
    L += 0.5 * mtimes(e.T(), mtimes(MX(Q), e));
    L += 0.5 * mtimes(u.T(), mtimes(MX(R), u));
    return dt() * L;
  }

  virtual casadi::MX terminal_cost(casadi::MX x) override {
    using namespace casadi;
    auto e = x - MX(x_ref);
    return 0.5 * mtimes(e.T(), mtimes(MX(Qf), e));
  }

  const double l = 0.3; // [m]
  const double m = 0.6; // [kg]
  const double g = 9.8; // [m/s^2]

  casadi::DM x_ref;
  casadi::DM Q, R, Qf;
};

void animate(matplotlibcpp17::pyplot::PyPlot &plt, const std::vector<double> &angle,
             const std::vector<double> &u) {
  (void)u;
  for (size_t i = 0; i < angle.size(); i += 3) {
    plt.clf();

    double pole_length = 0.5;

    const double pole_start_x = 0.0;
    const double pole_start_y = 0.0;
    const double pole_end_x = pole_start_x + pole_length * sin(angle[i]);
    const double pole_end_y = pole_start_y - pole_length * cos(angle[i]);

    std::vector<double> pole_x_data = {pole_start_x, pole_end_x};
    std::vector<double> pole_y_data = {pole_start_y, pole_end_y};

    plt.gca().set_aspect(pybind11::make_tuple(1.0));
    plt.plot(pybind11::make_tuple(pole_x_data, pole_y_data, "r-"));
    const double range_max = pole_length + 0.5;
    plt.xlim(pybind11::make_tuple(-range_max, range_max));
    plt.ylim(pybind11::make_tuple(-range_max, range_max));
    plt.pause(pybind11::make_tuple(0.01));
    // std::cout << i+1 << "/" << angle.size() << std::endl;
  }
}

int main() {
  using namespace simple_casadi_mpc;
  std::cout << "inverted pendulum mpc example" << std::endl;
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();
  auto prob = std::make_shared<InvertedPendulumProb>();
  MPC<casadi::MX> mpc(prob);

  Eigen::VectorXd x = Eigen::VectorXd::Zero(prob->nx());

  std::cout << "simulation" << x << std::endl;
  const double dt = 0.01;
  const size_t sim_len = 400;
  std::vector<double> i_log(sim_len), t_log(sim_len), angle_log(sim_len), u_log(sim_len),
      dt_log(sim_len);

  auto t_all_start = std::chrono::system_clock::now();
  for (size_t i = 0; i < sim_len; i++) {
    auto t_start = std::chrono::system_clock::now();
    // MPCで最適入力を計算
    Eigen::VectorXd u = mpc.solve(x);
    auto t_end = std::chrono::system_clock::now();
    double solve_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() * 1e-6;
    // std::cout << "solve time: " << solve_time << std::endl;

    // シミュレーション
    x = prob->simulate(x, u, dt);
    i_log[i] = i;
    t_log[i] = i * dt;
    angle_log[i] = x[0];
    u_log[i] = u[0];
    dt_log[i] = solve_time * 1e3;
  }
  auto t_all_end = std::chrono::system_clock::now();
  double all_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_all_end - t_all_start).count() * 1e-6;
  std::cout << "all time: " << all_time << std::endl;

  plt.figure();
  plt.plot(pybind11::make_tuple(t_log, u_log), pybind11::dict("label"_a = "u"));
  plt.plot(pybind11::make_tuple(t_log, angle_log), pybind11::dict("label"_a = "angle"));
  plt.legend();
  plt.show();

  plt.figure();
  plt.plot(pybind11::make_tuple(i_log, dt_log));
  plt.xlabel(pybind11::make_tuple("iteration"));
  plt.ylabel(pybind11::make_tuple("MPC solve time [ms]"));
  plt.show();

  animate(plt, angle_log, u_log);

  return 0;
}
