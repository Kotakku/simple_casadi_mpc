#include "double_integrator_problem.hpp"
#include <casadi/casadi.hpp>
#include <iostream>
#include <matplotlibcpp17/pyplot.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

using namespace pybind11::literals;

int main() {
  using namespace simple_casadi_mpc;

  std::cout << "double integrator mpc example" << std::endl;
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();
  auto prob = std::make_shared<DoubleIntegratorProb>();
  MPC mpc(prob);

  Eigen::VectorXd x = Eigen::VectorXd::Constant(prob->nx(), 1.0);

  std::cout << "simulation" << std::endl;
  const double dt = 0.01;
  const size_t sim_len = 1000;
  std::vector<double> t_log(sim_len), x_log(sim_len), v_log(sim_len), u_log(sim_len);
  for (size_t i = 0; i < sim_len; i++) {
    Eigen::VectorXd u = mpc.solve(x);
    x = prob->simulate(x, u, dt);
    // std::cout << u << ", " << x.transpose() << std::endl;
    t_log[i] = i * dt;
    x_log[i] = x[0];
    v_log[i] = x[1];
    u_log[i] = u[0];
  }

  // plot
  plt.figure();
  plt.plot(pybind11::make_tuple(t_log, u_log), pybind11::dict("label"_a = "u"));
  plt.plot(pybind11::make_tuple(t_log, x_log), pybind11::dict("label"_a = "pos"));
  plt.plot(pybind11::make_tuple(t_log, v_log), pybind11::dict("label"_a = "vel"));
  plt.legend();
  plt.show();

  return 0;
}
