#include "simple_casadi_mpc/simple_casadi_mpc.hpp"
#include <casadi/casadi.hpp>
#include <chrono>
#include <iostream>
#include <matplotlibcpp17/pyplot.h>
#include <numeric>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

using namespace pybind11::literals;

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

void animate(matplotlibcpp17::pyplot::PyPlot &plt, const std::vector<double> &x,
             const std::vector<double> &angle, const std::vector<double> &u,
             const std::vector<double> &target, size_t skip = 4) {
  (void)u;

  double cart_width = 0.5;
  double cart_height = 0.2;
  double pole_length = 0.5;

  double x_max = *std::max_element(x.begin(), x.end());
  double x_min = *std::min_element(x.begin(), x.end());
  double range_max = std::max(std::abs(x_max), std::abs(x_min)) + cart_width;
  for (size_t i = 0; i < x.size(); i += skip) {
    plt.clf();

    double cart_x = x[i];
    double cart_y = 0.0;
    double pole_x = cart_x + pole_length * sin(angle[i]);
    double pole_y = cart_y - pole_length * cos(angle[i]);

    std::vector<double> cart_x_data = {cart_x - cart_width / 2, cart_x + cart_width / 2,
                                       cart_x + cart_width / 2, cart_x - cart_width / 2,
                                       cart_x - cart_width / 2};
    std::vector<double> cart_y_data = {cart_y - cart_height / 2, cart_y - cart_height / 2,
                                       cart_y + cart_height / 2, cart_y + cart_height / 2,
                                       cart_y - cart_height / 2};
    std::vector<double> pole_x_data = {cart_x, pole_x};
    std::vector<double> pole_y_data = {cart_y, pole_y};
    std::vector<double> target_x_data = {target[i], target[i]};
    std::vector<double> target_y_data = {-range_max, range_max};

    plt.gca().set_aspect(pybind11::make_tuple(1.0));
    plt.plot(pybind11::make_tuple(target_x_data, target_y_data, "k--"));
    plt.plot(pybind11::make_tuple(cart_x_data, cart_y_data, "b-"));
    plt.plot(pybind11::make_tuple(pole_x_data, pole_y_data, "r-"));

    plt.xlim(pybind11::make_tuple(-range_max, range_max));
    plt.ylim(pybind11::make_tuple(-range_max, range_max));
    plt.pause(pybind11::make_tuple(0.01));
    // std::cout << i+1 << "/" << x.size() << std::endl;
  }
}

int main() {
  using namespace simple_casadi_mpc;
  std::cout << "Cartpole MPC vs JITMPC Comparison" << std::endl;
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();
  auto prob = std::make_shared<CartpoleProb>();

  // Create both regular MPC and JITMPC
  // Use FATROP for better performance
  auto fatrop_config = default_fatrop_config();
  fatrop_config["fatrop.tol"] = 1e-2;
  fatrop_config["fatrop.acceptable_tol"] = 5e-2;
  fatrop_config["print_time"] = false;

  std::cout << "\n=== Creating Regular MPC ===" << std::endl;
  MPC<casadi::MX> mpc_regular(prob, "fatrop", fatrop_config);

  std::cout << "\n=== Creating JITMPC ===" << std::endl;
  JITMPC<casadi::MX> mpc_codegen("cartpole", prob, "fatrop", fatrop_config);

  casadi::DMDict param_list;
  double target_pos = -0.5;
  param_list["x_ref"] = {target_pos, M_PI, 0, 0};

  // Simulation for both MPCs
  const double dt = 0.01;
  const size_t sim_len = 600; // Shorter simulation for comparison

  std::cout << "\n=== Running Regular MPC ===" << std::endl;
  Eigen::VectorXd x_regular = Eigen::VectorXd::Zero(prob->nx());
  std::vector<double> i_log(sim_len), target_log(sim_len), t_log(sim_len);
  std::vector<double> x_log_regular(sim_len), angle_log_regular(sim_len), u_log_regular(sim_len);
  std::vector<double> dt_log_regular(sim_len);

  auto t_all_start = std::chrono::system_clock::now();
  for (size_t i = 0; i < sim_len; i++) {
    if (i == 300) {
      target_pos = 0.5;
      param_list["x_ref"] = {target_pos, M_PI, 0, 0};
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd u = mpc_regular.solve(x_regular, param_list);
    auto t_end = std::chrono::high_resolution_clock::now();
    x_regular = prob->simulate(x_regular, u, dt);
    double solve_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() * 1e-6;

    target_log[i] = target_pos;
    t_log[i] = i * dt;
    i_log[i] = i;
    x_log_regular[i] = x_regular[0];
    angle_log_regular[i] = x_regular[1];
    u_log_regular[i] = u[0];
    dt_log_regular[i] = solve_time * 1e3;
  }
  auto t_all_end = std::chrono::system_clock::now();
  double regular_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_all_end - t_all_start).count() * 1e-6;
  std::cout << "Regular MPC total time: " << regular_time << " sec" << std::endl;
  std::cout << "Regular MPC average: " << regular_time / sim_len * 1000 << " ms/iter" << std::endl;

  std::cout << "\n=== Running JITMPC ===" << std::endl;
  std::cout << "First solve will generate and compile code..." << std::endl;

  target_pos = -0.5; // Reset target
  param_list["x_ref"] = {target_pos, M_PI, 0, 0};
  Eigen::VectorXd x_codegen = Eigen::VectorXd::Zero(prob->nx());
  std::vector<double> x_log_codegen(sim_len), angle_log_codegen(sim_len), u_log_codegen(sim_len);
  std::vector<double> dt_log_codegen(sim_len);

  t_all_start = std::chrono::system_clock::now();
  for (size_t i = 0; i < sim_len; i++) {
    if (i == 300) {
      target_pos = 0.5;
      param_list["x_ref"] = {target_pos, M_PI, 0, 0};
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd u = mpc_codegen.solve(x_codegen, param_list);
    auto t_end = std::chrono::high_resolution_clock::now();
    x_codegen = prob->simulate(x_codegen, u, dt);
    double solve_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() * 1e-6;

    if (i == 0) {
      std::cout << "First solve (with code generation): " << solve_time * 1000 << " ms"
                << std::endl;
    } else if (i == 1) {
      std::cout << "Second solve (using generated code): " << solve_time * 1000 << " ms"
                << std::endl;
    }

    x_log_codegen[i] = x_codegen[0];
    angle_log_codegen[i] = x_codegen[1];
    u_log_codegen[i] = u[0];
    dt_log_codegen[i] = solve_time * 1e3;
  }
  t_all_end = std::chrono::system_clock::now();
  double codegen_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_all_end - t_all_start).count() * 1e-6;
  std::cout << "JITMPC total time: " << codegen_time << " sec" << std::endl;
  std::cout << "JITMPC average: " << codegen_time / sim_len * 1000 << " ms/iter" << std::endl;
  std::cout << "\nSpeedup factor: " << regular_time / codegen_time << "x" << std::endl;

  // Plot state trajectories
  auto fig = plt.figure();
  plt.subplot(311);
  plt.plot(pybind11::make_tuple(t_log, x_log_regular), pybind11::dict("label"_a = "Regular MPC"));
  plt.plot(pybind11::make_tuple(t_log, x_log_codegen), pybind11::dict("label"_a = "JITMPC"));
  plt.ylabel(pybind11::make_tuple("Position [m]"));
  plt.legend();
  plt.grid(pybind11::make_tuple(true));

  plt.subplot(312);
  plt.plot(pybind11::make_tuple(t_log, angle_log_regular),
           pybind11::dict("label"_a = "Regular MPC"));
  plt.plot(pybind11::make_tuple(t_log, angle_log_codegen), pybind11::dict("label"_a = "JITMPC"));
  plt.ylabel(pybind11::make_tuple("Angle [rad]"));
  plt.legend();
  plt.grid(pybind11::make_tuple(true));

  plt.subplot(313);
  plt.plot(pybind11::make_tuple(t_log, u_log_regular), pybind11::dict("label"_a = "Regular MPC"));
  plt.plot(pybind11::make_tuple(t_log, u_log_codegen), pybind11::dict("label"_a = "JITMPC"));
  plt.xlabel(pybind11::make_tuple("Time [s]"));
  plt.ylabel(pybind11::make_tuple("Control [N]"));
  plt.legend();
  plt.grid(pybind11::make_tuple(true));

  fig.suptitle(pybind11::make_tuple("State Trajectories Comparison"));
  plt.show();

  // Plot timing comparison
  plt.figure();
  plt.plot(pybind11::make_tuple(i_log, dt_log_regular, "b-"), pybind11::dict("label"_a = "MPC"));
  plt.plot(pybind11::make_tuple(i_log, dt_log_codegen, "r-"),
           pybind11::dict("label"_a = "MPC (with CodeGen)"));
  plt.xlabel(pybind11::make_tuple("Iteration"));
  plt.ylabel(pybind11::make_tuple("Solve time [ms]"));
  plt.title(pybind11::make_tuple("MPC Performance Comparison"));
  plt.legend();
  plt.ylim(pybind11::make_tuple(
      0.0, *std::max_element(dt_log_regular.begin(), dt_log_regular.end()) * 1.1));
  plt.grid(pybind11::make_tuple(true));

  // Add average lines
  double avg_regular =
      std::accumulate(dt_log_regular.begin() + 10, dt_log_regular.end(), 0.0) / (sim_len - 10);
  double avg_codegen =
      std::accumulate(dt_log_codegen.begin() + 10, dt_log_codegen.end(), 0.0) / (sim_len - 10);
  std::vector<double> avg_regular_line(sim_len, avg_regular);
  std::vector<double> avg_codegen_line(sim_len, avg_codegen);

  plt.plot(pybind11::make_tuple(i_log, avg_regular_line),
           pybind11::dict("label"_a = "Avg Regular"));
  plt.plot(pybind11::make_tuple(i_log, avg_codegen_line),
           pybind11::dict("label"_a = "Avg CodeGen"));

  auto perf_ax = plt.gca();

  // Add text annotations
  std::string text_regular = "Avg Regular: " + std::to_string(avg_regular).substr(0, 5) + " ms";
  std::string text_codegen = "Avg CodeGen: " + std::to_string(avg_codegen).substr(0, 5) + " ms";
  std::string text_speedup =
      "Speedup: " + std::to_string(avg_regular / avg_codegen).substr(0, 4) + "x";

  perf_ax.text(pybind11::make_tuple(sim_len * 0.6, avg_regular * 1.1, text_regular));
  perf_ax.text(pybind11::make_tuple(sim_len * 0.6, avg_codegen * 1.1, text_codegen));
  perf_ax.text(pybind11::make_tuple(sim_len * 0.6, avg_regular + (avg_regular + avg_codegen) / 2,
                                    text_speedup));

  plt.show();

  // Animate only JITMPC result
  animate(plt, x_log_codegen, angle_log_codegen, u_log_codegen, target_log);

  return 0;
}
