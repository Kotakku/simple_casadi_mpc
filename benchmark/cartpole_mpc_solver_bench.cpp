#include "cartpole_problem.hpp"
#include "cartpole_solver_bench_config.hpp"
#include "simple_casadi_mpc/simple_casadi_mpc.hpp"
#include <casadi/casadi.hpp>
#include <chrono>
#include <iostream>
#include <matplotlibcpp17/animation.h>
#include <matplotlibcpp17/patches.h>
#include <matplotlibcpp17/pyplot.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include <pybind11/embed.h>

using namespace pybind11::literals;

struct TestResult {
  std::vector<double> i_log;
  std::vector<double> target_log;
  std::vector<double> t_log;
  std::vector<double> x_log;
  std::vector<double> angle_log;
  std::vector<double> u_log;
  std::vector<double> solve_time_log;

  double all_time;
};

template <class T> TestResult run_simulation(T &mpc, std::shared_ptr<CartpoleProb> prob) {
  casadi::DMDict param_list;
  double target_pos = -0.5;
  param_list["x_ref"] = {target_pos, M_PI, 0, 0};

  Eigen::VectorXd x = Eigen::VectorXd::Zero(prob->nx());

  std::cout << "simulation" << std::endl;
  const double dt = 0.01;
  const size_t sim_len = 1200;

  TestResult result;
  result.i_log.resize(sim_len);
  result.target_log.resize(sim_len);
  result.t_log.resize(sim_len);
  result.x_log.resize(sim_len);
  result.angle_log.resize(sim_len);
  result.u_log.resize(sim_len);
  result.solve_time_log.resize(sim_len);

  auto t_all_start = std::chrono::system_clock::now();
  for (size_t i = 0; i < sim_len; i++) {
    if (i == 700) {
      // change target
      target_pos = 0.5;
      param_list["x_ref"] = {target_pos, M_PI, 0, 0};
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd u = mpc.solve(x, param_list);
    auto t_end = std::chrono::high_resolution_clock::now();
    x = prob->simulate(x, u, dt);
    const double solve_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() * 1e-6;
    result.target_log[i] = target_pos;
    result.t_log[i] = i * dt;
    result.i_log[i] = i;
    result.x_log[i] = x[0];
    result.angle_log[i] = x[1];
    result.u_log[i] = u[0];
    result.solve_time_log[i] = solve_time * 1e3;
  }
  auto t_all_end = std::chrono::system_clock::now();
  result.all_time =
      std::chrono::duration_cast<std::chrono::microseconds>(t_all_end - t_all_start).count() * 1e-6;
  return result;
}

int main() {
  using namespace simple_casadi_mpc;
  std::cout << "cartpole mpc benchmark" << std::endl;

  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();

  auto prob = std::make_shared<CartpoleProb>();

  // MPC (IPOPT)
  const auto mpc_result = [&]() {
    MPC mpc(prob);
    return run_simulation(mpc, prob);
  }();

  // MPC (FATROP)
  const auto mpc_fatrop_result = [&]() {
    auto solver_config = MPC::default_fatrop_config();
    MPC mpc(prob, "fatrop", solver_config);
    return run_simulation(mpc, prob);
  }();

  // JIT MPC (FATROP)
  const auto jit_mpc_result = [&]() {
    auto solver_config = MPC::default_fatrop_config();
    JITMPC mpc("cartpole_jit_mpc", prob, "fatrop", solver_config);
    return run_simulation(mpc, prob);
  }();

  // Compiled MPC (FATROP)
  const auto compiled_mpc_retult = [&]() {
    auto lib_config = get_cartpole_solver_bench_compiled_library_options();
    CompiledMPC mpc(lib_config, prob);
    return run_simulation(mpc, prob);
  }();

  auto trajectory_plot = [&](const TestResult &result, const std::string file_name) {
    plt.figure();
    plt.plot(pybind11::make_tuple(result.t_log, result.u_log), pybind11::dict("label"_a = "u"));
    plt.plot(pybind11::make_tuple(result.t_log, result.x_log), pybind11::dict("label"_a = "x"));
    plt.plot(pybind11::make_tuple(result.t_log, result.angle_log),
             pybind11::dict("label"_a = "angle"));
    plt.legend();
    plt.savefig(Args(file_name));
  };

  trajectory_plot(mpc_result, "bench_cartpole_mpc_trajectory.png");
  trajectory_plot(mpc_fatrop_result, "bench_cartpole_mpc_fatrop_trajectory.png");
  trajectory_plot(jit_mpc_result, "bench_cartpole_jit_mpc_trajectory.png");
  trajectory_plot(compiled_mpc_retult, "bench_cartpole_compiled_mpc_trajectory.png");

  const auto solve_time_str = [](const TestResult &result) {
    return "( " + std::to_string(static_cast<int>(std::round(result.all_time * 1e3))) + " ms)";
  };

  // compare solve time
  plt.figure();
  plt.plot(pybind11::make_tuple(mpc_result.i_log, mpc_result.solve_time_log),
           pybind11::dict("label"_a = "MPC (IPOPT) " + solve_time_str(mpc_result)));
  plt.plot(pybind11::make_tuple(mpc_fatrop_result.i_log, mpc_fatrop_result.solve_time_log),
           pybind11::dict("label"_a = "MPC (FATROP) " + solve_time_str(mpc_fatrop_result)));
  plt.plot(pybind11::make_tuple(jit_mpc_result.i_log, jit_mpc_result.solve_time_log),
           pybind11::dict("label"_a = "JIT MPC (FATROP) " + solve_time_str(jit_mpc_result)));
  plt.plot(
      pybind11::make_tuple(compiled_mpc_retult.i_log, compiled_mpc_retult.solve_time_log),
      pybind11::dict("label"_a = "Compiled MPC (FATROP) " + solve_time_str(compiled_mpc_retult)));
  plt.legend();
  plt.xlabel(pybind11::make_tuple("iteration"));
  plt.ylabel(pybind11::make_tuple("MPC solve time [ms]"));

  // set y limit and mpc_result max solve time + 3.0 ms
  double max_solve_time =
      *std::max_element(mpc_result.solve_time_log.begin(), mpc_result.solve_time_log.end());
  plt.ylim(pybind11::make_tuple(0, max_solve_time + 3.0));

  plt.savefig(Args("bench_cartpole_mpc_solve_time_comparison.png"));

  plt.ylim(pybind11::make_tuple(0, 10.0));
  plt.savefig(Args("bench_cartpole_mpc_solve_time_comparison_zoom.png"));

  // compare JIT vs Compiled
  plt.figure();
  plt.plot(pybind11::make_tuple(jit_mpc_result.i_log, jit_mpc_result.solve_time_log),
           pybind11::dict("label"_a = "JIT MPC (FATROP) " + solve_time_str(jit_mpc_result)));
  plt.plot(
      pybind11::make_tuple(compiled_mpc_retult.i_log, compiled_mpc_retult.solve_time_log),
      pybind11::dict("label"_a = "Compiled MPC (FATROP) " + solve_time_str(compiled_mpc_retult)));
  plt.legend();
  plt.xlabel(pybind11::make_tuple("iteration"));
  plt.ylabel(pybind11::make_tuple("MPC solve time [ms]"));
  max_solve_time = std::max(
      *std::max_element(jit_mpc_result.solve_time_log.begin(), jit_mpc_result.solve_time_log.end()),
      *std::max_element(compiled_mpc_retult.solve_time_log.begin(),
                        compiled_mpc_retult.solve_time_log.end()));
  plt.ylim(pybind11::make_tuple(0, max_solve_time + 0.1));

  plt.savefig(Args("bench_cartpole_jit_vs_compiled_mpc_solve_time_comparison.png"));

  return 0;
}
