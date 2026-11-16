// Benchmark: quadrotor MPC with the orientation on SO(3) (casadi::Manifold) versus
// formulations available with upstream CasADi (quaternion with a unit-norm constraint,
// unconstrained quaternion, ZYX Euler angles), for several NLP solvers.
//
// Experiment 1 (ocp): a single cold-started OCP solve from random initial attitudes
//   (0-180 deg, i.e. up to upside down). Records success, iterations, solve time and the
//   quality of the solution: the optimal inputs are applied to a high-accuracy plant and
//   the resulting trajectory is evaluated with the common cost ("true cost"), the
//   attitude error at the end of the horizon and the deviation between prediction and
//   plant, as well as the quaternion norm drift of the prediction.
// Experiment 2 (closed_loop): warm-started receding-horizon control of the plant.
//
// Results are written as CSV files; plot them with plot_quadrotor_manifold_bench.py.
//
// The manifold state is benchmarked with two integrators: Runge-Kutta-Munthe-Kaas
// ("manifold") and RK4 in the ambient coordinates followed by a projection ("manifold_proj").
//
// Usage: quadrotor_manifold_bench [--out DIR] [--solvers fatrop,fatrop_jit_sd,ipopt,...]
//                                 [--ocp-scenarios N] [--cl-scenarios N]
//                                 [--sqp-scenarios N] [--sqp-closed-loop] [--sim-time T]
//                                 [--seed S] [--ocp-timeout S] [--cl-timeout S]
// A "_jit" suffix uses JITMPC (the solver JIT-compiled at construction; the build time is
// written to build_times.csv), "_sd" the option structured_derivatives (stage-wise Hessian
// and Jacobian). By default fatrop, fatrop_sd, fatrop_jit and fatrop_jit_sd are run. Each OCP solve and each closed-loop scenario runs
// in a child process and is recorded as "timeout" if it exceeds the time limit (some
// formulations make a solver spin, e.g. Euler angles close to gimbal lock).

#include "quadrotor_problems.hpp"
#include "simple_casadi_mpc/simple_casadi_mpc.hpp"
#include <casadi/casadi.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <string>

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace quadrotor;
using casadi::DM;
using casadi::Function;
using simple_casadi_mpc::JITMPC;
using simple_casadi_mpc::MPC;
using simple_casadi_mpc::Problem;
using clk = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
struct Options {
  std::string out = "quadrotor_manifold_bench_results";
  std::vector<std::string> solvers = {"fatrop", "fatrop_sd", "fatrop_jit", "fatrop_jit_sd"};
  int ocp_scenarios = 40;
  int cl_scenarios = 12;
  int sqp_scenarios = 10;
  bool sqp_closed_loop = false;
  double sim_time = 3.0;
  unsigned seed = 1;
  double ocp_timeout = 60.0;  // [s] per OCP solve
  double cl_timeout = 180.0;  // [s] per closed-loop scenario
};

Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument("missing value for " + a);
      return argv[++i];
    };
    if (a == "--out")
      o.out = next();
    else if (a == "--solvers") {
      o.solvers.clear();
      std::stringstream ss(next());
      std::string item;
      while (std::getline(ss, item, ','))
        o.solvers.push_back(item);
    } else if (a == "--ocp-scenarios")
      o.ocp_scenarios = std::stoi(next());
    else if (a == "--cl-scenarios")
      o.cl_scenarios = std::stoi(next());
    else if (a == "--sqp-scenarios")
      o.sqp_scenarios = std::stoi(next());
    else if (a == "--sqp-closed-loop")
      o.sqp_closed_loop = true;
    else if (a == "--sim-time")
      o.sim_time = std::stod(next());
    else if (a == "--ocp-timeout")
      o.ocp_timeout = std::stod(next());
    else if (a == "--cl-timeout")
      o.cl_timeout = std::stod(next());
    else if (a == "--seed")
      o.seed = static_cast<unsigned>(std::stoul(next()));
    else
      throw std::invalid_argument("unknown argument " + a);
  }
  return o;
}

// ---------------------------------------------------------------------------
// Solver configurations
// ---------------------------------------------------------------------------
casadi::Dict solver_config(const std::string &solver) {
  if (solver == "fatrop") {
    casadi::Dict c = MPC::default_fatrop_config();
    c["fatrop.max_iter"] = 300;
    return c;
  }
  if (solver == "ipopt") {
    casadi::Dict c = MPC::default_ipopt_config();
    c["ipopt.max_iter"] = 300;
    return c;
  }
  // sqpmethod with the built-in active-set QP solver (no external QP solver needed)
  return casadi::Dict{
      {"qpsol", "qrqp"},
      {"max_iter", 100},
      {"print_header", false},
      {"print_iteration", false},
      {"print_status", false},
      {"print_time", false},
      {"qpsol_options",
       casadi::Dict{{"print_iter", false}, {"print_header", false}, {"error_on_fail", false}}}};
}

/// "<solver>[_jit][_sd]": "_jit" builds a JITMPC (solver JIT-compiled with the default JIT
/// options), "_sd" enables structured_derivatives (stage-wise Hessian / Jacobian)
std::unique_ptr<MPC> make_mpc(const std::string &f, const std::string &solver,
                              std::shared_ptr<Problem> prob, double &build_ms) {
  std::string base = solver;
  bool jit = false, sd = false;
  auto strip = [&](const std::string &suffix) {
    if (base.size() > suffix.size() &&
        base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
      base = base.substr(0, base.size() - suffix.size());
      return true;
    }
    return false;
  };
  for (bool changed = true; changed;) {
    changed = false;
    if (strip("_sd"))
      changed = sd = true;
    if (strip("_jit"))
      changed = jit = true;
  }
  casadi::Dict cfg = solver_config(base);
  if (sd)
    cfg["structured_derivatives"] = true;
  auto t0 = clk::now();
  std::unique_ptr<MPC> mpc;
  if (jit)
    mpc = std::make_unique<JITMPC>("quadrotor_bench_" + f + "_" + solver, prob, base, cfg);
  else
    mpc = std::make_unique<MPC>(prob, base, cfg);
  build_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  return mpc;
}

// ---------------------------------------------------------------------------
// Formulations: construction and state conversion
// ---------------------------------------------------------------------------
const std::vector<std::string> FORMULATIONS = {"manifold", "manifold_proj", "quat_norm",
                                               "quat_free", "euler"};

std::shared_ptr<Problem> make_problem(const std::string &f, const Params &P) {
  if (f == "manifold")
    return std::make_shared<QuadrotorManifoldProb>(P);
  if (f == "manifold_proj")
    return std::make_shared<QuadrotorManifoldProb>(P,
                                                   Problem::DynamicsType::ContinuesProjectedRK4);
  if (f == "quat_norm")
    return std::make_shared<QuadrotorQuatNormProb>(P);
  if (f == "quat_free")
    return std::make_shared<QuadrotorQuatFreeProb>(P);
  return std::make_shared<QuadrotorEulerProb>(P);
}

// plant state [p; v; q; w] (13) -> state of the formulation
Eigen::VectorXd to_formulation(const std::string &f, const Eigen::VectorXd &x) {
  if (f != "euler")
    return x;
  Eigen::VectorXd y(12);
  y.head(6) = x.head(6);
  DM e = quat_to_euler(casadi_utils::to_casadi(Eigen::VectorXd(x.segment(6, 4))));
  for (int i = 0; i < 3; ++i)
    y[6 + i] = e(i).scalar();
  y.tail(3) = x.tail(3);
  return y;
}

// state of the formulation -> plant state (quaternion normalized)
Eigen::VectorXd to_plant(const std::string &f, const Eigen::VectorXd &y) {
  Eigen::VectorXd x(13);
  if (f == "euler") {
    x.head(6) = y.head(6);
    DM q = euler_to_quat(casadi_utils::to_casadi(Eigen::VectorXd(y.segment(6, 3))));
    for (int i = 0; i < 4; ++i)
      x[6 + i] = q(i).scalar();
    x.tail(3) = y.tail(3);
  } else {
    x = y;
    x.segment(6, 4).normalize();
  }
  return x;
}

double quat_norm_deviation(const std::string &f, const Eigen::VectorXd &y) {
  if (f == "euler")
    return 0.0;
  return std::abs(y.segment(6, 4).norm() - 1.0);
}

// geodesic angle between two unit quaternions [deg]
double attitude_error_deg(const Eigen::VectorXd &qa, const Eigen::VectorXd &qb) {
  double c = std::min(1.0, std::abs(qa.normalized().dot(qb.normalized())));
  return 2.0 * std::acos(c) * 180.0 / M_PI;
}

// ---------------------------------------------------------------------------
// Plant (high-accuracy Lie RK4 with substeps) and common cost
// ---------------------------------------------------------------------------
struct Plant {
  Function step;      // (x, u) -> x after one control interval
  Function cost;      // (x, u, p_goal, q_goal) -> stage cost
  Function cost_term; // (x, p_goal, q_goal) -> terminal cost

  Plant(const Params &P, int substeps = 4) {
    using casadi::MX;
    QuadrotorManifoldProb prob(P);
    casadi::Manifold M = QuadrotorManifoldProb::manifold();
    MX x = MX::sym("x", 13), u = MX::sym("u", 4);
    std::function<MX(MX, MX)> f = [&](MX xx, MX uu) { return prob.dynamics(xx, uu); };
    MX xn = x;
    for (int i = 0; i < substeps; ++i)
      xn = simple_casadi_mpc::integrate_dynamics_lie_rk4<MX>(M, P.dt / substeps, xn, u, f);
    step = Function("plant_step", {x, u}, {M.project(xn)}).expand();

    MX pg = MX::sym("p_goal", 3), qg = MX::sym("q_goal", 4);
    MX sc = state_cost(P, x(casadi::Slice(0, 3)), x(casadi::Slice(3, 6)), x(casadi::Slice(6, 10)),
                       x(casadi::Slice(10, 13)), pg, qg);
    cost = Function("cost", {x, u, pg, qg}, {sc + input_cost(P, u)}).expand();
    cost_term = Function("cost_term", {x, pg, qg}, {P.terminal_scale * sc}).expand();
  }

  Eigen::VectorXd advance(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {
    return casadi_utils::to_eigen(step(std::vector<DM>{casadi_utils::to_casadi(x),
                                                       casadi_utils::to_casadi(u)})[0]);
  }
};

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------
struct Scenario {
  int id;
  double angle_deg; // initial tilt from the goal attitude
  Eigen::VectorXd x0; // plant state
};

std::vector<Scenario> make_scenarios(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nrm(0.0, 1.0);
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  casadi::Manifold SO3 = casadi::Manifold::SO3();
  std::vector<Scenario> s;
  for (int i = 0; i < n; ++i) {
    // initial attitude angles spread over (0, 180) deg with random axes
    double angle = M_PI * (i + 0.5) / n;
    Eigen::Vector3d axis(nrm(rng), nrm(rng), nrm(rng));
    axis.normalize();
    Eigen::VectorXd rotvec = angle * axis;
    DM q = SO3.exp(casadi_utils::to_casadi(rotvec));
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(13);
    for (int k = 0; k < 3; ++k)
      x0[k] = uni(rng);
    for (int k = 0; k < 4; ++k)
      x0[6 + k] = q(k).scalar();
    s.push_back({i, angle * 180.0 / M_PI, x0});
  }
  return s;
}

// ---------------------------------------------------------------------------
// Helpers to read the solution
// ---------------------------------------------------------------------------
struct Solution {
  std::vector<Eigen::VectorXd> X; // states of the formulation
  std::vector<Eigen::VectorXd> U;
};

Solution read_solution(const MPC &mpc, const Problem &prob) {
  const DM &w = mpc.last_solution();
  const size_t nx = prob.nx(), nu = prob.nu(), N = prob.horizon();
  const size_t stage = nx + nu;
  Solution s;
  for (size_t k = 0; k <= N; ++k) {
    Eigen::VectorXd x(nx);
    for (size_t l = 0; l < nx; ++l)
      x[l] = w(k * stage + l).scalar();
    s.X.push_back(x);
    if (k < N) {
      Eigen::VectorXd u(nu);
      for (size_t l = 0; l < nu; ++l)
        u[l] = w(k * stage + nx + l).scalar();
      s.U.push_back(u);
    }
  }
  return s;
}

bool stats_success(const MPC &mpc) {
  auto st = mpc.stats();
  auto it = st.find("success");
  return it != st.end() && it->second.is_bool() && it->second.to_bool();
}

std::string stats_status(const MPC &mpc) {
  auto st = mpc.stats();
  auto it = st.find("return_status");
  if (it == st.end())
    return "unknown";
  if (it->second.is_string())
    return it->second.to_string();
  return casadi::str(it->second);
}

int stats_iterations(const MPC &mpc) {
  auto st = mpc.stats();
  auto it = st.find("iter_count");
  return (it != st.end() && it->second.is_int()) ? static_cast<int>(it->second.to_int()) : -1;
}

// ---------------------------------------------------------------------------
// Process isolation with a time limit
// ---------------------------------------------------------------------------
struct IsolatedResult {
  enum Status { Ok, Timeout, Crash } status;
  std::string output;
};

/// Run fn in a forked child and collect the string it returns; kill it after timeout_s.
IsolatedResult run_isolated(double timeout_s, const std::function<std::string()> &fn) {
  int fd[2];
  if (pipe(fd) != 0)
    throw std::runtime_error("pipe failed");
  std::cout.flush();
  pid_t pid = fork();
  if (pid < 0)
    throw std::runtime_error("fork failed");
  if (pid == 0) {
    close(fd[0]);
    std::string out;
    try {
      out = fn();
    } catch (...) {
      _exit(2);
    }
    const char *p = out.data();
    size_t left = out.size();
    while (left > 0) {
      ssize_t n = write(fd[1], p, left);
      if (n <= 0)
        break;
      p += n;
      left -= static_cast<size_t>(n);
    }
    close(fd[1]);
    _exit(0);
  }
  close(fd[1]);
  IsolatedResult res{IsolatedResult::Ok, ""};
  auto t0 = clk::now();
  char buf[1 << 14];
  while (true) {
    double elapsed = std::chrono::duration<double>(clk::now() - t0).count();
    int remaining_ms = static_cast<int>(std::max(0.0, timeout_s - elapsed) * 1000);
    if (remaining_ms == 0) {
      kill(pid, SIGKILL);
      res.status = IsolatedResult::Timeout;
      break;
    }
    pollfd pfd{fd[0], POLLIN, 0};
    int r = poll(&pfd, 1, remaining_ms);
    if (r > 0) {
      ssize_t n = read(fd[0], buf, sizeof(buf));
      if (n > 0)
        res.output.append(buf, static_cast<size_t>(n));
      else
        break; // EOF: the child finished
    }
  }
  int wstatus = 0;
  waitpid(pid, &wstatus, 0);
  close(fd[0]);
  if (res.status == IsolatedResult::Ok &&
      !(WIFEXITED(wstatus) && WEXITSTATUS(wstatus) == 0))
    res.status = IsolatedResult::Crash;
  return res;
}

// ---------------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------------
void run_ocp(const Options &o, const Params &P, const Plant &plant,
             const std::vector<Scenario> &scenarios, std::ofstream &csv,
             std::ofstream &build_csv) {
  build_csv << "formulation,solver,build_time_ms\n";
  csv << "scenario,angle_deg,formulation,solver,success,status,iterations,solve_time_ms,"
         "true_cost,terminal_att_err_deg,terminal_pos_err,pred_att_err_deg,pred_pos_err,"
         "quat_norm_dev\n";
  const DM p_goal = DM::zeros(3, 1);
  const DM q_goal = DM(std::vector<double>{1, 0, 0, 0});
  const casadi::DMDict par = {{"p_goal", p_goal}, {"q_goal", q_goal}};
  Eigen::VectorXd u_hover(4);
  u_hover << P.mass * P.g, 0, 0, 0;

  for (const auto &f : FORMULATIONS) {
    for (const auto &solver : o.solvers) {
      auto prob = make_problem(f, P);
      std::unique_ptr<MPC> mpc;
      try {
        double build_ms = 0;
        mpc = make_mpc(f, solver, prob, build_ms);
        build_csv << f << "," << solver << "," << build_ms << "\n";
        build_csv.flush();
      } catch (std::exception &e) {
        std::cerr << f << "/" << solver << ": construction failed: " << e.what() << std::endl;
        continue;
      }
      int n = solver == "sqpmethod" ? std::min(o.sqp_scenarios, o.ocp_scenarios)
                                    : o.ocp_scenarios;
      std::cout << "[ocp] " << f << "/" << solver << " (" << n << " scenarios)" << std::flush;
      for (int i = 0; i < n; ++i) {
        // spread the subset over all angles
        const Scenario &sc = scenarios[(i * scenarios.size()) / n];
        auto solve_one = [&]() -> std::string {
          std::ostringstream row;
          Eigen::VectorXd y0 = to_formulation(f, sc.x0);
          mpc->initialize_trajectory(y0, u_hover);
          bool ok = true;
          std::string status;
          double ms = 0;
          try {
            auto t0 = clk::now();
            mpc->solve(y0, par);
            ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            ok = stats_success(*mpc);
            status = stats_status(*mpc);
          } catch (std::exception &e) {
            ok = false;
            status = "exception";
          }
          Solution sol = read_solution(*mpc, *prob);
          // roll out the optimal inputs on the plant
          Eigen::VectorXd x = sc.x0;
          double cost = 0, pred_att = 0, pred_pos = 0, qdev = 0;
          bool finite = true;
          for (size_t k = 0; k <= P.horizon; ++k) {
            Eigen::VectorXd xp = to_plant(f, sol.X[k]);
            if (!xp.allFinite() || !x.allFinite()) {
              finite = false;
              break;
            }
            pred_att = std::max(pred_att, attitude_error_deg(xp.segment(6, 4), x.segment(6, 4)));
            pred_pos = std::max(pred_pos, (xp.head(3) - x.head(3)).norm());
            qdev = std::max(qdev, quat_norm_deviation(f, sol.X[k]));
            if (k == P.horizon) {
              cost += plant.cost_term(std::vector<DM>{casadi_utils::to_casadi(x), p_goal,
                                                      q_goal})[0]
                          .scalar();
            } else {
              Eigen::VectorXd u = sol.U[k];
              cost += plant.cost(std::vector<DM>{casadi_utils::to_casadi(x),
                                                 casadi_utils::to_casadi(u), p_goal, q_goal})[0]
                          .scalar();
              x = plant.advance(x, u);
            }
          }
          double nan = std::numeric_limits<double>::quiet_NaN();
          Eigen::VectorXd qg(4);
          qg << 1, 0, 0, 0;
          row << sc.id << "," << sc.angle_deg << "," << f << "," << solver << "," << (ok ? 1 : 0)
              << "," << status << "," << stats_iterations(*mpc) << "," << ms << ","
              << (finite ? cost : nan) << ","
              << (finite ? attitude_error_deg(x.segment(6, 4), qg) : nan) << ","
              << (finite ? x.head(3).norm() : nan) << "," << (finite ? pred_att : nan) << ","
              << (finite ? pred_pos : nan) << "," << (finite ? qdev : nan) << "\n";
          return row.str();
        };
        IsolatedResult r = run_isolated(o.ocp_timeout, solve_one);
        if (r.status == IsolatedResult::Ok) {
          csv << r.output;
          std::cout << "." << std::flush;
        } else {
          bool to = r.status == IsolatedResult::Timeout;
          csv << sc.id << "," << sc.angle_deg << "," << f << "," << solver << ",0,"
              << (to ? "timeout" : "crash") << ",-1," << (to ? o.ocp_timeout * 1e3 : 0.0)
              << ",nan,nan,nan,nan,nan,nan\n";
          std::cout << (to ? "T" : "C") << std::flush;
        }
        csv.flush();
      }
      std::cout << std::endl;
    }
  }
}

void run_closed_loop(const Options &o, const Params &P, const Plant &plant,
                     const std::vector<Scenario> &scenarios, std::ofstream &summary,
                     std::ofstream &steps) {
  summary << "scenario,angle_deg,formulation,solver,success,failed_solves,final_pos_err,"
             "final_att_err_deg,mean_solve_ms,max_solve_ms\n";
  steps << "scenario,formulation,solver,step,time,solve_time_ms,success,pos_err,att_err_deg\n";
  const DM p_goal = DM::zeros(3, 1);
  const DM q_goal = DM(std::vector<double>{1, 0, 0, 0});
  const casadi::DMDict par = {{"p_goal", p_goal}, {"q_goal", q_goal}};
  Eigen::VectorXd u_hover(4);
  u_hover << P.mass * P.g, 0, 0, 0;
  Eigen::VectorXd qg(4);
  qg << 1, 0, 0, 0;
  const int n_steps = static_cast<int>(std::round(o.sim_time / P.dt));

  for (const auto &f : FORMULATIONS) {
    for (const auto &solver : o.solvers) {
      if (solver == "sqpmethod" && !o.sqp_closed_loop)
        continue;
      auto prob = make_problem(f, P);
      std::unique_ptr<MPC> mpc;
      try {
        double build_ms = 0;
        mpc = make_mpc(f, solver, prob, build_ms);
      } catch (std::exception &e) {
        std::cerr << f << "/" << solver << ": construction failed: " << e.what() << std::endl;
        continue;
      }
      std::cout << "[closed loop] " << f << "/" << solver << std::flush;
      for (const auto &sc : scenarios) {
       auto run_scenario = [&]() -> std::string {
        std::ostringstream summary, steps;
        Eigen::VectorXd x = sc.x0;
        mpc->initialize_trajectory(to_formulation(f, x), u_hover);
        int failed = 0;
        double sum_ms = 0, max_ms = 0;
        bool diverged = false;
        for (int k = 0; k < n_steps; ++k) {
          Eigen::VectorXd u = u_hover;
          bool ok = true;
          double ms = 0;
          try {
            auto t0 = clk::now();
            u = mpc->solve(to_formulation(f, x), par);
            ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            ok = stats_success(*mpc);
          } catch (std::exception &e) {
            ok = false;
          }
          if (!u.allFinite()) {
            // Recover from a broken warm start: restart from the current state
            ok = false;
            u = u_hover;
            mpc->initialize_trajectory(to_formulation(f, x), u_hover);
          }
          failed += ok ? 0 : 1;
          sum_ms += ms;
          max_ms = std::max(max_ms, ms);
          x = plant.advance(x, u);
          if (!x.allFinite() || x.head(3).norm() > 100) {
            diverged = true;
          }
          steps << sc.id << "," << f << "," << solver << "," << k << "," << (k + 1) * P.dt << ","
                << ms << "," << (ok ? 1 : 0) << "," << x.head(3).norm() << ","
                << attitude_error_deg(x.segment(6, 4), qg) << "\n";
          if (diverged)
            break;
        }
        double pos_err = diverged ? std::numeric_limits<double>::quiet_NaN() : x.head(3).norm();
        double att_err =
            diverged ? std::numeric_limits<double>::quiet_NaN() : attitude_error_deg(x.segment(6, 4), qg);
        bool success = !diverged && pos_err < 0.1 && att_err < 5.0;
        summary << sc.id << "," << sc.angle_deg << "," << f << "," << solver << ","
                << (success ? 1 : 0) << "," << failed << "," << pos_err << "," << att_err << ","
                << sum_ms / n_steps << "," << max_ms << "\n";
        // summary line first, then the steps
        return summary.str() + steps.str();
       };
       IsolatedResult r = run_isolated(o.cl_timeout, run_scenario);
       if (r.status == IsolatedResult::Ok) {
         size_t eol = r.output.find('\n');
         summary << r.output.substr(0, eol + 1);
         steps << r.output.substr(eol + 1);
         std::cout << "." << std::flush;
       } else {
         bool to = r.status == IsolatedResult::Timeout;
         summary << sc.id << "," << sc.angle_deg << "," << f << "," << solver
                 << ",0,-1,nan,nan,nan," << (to ? o.cl_timeout * 1e3 : 0.0) << "\n";
         std::cout << (to ? "T" : "C") << std::flush;
       }
        summary.flush();
        steps.flush();
      }
      std::cout << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  Options o = parse(argc, argv);
  Params P;
  std::filesystem::create_directories(o.out);
  Plant plant(P);

  std::cout << "Quadrotor manifold benchmark: horizon " << P.horizon << ", dt " << P.dt
            << ", output " << o.out << std::endl;

  {
    std::ofstream csv(o.out + "/ocp_results.csv");
    std::ofstream build_csv(o.out + "/build_times.csv");
    run_ocp(o, P, plant, make_scenarios(o.ocp_scenarios, o.seed), csv, build_csv);
  }
  {
    std::ofstream summary(o.out + "/closed_loop_summary.csv");
    std::ofstream steps(o.out + "/closed_loop_steps.csv");
    run_closed_loop(o, P, plant, make_scenarios(o.cl_scenarios, o.seed + 1), summary, steps);
  }
  std::cout << "Done. Plot with: python3 plot_quadrotor_manifold_bench.py " << o.out << std::endl;
  return 0;
}
