#pragma once
#include "casadi_utils.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <filesystem>
#include <map>
#include <memory>
#include <vector>

namespace simple_casadi_mpc {

template <class T>
static T integrate_dynamics_forward_euler(double dt, T x, T u, std::function<T(T, T)> dynamics) {
  return x + dt * dynamics(x, u);
}

template <class T>
static T integrate_dynamics_modified_euler(double dt, T x, T u, std::function<T(T, T)> dynamics) {
  T k1 = dynamics(x, u);
  T k2 = dynamics(x + dt * k1, u);

  return x + dt * (k1 + k2) / 2;
}

template <class T>
static T integrate_dynamics_rk4(double dt, T x, T u, std::function<T(T, T)> dynamics) {
  T k1 = dynamics(x, u);
  T k2 = dynamics(x + dt / 2 * k1, u);
  T k3 = dynamics(x + dt / 2 * k2, u);
  T k4 = dynamics(x + dt * k3, u);
  return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

class Problem {
public:
  enum class DynamicsType {
    ContinuesForwardEuler,
    ContinuesModifiedEuler,
    ContinuesRK4,
    Discretized,
  };

  enum class ConstraintType {
    Equality,
    Inequality
  };

  Problem(DynamicsType dyn_type, size_t _nx, size_t _nu, size_t _horizon, double _dt)
      : dyn_type_(dyn_type), nx_(_nx), nu_(_nu), horizon_(_horizon), dt_(_dt) {
    double inf = std::numeric_limits<double>::infinity();

    Eigen::VectorXd uub = Eigen::VectorXd::Constant(nu(), inf);
    Eigen::VectorXd ulb = -uub;
    u_bounds_ = std::vector<LUbound>{horizon(), {ulb, uub}};

    Eigen::VectorXd xub = Eigen::VectorXd::Constant(nx(), inf);
    Eigen::VectorXd xlb = -xub;
    x_bounds_ = std::vector<LUbound>{horizon(), {xlb, xub}};
  }

  // ダイナミクス
  // 連続系の場合は、xとuを引数にとり、dxを返す、コンストラクタで離散化手法を{ContinuesForwardEuler,
  // ContinuesModifiedEuler, ContinuesRK4}から選択する
  // 離散系の場合は、xとuを引数にとり、x(k+1)を返す、コンストラクタでDiscretizedと指定する
  virtual casadi::MX dynamics(casadi::MX x, casadi::MX u) = 0;

  Eigen::VectorXd dynamics_eval(Eigen::VectorXd x, Eigen::VectorXd u) {
    casadi::DM x_dm = casadi::DM::zeros(nx(), 1);
    casadi::DM u_dm = casadi::DM::zeros(nu(), 1);
    for (size_t i = 0; i < nx(); i++) {
      x_dm(i) = x[i];
    }
    for (size_t i = 0; i < nu(); i++) {
      u_dm(i) = u[i];
    }
    casadi::MX dx_mx = dynamics(x_dm, u_dm);
    casadi::DM dx_dm = casadi::MX::evalf(dx_mx);
    Eigen::VectorXd dx = casadi_utils::to_eigen(dx_dm);
    return dx;
  }

  Eigen::VectorXd simulate(Eigen::VectorXd x0, Eigen::MatrixXd u) {
    assert(dyn_type_ == DynamicsType::Discretized);
    return dynamics_eval(x0, u);
  }

  Eigen::VectorXd simulate(Eigen::VectorXd x0, Eigen::MatrixXd u, double dt) {
    assert(dyn_type_ != DynamicsType::Discretized);
    auto dyn = std::bind(&Problem::dynamics_eval, this, std::placeholders::_1, std::placeholders::_2);
    switch (dyn_type_) {
    case DynamicsType::ContinuesForwardEuler:
      return integrate_dynamics_forward_euler<Eigen::VectorXd>(dt, x0, u, dyn);
      break;
    case DynamicsType::ContinuesModifiedEuler:
      return integrate_dynamics_modified_euler<Eigen::VectorXd>(dt, x0, u, dyn);
      break;
    case DynamicsType::ContinuesRK4:
      return integrate_dynamics_rk4<Eigen::VectorXd>(dt, x0, u, dyn);
      break;
    case DynamicsType::Discretized:
      break;
    }
    return x0;
  }

  // 操作量と状態量の上下限
  void set_input_bound(Eigen::VectorXd lb, Eigen::VectorXd ub, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      u_bounds_[i] = {lb, ub};
    }
  }

  void set_input_lower_bound(Eigen::VectorXd lb, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      u_bounds_[i].first = lb;
    }
  }

  void set_input_upper_bound(Eigen::VectorXd ub, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      u_bounds_[i].second = ub;
    }
  }

  void set_state_bound(Eigen::VectorXd lb, Eigen::VectorXd ub, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      x_bounds_[i] = {lb, ub};
    }
  }

  void set_state_lower_bound(Eigen::VectorXd lb, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      x_bounds_[i].first = lb;
    }
  }

  void set_state_upper_bound(Eigen::VectorXd ub, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      x_bounds_[i].second = ub;
    }
  }

  void add_constraint(ConstraintType type, std::function<casadi::MX(casadi::MX, casadi::MX)> constrinat) {
    if (type == ConstraintType::Equality) {
      equality_constrinats_.push_back(constrinat);
    } else {
      inequality_constrinats_.push_back(constrinat);
    }
  }

  // k番目のステージコスト
  virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u, size_t k) {
    (void)x;
    (void)u;
    (void)k;
    return 0;
  }

  // 終端のコスト
  virtual casadi::MX terminal_cost(casadi::MX x) {
    (void)x;
    return 0;
  }

  DynamicsType dynamics_type() const { return dyn_type_; }
  size_t nx() const { return nx_; }
  size_t nu() const { return nu_; }
  size_t horizon() const { return horizon_; }
  double dt() const { return dt_; }

  casadi::MX parameter(std::string name, size_t rows, size_t cols) {
    // std::string param_name = "p" + std::to_string(param_list_.size());
    auto param = casadi::MX::sym(name, rows, cols);
    param_list_[name] = {param, casadi::DM::zeros(rows, cols)};
    return param;
  }

  // Helper function to define a reference trajectory parameter
  // The trajectory should have shape (nx, horizon) where each column is the reference state at that horizon step
  casadi::MX reference_trajectory(std::string name = "x_ref") {
    return parameter(name, nx_, horizon_);
  }

private:
  std::pair<int, int> index_range(int start, int end) {
    if (start == -1 && end == -1) {
      return {0, horizon_};
    }
    if (start != -1 && end == -1) {
      return {start, start + 1};
    }
    return {start, end};
  }

  const DynamicsType dyn_type_;
  const size_t nx_;
  const size_t nu_;
  const size_t horizon_;
  const double dt_;

  using ConstraintFunc = std::function<casadi::MX(casadi::MX, casadi::MX)>;
  std::vector<ConstraintFunc> equality_constrinats_;
  std::vector<ConstraintFunc> inequality_constrinats_;

  using LUbound = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
  std::vector<LUbound> u_bounds_;
  std::vector<LUbound> x_bounds_;

  struct MXDMPair {
    casadi::MX mx;
    casadi::DM dm;
  };
  std::map<std::string, MXDMPair> param_list_;

  friend class MPC;
  friend class JITMPC;
  friend class CompiledMPC;
};

class MPC {
public:
  static casadi::Dict default_ipopt_config() {
    casadi::Dict config = {
        {"calc_lam_p", true},
        {"calc_lam_x", true},
        {"ipopt.sb", "yes"},
        {"ipopt.print_level", 0},
        {"print_time", false},
        {"ipopt.warm_start_init_point", "yes"},
        {"expand", true}};
    return config;
  }

  static casadi::Dict default_qpoases_config() {
    casadi::Dict config = {{"calc_lam_p", true},
                           {"calc_lam_x", true},
                           {"max_iter", 100},
                           {"print_header", false},
                           {"print_iteration", false},
                           {"print_status", false},
                           {"print_time", false},
                           {"qpsol", "qpoases"},
                           {"qpsol_options", casadi::Dict{{"enableRegularisation", true}, {"printLevel", "none"}}},
                           {"expand", true}};
    return config;
  }

  static casadi::Dict default_fatrop_config() {
    casadi::Dict config = {
        {"calc_lam_p", true},
        {"calc_lam_x", true},
        {"expand", true},
        {"print_time", false},
        {"fatrop.print_level", 0},
        {"fatrop.max_iter", 500},
        {"fatrop.mu_init", 0.1},
        {"structure_detection", "auto"},
        {"fatrop.warm_start_init_point", true},
        {"fatrop.tol", 1e-6},
        {"fatrop.acceptable_tol", 5e-3},
        // {"debug", true},
    };
    return config;
  }

  static bool equality_required(const std::string &solver_name, const casadi::Dict &config) {
    if (solver_name == "fatrop") {
      auto it = config.find("structure_detection");
      if (it != config.end() && it->second == "auto") {
        return true;
      }
    }
    return false;
  }

  template <class T>
  MPC(std::shared_ptr<T> prob, std::string solver_name = "ipopt", casadi::Dict config = default_ipopt_config())
      : prob_(prob), solver_name_(solver_name), config_(config) {
    using namespace casadi;
    static_assert(std::is_base_of_v<Problem, T>, "prob must be based SimpleProb");

    const size_t nx = prob_->nx();
    const size_t nu = prob_->nu();
    const size_t N = prob_->horizon();

    build_with_map(nx, nu, N);

    if (equality_required(solver_name_, config_)) {
      // Convert std::vector<bool> to std::vector<casadi_int> for CasADi
      std::vector<casadi_int> equality_int(equality_.begin(), equality_.end());
      config_["equality"] = equality_int;
    }

    build_solver();
  }

  virtual Eigen::VectorXd solve(Eigen::VectorXd x0, casadi::DMDict new_param_list = casadi::DMDict()) {
    using namespace casadi;

    // Set new parameter
    for (auto &[param_name, param] : new_param_list) {
      prob_->param_list_[param_name].dm = param;
    }

    const size_t nx = prob_->nx();
    const size_t nu = prob_->nu();

    for (size_t l = 0; l < nx; l++) {
      lbw_(l) = x0[l];
      ubw_(l) = x0[l];
    }

    DMDict arg;
    arg["x0"] = w0_;
    arg["lbx"] = lbw_;
    arg["ubx"] = ubw_;
    arg["lbg"] = lbg_;
    arg["ubg"] = ubg_;
    arg["lam_x0"] = lam_x0_;
    arg["lam_g0"] = lam_g0_;
    param_vec_.clear();
    param_vec_.reserve(prob_->param_list_.size());
    for (auto &[param_name, param_pair] : prob_->param_list_) {
      param_vec_.push_back(param_pair.dm);
    }
    arg["p"] = vertcat(param_vec_);
    DMDict sol = solver_(arg);

    w0_ = sol["x"];
    lam_x0_ = sol["lam_x"];
    lam_g0_ = sol["lam_g"];

    Eigen::VectorXd opt_u(nu);
    std::copy(w0_.ptr() + nx, w0_.ptr() + nx + nu, opt_u.data());

    return opt_u;
  }

  casadi::MXDict casadi_prob() const { return casadi_prob_; }
  const std::string &solver_name() const { return solver_name_; }
  casadi::Dict solver_config() const { return config_; }
  std::vector<casadi_int> equality_flags() const {
    return std::vector<casadi_int>(equality_.begin(), equality_.end());
  }

protected:
  std::shared_ptr<Problem> prob_;
  std::string solver_name_;
  casadi::Dict config_;
  casadi::MXDict casadi_prob_;
  casadi::Function solver_;
  std::vector<casadi::MX> Xs = {};
  std::vector<casadi::MX> Us = {};

  casadi::DM lbw_;
  casadi::DM ubw_;
  casadi::DM lbg_;
  casadi::DM ubg_;
  std::vector<casadi::DM> param_vec_ = {};

  std::vector<bool> equality_ = {}; // ダイナミクスと追加の制約が等式か不等式か

  casadi::DM w0_;
  casadi::DM lam_x0_;
  casadi::DM lam_g0_;

  // Build NLP with map (for expand=false, faster JIT compilation)
  void build_with_map(size_t nx, size_t nu, casadi_int N) {
    using namespace casadi;
    double inf = std::numeric_limits<double>::infinity(); // Make sure inf is defined

    // 1. Symbolic variables - create individual variables for each stage
    Xs.reserve(N + 1);
    Us.reserve(N);
    for (casadi_int i = 0; i < N; i++) {
      Xs.push_back(MX::sym("X_" + std::to_string(i), nx, 1));
      Us.push_back(MX::sym("U_" + std::to_string(i), nu, 1));
    }
    Xs.push_back(MX::sym("X_" + std::to_string(N), nx, 1));

    // Create matrices for map operations
    MX X = horzcat(Xs);
    MX U = horzcat(Us);

    MX x_k = MX::sym("x_k", nx);
    MX u_k = MX::sym("u_k", nu);

    // Collect all parameters
    std::vector<MX> params_mx;
    for (auto &[param_name, param_pair] : prob_->param_list_)
      params_mx.push_back(param_pair.mx);

    // 2. CasADi Functions for one step (unchanged)
    // std::function<MX(MX, MX)> dynamics_func;
    // ... same as your code ...
    MX x_next;
    switch (prob_->dynamics_type()) {
    case Problem::DynamicsType::ContinuesForwardEuler: {
      std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
          std::bind(&Problem::dynamics, prob_.get(), std::placeholders::_1, std::placeholders::_2);
      x_next = integrate_dynamics_forward_euler<casadi::MX>(prob_->dt(), x_k, u_k, con_dyn);
      break;
    }
    case Problem::DynamicsType::ContinuesModifiedEuler: {
      std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
          std::bind(&Problem::dynamics, prob_.get(), std::placeholders::_1, std::placeholders::_2);
      x_next = integrate_dynamics_modified_euler<casadi::MX>(prob_->dt(), x_k, u_k, con_dyn);
      break;
    }
    case Problem::DynamicsType::ContinuesRK4: {
      std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
          std::bind(&Problem::dynamics, prob_.get(), std::placeholders::_1, std::placeholders::_2);
      x_next = integrate_dynamics_rk4<casadi::MX>(prob_->dt(), x_k, u_k, con_dyn);
      break;
    }
    case Problem::DynamicsType::Discretized:
      x_next = prob_->dynamics(x_k, u_k);
      break;
    }
    Function F("F_dynamics", {x_k, u_k}, {x_next});

    // Stage cost function (will be called individually for each horizon step with correct k index)
    std::vector<MX> L_inputs = {x_k, u_k};
    L_inputs.insert(L_inputs.end(), params_mx.begin(), params_mx.end());
    MX stage_cost = prob_->stage_cost(x_k, u_k, 0);
    Function L("L_stage_cost", L_inputs, {stage_cost});

    // 制約一覧
    std::vector<MX> g_k_vec;
    for (auto &con : prob_->equality_constrinats_) {
      g_k_vec.push_back(con(x_k, u_k));
    }
    for (auto &con : prob_->inequality_constrinats_) {
      g_k_vec.push_back(con(x_k, u_k));
    }
    MX g_k = vertcat(g_k_vec);

    std::vector<MX> G_inputs = {x_k, u_k};
    G_inputs.insert(G_inputs.end(), params_mx.begin(), params_mx.end());
    Function G_constraints("G_constraints", G_inputs, {g_k});

    // 3. Map application
    MX X_next_cal = F.map(N)(std::vector<MX>{X(Slice(), Slice(0, N)), U})[0];

    // Calculate stage costs individually for each horizon step to support trajectory-based references
    // This allows each step to use the correct index k in stage_cost()
    std::vector<MX> stage_costs;
    stage_costs.reserve(N);
    for (casadi_int i = 0; i < N; ++i) {
      std::vector<MX> stage_cost_inputs = {Xs[i], Us[i]};
      stage_cost_inputs.insert(stage_cost_inputs.end(), params_mx.begin(), params_mx.end());

      // Call stage_cost with the correct horizon index k=i
      MX cost_i = prob_->stage_cost(Xs[i], Us[i], i);

      // Create function to maintain parameter dependencies
      Function L_i("L_stage_cost_" + std::to_string(i), stage_cost_inputs, {cost_i});
      stage_costs.push_back(L_i(stage_cost_inputs)[0]);
    }
    MX J_stage = sum(vertcat(stage_costs));

    // Terminal cost
    MX terminal_val = prob_->terminal_cost(Xs[N]);
    MX J = J_stage + terminal_val;

    // Path constraints
    MX G_path;
    if (!g_k.is_empty()) {
      std::vector<MX> G_map_inputs = {X(Slice(), Slice(0, N)), U};
      for (auto &param : params_mx) {
        G_map_inputs.push_back(repmat(param, 1, N));
      }
      G_path = G_constraints.map(N)(G_map_inputs)[0];
    }

    // 4. NLP construction
    std::vector<MX> w_vec;
    w_vec.reserve(2 * N + 1);
    for (casadi_int i = 0; i < N; ++i) {
      w_vec.push_back(Xs[i]);
      w_vec.push_back(Us[i]);
    }
    w_vec.push_back(Xs[N]);
    MX w = vertcat(w_vec);

    std::vector<MX> g_vec;
    g_vec.push_back(reshape(X(Slice(), Slice(1, N + 1)) - X_next_cal, nx * N, 1));
    if (!g_k.is_empty()) {
      g_vec.push_back(reshape(G_path, G_path.size1() * G_path.size2(), 1));
    }

    // --- [FIX] Build bounds in temporary double vectors first ---
    std::vector<double> lbw_numeric, ubw_numeric, lbg_numeric, ubg_numeric;

    auto &u_bounds = prob_->u_bounds_;
    auto &x_bounds = prob_->x_bounds_;

    // Bounds for w
    for (casadi_int i = 0; i < N; ++i) {
      if (i == 0) { // Dummy bounds for x_0 (will be overwritten by x0)
        lbw_numeric.insert(lbw_numeric.end(), nx, 0.0);
        ubw_numeric.insert(ubw_numeric.end(), nx, 0.0);
      } else {
        lbw_numeric.insert(lbw_numeric.end(), x_bounds[i - 1].first.data(), x_bounds[i - 1].first.data() + nx);
        ubw_numeric.insert(ubw_numeric.end(), x_bounds[i - 1].second.data(), x_bounds[i - 1].second.data() + nx);
      }
      lbw_numeric.insert(lbw_numeric.end(), u_bounds[i].first.data(), u_bounds[i].first.data() + nu);
      ubw_numeric.insert(ubw_numeric.end(), u_bounds[i].second.data(), u_bounds[i].second.data() + nu);
    }
    // Bounds for x_N
    lbw_numeric.insert(lbw_numeric.end(), x_bounds[N - 1].first.data(), x_bounds[N - 1].first.data() + nx);
    ubw_numeric.insert(ubw_numeric.end(), x_bounds[N - 1].second.data(), x_bounds[N - 1].second.data() + nx);

    // Bounds for g
    // Continuity constraints are all zero
    lbg_numeric.insert(lbg_numeric.end(), nx * N, 0.0);
    ubg_numeric.insert(ubg_numeric.end(), nx * N, 0.0);
    equality_.insert(equality_.end(), nx * N, true);

    // Path constraints bounds
    for (casadi_int i = 0; i < N; ++i) {
      for (auto &con : prob_->equality_constrinats_) {
        auto con_val = con(x_k, u_k); // For getting size
        lbg_numeric.insert(lbg_numeric.end(), con_val.size1(), 0.0);
        ubg_numeric.insert(ubg_numeric.end(), con_val.size1(), 0.0);

        equality_.insert(equality_.end(), con_val.size1(), true);
      }
      for (auto &con : prob_->inequality_constrinats_) {
        auto con_val = con(x_k, u_k); // For getting size
        lbg_numeric.insert(lbg_numeric.end(), con_val.size1(), -inf);
        ubg_numeric.insert(ubg_numeric.end(), con_val.size1(), 0.0);

        equality_.insert(equality_.end(), con_val.size1(), false);
      }
    }

    // Assign from the temporary numeric vectors
    lbw_ = casadi::DM(lbw_numeric);
    ubw_ = casadi::DM(ubw_numeric);
    lbg_ = casadi::DM(lbg_numeric);
    ubg_ = casadi::DM(ubg_numeric);

    MX g_all = vertcat(g_vec);
    MX p_all = vertcat(params_mx);

    casadi_prob_ = {{"x", w}, {"f", J}, {"g", g_all}, {"p", p_all}};

    // Initialize w0_, lam_x0_, lam_g0_ for warm start
    w0_ = DM::zeros(w.size1(), 1);
    lam_x0_ = DM::zeros(w.size1(), 1);
    lam_g0_ = DM::zeros(vertcat(g_vec).size1(), 1);
  }

  virtual void build_solver() {
    solver_ = nlpsol("solver", solver_name_, casadi_prob_, config_);
  }

private:
};

class JITMPC : public MPC {
public:
  template <class T>
  JITMPC(const std::string &prob_name, std::shared_ptr<T> prob, std::string solver_name = "ipopt", casadi::Dict config = MPC::default_ipopt_config(), const bool verbose = false)
      : MPC(prob, solver_name, config), prob_(prob), prob_name_(prob_name) {
    static_assert(std::is_base_of_v<Problem, T>, "prob must be based SimpleProb");

    if (verbose)
      std::cout << "Generating and compiling optimized code..." << std::endl;
    generate_and_compile_code(prob_name);
    if (verbose)
      std::cout << "Code generation completed." << std::endl;
  }

  Eigen::VectorXd solve(Eigen::VectorXd x0, casadi::DMDict new_param_list = casadi::DMDict()) override {
    using namespace casadi;

    for (auto &[param_name, param] : new_param_list) {
      prob_->param_list_[param_name].dm = param;
    }

    const size_t nx = prob_->nx();
    const size_t nu = prob_->nu();

    for (size_t l = 0; l < nx; l++) {
      lbw_(l) = x0[l];
      ubw_(l) = x0[l];
    }

    DMDict arg;
    arg["x0"] = w0_;
    arg["lbx"] = lbw_;
    arg["ubx"] = ubw_;
    arg["lbg"] = lbg_;
    arg["ubg"] = ubg_;
    arg["lam_x0"] = lam_x0_;
    arg["lam_g0"] = lam_g0_;
    param_vec_.clear();
    param_vec_.reserve(prob_->param_list_.size());
    for (auto &[param_name, param_pair] : prob_->param_list_) {
      param_vec_.push_back(param_pair.dm);
    }
    arg["p"] = vertcat(param_vec_);
    DMDict sol = compiled_solver_(arg);

    w0_ = sol["x"];
    lam_x0_ = sol["lam_x"];
    lam_g0_ = sol["lam_g"];

    Eigen::VectorXd opt_u(nu);
    std::copy(w0_.ptr() + nx, w0_.ptr() + nx + nu, opt_u.data());

    return opt_u;
  }

private:
  void generate_and_compile_code(const std::string &prob_name) {
    using namespace casadi;

    // JIT compile the solver for better performance
    Dict jit_options = config_;
    jit_options["jit"] = true;
    jit_options["jit_options"] = Dict{
        {"compiler", "ccache gcc"},
        {"flags", "-O3 -march=native"},
        {"verbose", false},
    };
    jit_options["jit_name"] = "jit_" + prob_name;
    jit_options["jit_temp_suffix"] = false;

    compiled_solver_ = nlpsol("compiled_solver", solver_name_, casadi_prob_, jit_options);
  }

  virtual void build_solver() override {
    // Do nothing, as solver will be built via JIT compilation
  }

  casadi::Function compiled_solver_;
  std::shared_ptr<Problem> prob_;
  std::string prob_name_;
};

class CompiledMPC : public MPC {
public:
  struct CompiledLibraryConfig {
    std::string export_solver_name;
    std::string shared_library_path;
  };

  template <class T>
  CompiledMPC(const CompiledLibraryConfig &lib_config, std::shared_ptr<T> prob)
      : MPC(prob), prob_(prob), lib_config_(lib_config) {
    static_assert(std::is_base_of_v<Problem, T>, "prob must be based SimpleProb");
    load_compiled_solver();
  }

  Eigen::VectorXd solve(Eigen::VectorXd x0, casadi::DMDict new_param_list = casadi::DMDict()) override {
    using namespace casadi;

    for (auto &[param_name, param] : new_param_list) {
      prob_->param_list_[param_name].dm = param;
    }

    const size_t nx = prob_->nx();
    const size_t nu = prob_->nu();

    for (size_t l = 0; l < nx; l++) {
      lbw_(l) = x0[l];
      ubw_(l) = x0[l];
    }

    DMDict arg;
    arg["x0"] = w0_;
    arg["lbx"] = lbw_;
    arg["ubx"] = ubw_;
    arg["lbg"] = lbg_;
    arg["ubg"] = ubg_;
    arg["lam_x0"] = lam_x0_;
    arg["lam_g0"] = lam_g0_;
    param_vec_.clear();
    param_vec_.reserve(prob_->param_list_.size());
    for (auto &[param_name, param_pair] : prob_->param_list_) {
      param_vec_.push_back(param_pair.dm);
    }
    arg["p"] = vertcat(param_vec_);
    DMDict sol = compiled_solver_(arg);

    w0_ = sol["x"];
    lam_x0_ = sol["lam_x"];
    lam_g0_ = sol["lam_g"];

    Eigen::VectorXd opt_u(nu);
    std::copy(w0_.ptr() + nx, w0_.ptr() + nx + nu, opt_u.data());

    return opt_u;
  }

  template <class T>
  static void generate_code(const std::string &export_solver_name, const std::string &export_dir, const std::string &solver_name = "ipopt",
                            const casadi::Dict &solver_config = MPC::default_ipopt_config(), const casadi::Dict &codegen_options = {}) {
    static_assert(std::is_base_of_v<Problem, T>, "Problem type must inherit from Problem");
    namespace fs = std::filesystem;
    auto prob = std::make_shared<T>();
    MPC mpc(prob, solver_name, solver_config);

    fs::path out_dir = fs::path(export_dir);
    fs::create_directories(out_dir);
    fs::path c_path = out_dir / (export_solver_name + ".c");

    casadi::Dict solver_cfg = solver_config;
    if (MPC::equality_required(solver_name, solver_cfg)) {
      solver_cfg["equality"] = mpc.equality_flags();
    }

    casadi::Function solver = casadi::nlpsol(export_solver_name, solver_name, mpc.casadi_prob(), solver_cfg);
    casadi::Dict opts = codegen_options;
    if (opts.find("with_header") == opts.end())
      opts["with_header"] = true;
    casadi::CodeGenerator cg(export_solver_name, opts);
    cg.add(solver);
    cg.generate(out_dir.string() + "/");
    std::cout << "Generated solver source at: " << c_path << std::endl;
  }

private:
  void load_compiled_solver() { compiled_solver_ = casadi::external(lib_config_.export_solver_name, lib_config_.shared_library_path); }
  virtual void build_solver() override {
    // Compiled solver is loaded externally; do not construct a CasADi solver here.
  }

  casadi::Function compiled_solver_;
  std::shared_ptr<Problem> prob_;
  CompiledLibraryConfig lib_config_;
};

} // namespace simple_casadi_mpc
