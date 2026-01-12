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

// Traits for symbolic types
template <typename SymType> struct SymTraits;

template <> struct SymTraits<casadi::SX> {
  using Dict = casadi::SXDict;
  static casadi::SX sym(const std::string &name, casadi_int rows, casadi_int cols = 1) {
    return casadi::SX::sym(name, rows, cols);
  }
};

template <> struct SymTraits<casadi::MX> {
  using Dict = casadi::MXDict;
  static casadi::MX sym(const std::string &name, casadi_int rows, casadi_int cols = 1) {
    return casadi::MX::sym(name, rows, cols);
  }
};

/**
 * @brief Base class for MPC problem definition (templated for SX/MX)
 * @tparam SymType casadi::SX or casadi::MX
 */
template <typename SymType = casadi::MX> class Problem {
public:
  using Sym = SymType;

  enum class DynamicsType {
    ContinuesForwardEuler,
    ContinuesModifiedEuler,
    ContinuesRK4,
    Discretized,
  };

  enum class ConstraintType { Equality, Inequality };

  Problem(DynamicsType dyn_type, size_t _nx, size_t _nu, size_t _horizon, double _dt)
      : dyn_type_(dyn_type), nx_(_nx), nu_(_nu), horizon_(_horizon), dt_vec_(_horizon, _dt) {
    init_bounds();
  }

  Problem(DynamicsType dyn_type, size_t _nx, size_t _nu, std::vector<double> _dt_vec)
      : dyn_type_(dyn_type), nx_(_nx), nu_(_nu), horizon_(_dt_vec.size()),
        dt_vec_(std::move(_dt_vec)) {
    if (dyn_type == DynamicsType::Discretized) {
      throw std::invalid_argument("Adaptive time steps not supported for Discretized dynamics");
    }
    if (dt_vec_.empty()) {
      throw std::invalid_argument("dt_vec must not be empty");
    }
    init_bounds();
  }

private:
  void init_bounds() {
    double inf = std::numeric_limits<double>::infinity();

    Eigen::VectorXd uub = Eigen::VectorXd::Constant(nu(), inf);
    Eigen::VectorXd ulb = -uub;
    u_bounds_ = std::vector<LUbound>{horizon(), {ulb, uub}};

    Eigen::VectorXd xub = Eigen::VectorXd::Constant(nx(), inf);
    Eigen::VectorXd xlb = -xub;
    x_bounds_ = std::vector<LUbound>{horizon(), {xlb, xub}};
  }

public:
  virtual ~Problem() = default;

  // ダイナミクス
  // 連続系の場合は、xとuを引数にとり、dxを返す、コンストラクタで離散化手法を{ContinuesForwardEuler,
  // ContinuesModifiedEuler, ContinuesRK4}から選択する
  // 離散系の場合は、xとuを引数にとり、x(k+1)を返す、コンストラクタでDiscretizedと指定する
  virtual SymType dynamics(SymType x, SymType u) = 0;

  Eigen::VectorXd dynamics_eval(Eigen::VectorXd x, Eigen::VectorXd u) {
    if (dynamics_func_.is_null()) {
      SymType x_sym = SymTraits<SymType>::sym("x", nx());
      SymType u_sym = SymTraits<SymType>::sym("u", nu());
      SymType dx_sym = dynamics(x_sym, u_sym);
      dynamics_func_ = casadi::Function("dynamics_eval", {x_sym, u_sym}, {dx_sym});
    }

    casadi::DM x_dm = casadi_utils::to_casadi(x);
    casadi::DM u_dm = casadi_utils::to_casadi(u);
    std::vector<casadi::DM> result = dynamics_func_(std::vector<casadi::DM>{x_dm, u_dm});
    return casadi_utils::to_eigen(result[0]);
  }

  Eigen::VectorXd simulate(Eigen::VectorXd x0, Eigen::MatrixXd u) {
    assert(dyn_type_ == DynamicsType::Discretized);
    return dynamics_eval(x0, u);
  }

  Eigen::VectorXd simulate(Eigen::VectorXd x0, Eigen::MatrixXd u, double dt) {
    assert(dyn_type_ != DynamicsType::Discretized);
    auto dyn =
        std::bind(&Problem::dynamics_eval, this, std::placeholders::_1, std::placeholders::_2);
    switch (dyn_type_) {
    case DynamicsType::ContinuesForwardEuler:
      return integrate_dynamics_forward_euler<Eigen::VectorXd>(dt, x0, u, dyn);
    case DynamicsType::ContinuesModifiedEuler:
      return integrate_dynamics_modified_euler<Eigen::VectorXd>(dt, x0, u, dyn);
    case DynamicsType::ContinuesRK4:
      return integrate_dynamics_rk4<Eigen::VectorXd>(dt, x0, u, dyn);
    default:
      return x0;
    }
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

  void add_constraint(ConstraintType type, std::function<SymType(SymType, SymType)> constraint) {
    if (type == ConstraintType::Equality) {
      equality_constraints_.push_back(constraint);
    } else {
      inequality_constraints_.push_back(constraint);
    }
  }

  // k番目のステージコスト
  virtual SymType stage_cost(SymType x, SymType u, size_t k) {
    (void)x;
    (void)u;
    (void)k;
    return 0;
  }

  // 終端のコスト
  virtual SymType terminal_cost(SymType x) {
    (void)x;
    return 0;
  }

  DynamicsType dynamics_type() const { return dyn_type_; }
  size_t nx() const { return nx_; }
  size_t nu() const { return nu_; }
  size_t horizon() const { return horizon_; }
  double dt(size_t k = 0) const { return dt_vec_.at(k); }
  const std::vector<double> &dt_vec() const { return dt_vec_; }
  bool is_adaptive_dt() const {
    if (dt_vec_.empty())
      return false;
    double first = dt_vec_[0];
    for (size_t i = 1; i < dt_vec_.size(); ++i) {
      if (dt_vec_[i] != first)
        return true;
    }
    return false;
  }

  SymType parameter(std::string name, size_t rows, size_t cols) {
    auto param = SymTraits<SymType>::sym(name, rows, cols);
    param_list_[name] = {param, casadi::DM::zeros(rows, cols)};
    return param;
  }

  // Helper function to define a reference trajectory parameter
  // The trajectory should have shape (nx, horizon) where each column is the reference state at that
  // horizon step
  SymType reference_trajectory(std::string name = "x_ref") {
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
  std::vector<double> dt_vec_;

  using ConstraintFunc = std::function<SymType(SymType, SymType)>;
  std::vector<ConstraintFunc> equality_constraints_;
  std::vector<ConstraintFunc> inequality_constraints_;

  using LUbound = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
  std::vector<LUbound> u_bounds_;
  std::vector<LUbound> x_bounds_;

  struct SymDMPair {
    SymType sym;
    casadi::DM dm;
  };
  std::map<std::string, SymDMPair> param_list_;

  // Cached dynamics function for evaluation
  mutable casadi::Function dynamics_func_;

  template <typename S> friend class MPC;
  template <typename S> friend class JITMPC;
  template <typename S> friend class CompiledMPC;
};

// Type aliases for convenience
using MXProblem = Problem<casadi::MX>;
using SXProblem = Problem<casadi::SX>;

namespace mpc_config {
/**
 * @brief Get default IPOPT configuration
 * @tparam SymType casadi::MX (default) or casadi::SX
 */
template <typename SymType = casadi::MX> inline casadi::Dict default_ipopt_config() {
  casadi::Dict config = {{"calc_lam_p", true},  {"calc_lam_x", true},
                         {"ipopt.sb", "yes"},   {"ipopt.print_level", 0},
                         {"print_time", false}, {"ipopt.warm_start_init_point", "yes"}};
  // For MX, expand is useful; for SX, it's not needed
  if constexpr (std::is_same_v<SymType, casadi::MX>) {
    config["expand"] = true;
  }
  return config;
}

/**
 * @brief Get default qpOASES configuration
 * @tparam SymType casadi::MX (default) or casadi::SX
 */
template <typename SymType = casadi::MX> inline casadi::Dict default_qpoases_config() {
  casadi::Dict config = {
      {"calc_lam_p", true},
      {"calc_lam_x", true},
      {"max_iter", 100},
      {"print_header", false},
      {"print_iteration", false},
      {"print_status", false},
      {"print_time", false},
      {"qpsol", "qpoases"},
      {"qpsol_options", casadi::Dict{{"enableRegularisation", true}, {"printLevel", "none"}}}};
  if constexpr (std::is_same_v<SymType, casadi::MX>) {
    config["expand"] = true;
  }
  return config;
}

/**
 * @brief Get default FATROP configuration
 * @tparam SymType casadi::MX (default) or casadi::SX
 */
template <typename SymType = casadi::MX> inline casadi::Dict default_fatrop_config() {
  casadi::Dict config = {
      {"calc_lam_p", true},
      {"calc_lam_x", true},
      {"print_time", false},
      {"fatrop.print_level", 0},
      {"fatrop.max_iter", 500},
      {"fatrop.mu_init", 0.1},
      {"structure_detection", "auto"},
      {"fatrop.warm_start_init_point", true},
      {"fatrop.tol", 1e-6},
      {"fatrop.acceptable_tol", 5e-3},
  };
  if constexpr (std::is_same_v<SymType, casadi::MX>) {
    config["expand"] = true;
  }
  return config;
}

} // namespace mpc_config

/**
 * @brief MPC solver class (templated for SX/MX)
 * @tparam SymType casadi::SX or casadi::MX
 */
template <typename SymType = casadi::MX> class MPC {
public:
  using SymDict = typename SymTraits<SymType>::Dict;

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
  MPC(std::shared_ptr<T> prob, std::string solver_name = "ipopt",
      casadi::Dict config = mpc_config::default_ipopt_config<SymType>())
      : prob_(prob), solver_name_(solver_name), config_(config) {
    using namespace casadi;
    static_assert(std::is_base_of_v<Problem<SymType>, T>, "prob must be based on Problem<SymType>");

    const size_t nx = prob_->nx();
    const size_t nu = prob_->nu();
    const size_t N = prob_->horizon();

    build_nlp(nx, nu, N);

    if (equality_required(solver_name_, config_)) {
      std::vector<casadi_int> equality_int(equality_.begin(), equality_.end());
      config_["equality"] = equality_int;
    }

    build_solver();
  }

  virtual ~MPC() = default;

  virtual Eigen::VectorXd solve(Eigen::VectorXd x0,
                                casadi::DMDict new_param_list = casadi::DMDict()) {
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

    return casadi_utils::to_eigen(
        w0_(casadi::Slice(static_cast<casadi_int>(nx), static_cast<casadi_int>(nx + nu))));
  }

  SymDict casadi_prob() const { return casadi_prob_; }
  const std::string &solver_name() const { return solver_name_; }
  casadi::Dict solver_config() const { return config_; }
  std::vector<casadi_int> equality_flags() const {
    return std::vector<casadi_int>(equality_.begin(), equality_.end());
  }

protected:
  std::shared_ptr<Problem<SymType>> prob_;
  std::string solver_name_;
  casadi::Dict config_;
  SymDict casadi_prob_;
  casadi::Function solver_;
  std::vector<SymType> Xs;
  std::vector<SymType> Us;

  casadi::DM lbw_;
  casadi::DM ubw_;
  casadi::DM lbg_;
  casadi::DM ubg_;
  std::vector<casadi::DM> param_vec_;

  std::vector<bool> equality_;

  casadi::DM w0_;
  casadi::DM lam_x0_;
  casadi::DM lam_g0_;

  // Build NLP
  void build_nlp(size_t nx, size_t nu, casadi_int N) {
    using namespace casadi;
    double inf = std::numeric_limits<double>::infinity();

    // 1. Symbolic variables
    Xs.reserve(N + 1);
    Us.reserve(N);
    for (casadi_int i = 0; i < N; i++) {
      Xs.push_back(SymTraits<SymType>::sym("X_" + std::to_string(i), nx, 1));
      Us.push_back(SymTraits<SymType>::sym("U_" + std::to_string(i), nu, 1));
    }
    Xs.push_back(SymTraits<SymType>::sym("X_" + std::to_string(N), nx, 1));

    // Create matrices for map operations
    SymType X = horzcat(Xs);
    SymType U = horzcat(Us);

    SymType x_k = SymTraits<SymType>::sym("x_k", nx);
    SymType u_k = SymTraits<SymType>::sym("u_k", nu);

    // Collect all parameters
    std::vector<SymType> params_sym;
    for (auto &[param_name, param_pair] : prob_->param_list_)
      params_sym.push_back(param_pair.sym);

    // Constraints
    std::vector<SymType> g_k_vec;
    for (auto &con : prob_->equality_constraints_) {
      g_k_vec.push_back(con(x_k, u_k));
    }
    for (auto &con : prob_->inequality_constraints_) {
      g_k_vec.push_back(con(x_k, u_k));
    }
    SymType g_k = vertcat(g_k_vec);

    std::vector<SymType> G_inputs = {x_k, u_k};
    G_inputs.insert(G_inputs.end(), params_sym.begin(), params_sym.end());
    Function G_constraints("G_constraints", G_inputs, {g_k});

    // 2. Build dynamics constraints
    // For SX: use "serial" to avoid code explosion
    // For MX: use default (parallel) as MX handles it well
    std::string map_mode = "serial";

    std::vector<SymType> X_next_list;
    X_next_list.reserve(N);

    if (prob_->dynamics_type() == Problem<SymType>::DynamicsType::Discretized) {
      // Discretized: use F.map (no dt involved)
      SymType x_next = prob_->dynamics(x_k, u_k);
      Function F("F_dynamics", {x_k, u_k}, {x_next});
      SymType X_next_cal = F.map(N, map_mode)(std::vector<SymType>{X(Slice(), Slice(0, N)), U})[0];
      for (casadi_int i = 0; i < N; ++i) {
        X_next_list.push_back(X_next_cal(Slice(), i));
      }
    } else {
      std::function<SymType(SymType, SymType)> con_dyn = std::bind(
          &Problem<SymType>::dynamics, prob_.get(), std::placeholders::_1, std::placeholders::_2);
      const auto &dt_vec = prob_->dt_vec();

      if (!prob_->is_adaptive_dt()) {
        // Uniform dt - use efficient F.map(N)
        double dt = dt_vec[0];
        SymType x_next;
        switch (prob_->dynamics_type()) {
        case Problem<SymType>::DynamicsType::ContinuesForwardEuler:
          x_next = integrate_dynamics_forward_euler<SymType>(dt, x_k, u_k, con_dyn);
          break;
        case Problem<SymType>::DynamicsType::ContinuesModifiedEuler:
          x_next = integrate_dynamics_modified_euler<SymType>(dt, x_k, u_k, con_dyn);
          break;
        case Problem<SymType>::DynamicsType::ContinuesRK4:
          x_next = integrate_dynamics_rk4<SymType>(dt, x_k, u_k, con_dyn);
          break;
        default:
          break;
        }
        Function F("F_dynamics", {x_k, u_k}, {x_next});
        SymType X_next_cal =
            F.map(N, map_mode)(std::vector<SymType>{X(Slice(), Slice(0, N)), U})[0];
        for (casadi_int i = 0; i < N; ++i) {
          X_next_list.push_back(X_next_cal(Slice(), i));
        }
      } else {
        // Adaptive dt - unroll loop, create per-step dynamics
        for (casadi_int i = 0; i < N; ++i) {
          double dt_i = dt_vec[i];
          SymType x_next_i;
          switch (prob_->dynamics_type()) {
          case Problem<SymType>::DynamicsType::ContinuesForwardEuler:
            x_next_i = integrate_dynamics_forward_euler<SymType>(dt_i, Xs[i], Us[i], con_dyn);
            break;
          case Problem<SymType>::DynamicsType::ContinuesModifiedEuler:
            x_next_i = integrate_dynamics_modified_euler<SymType>(dt_i, Xs[i], Us[i], con_dyn);
            break;
          case Problem<SymType>::DynamicsType::ContinuesRK4:
            x_next_i = integrate_dynamics_rk4<SymType>(dt_i, Xs[i], Us[i], con_dyn);
            break;
          default:
            break;
          }
          X_next_list.push_back(x_next_i);
        }
      }
    }

    // Calculate stage costs individually for each horizon step
    std::vector<SymType> stage_costs;
    stage_costs.reserve(N);
    for (casadi_int i = 0; i < N; ++i) {
      std::vector<SymType> stage_cost_inputs = {Xs[i], Us[i]};
      stage_cost_inputs.insert(stage_cost_inputs.end(), params_sym.begin(), params_sym.end());

      SymType cost_i = prob_->stage_cost(Xs[i], Us[i], i);

      Function L_i("L_stage_cost_" + std::to_string(i), stage_cost_inputs, {cost_i});
      stage_costs.push_back(L_i(stage_cost_inputs)[0]);
    }
    SymType J_stage = sum(vertcat(stage_costs));

    // Terminal cost
    SymType terminal_val = prob_->terminal_cost(Xs[N]);
    SymType J = J_stage + terminal_val;

    // Path constraints with map
    SymType G_path;
    if (!g_k.is_empty()) {
      std::vector<SymType> G_map_inputs = {X(Slice(), Slice(0, N)), U};
      for (auto &param : params_sym) {
        G_map_inputs.push_back(repmat(param, 1, N));
      }
      G_path = G_constraints.map(N, map_mode)(G_map_inputs)[0];
    }

    // 4. NLP construction
    std::vector<SymType> w_vec;
    w_vec.reserve(2 * N + 1);
    for (casadi_int i = 0; i < N; ++i) {
      w_vec.push_back(Xs[i]);
      w_vec.push_back(Us[i]);
    }
    w_vec.push_back(Xs[N]);
    SymType w = vertcat(w_vec);

    // Build constraints interleaved per stage for FATROP compatibility
    // Structure: [dynamics_0, path_0, dynamics_1, path_1, ..., dynamics_N-1, path_N-1]
    std::vector<SymType> g_vec;
    for (casadi_int i = 0; i < N; ++i) {
      // Dynamics constraint for stage i: X[i+1] - F(X[i], U[i]) = 0
      g_vec.push_back(Xs[i + 1] - X_next_list[i]);
      // Path constraints for stage i
      if (!g_k.is_empty()) {
        g_vec.push_back(G_path(Slice(), i));
      }
    }

    // Build bounds
    std::vector<double> lbw_numeric, ubw_numeric, lbg_numeric, ubg_numeric;

    auto &u_bounds = prob_->u_bounds_;
    auto &x_bounds = prob_->x_bounds_;

    for (casadi_int i = 0; i < N; ++i) {
      if (i == 0) {
        lbw_numeric.insert(lbw_numeric.end(), nx, 0.0);
        ubw_numeric.insert(ubw_numeric.end(), nx, 0.0);
      } else {
        lbw_numeric.insert(lbw_numeric.end(), x_bounds[i - 1].first.data(),
                           x_bounds[i - 1].first.data() + nx);
        ubw_numeric.insert(ubw_numeric.end(), x_bounds[i - 1].second.data(),
                           x_bounds[i - 1].second.data() + nx);
      }
      lbw_numeric.insert(lbw_numeric.end(), u_bounds[i].first.data(),
                         u_bounds[i].first.data() + nu);
      ubw_numeric.insert(ubw_numeric.end(), u_bounds[i].second.data(),
                         u_bounds[i].second.data() + nu);
    }
    lbw_numeric.insert(lbw_numeric.end(), x_bounds[N - 1].first.data(),
                       x_bounds[N - 1].first.data() + nx);
    ubw_numeric.insert(ubw_numeric.end(), x_bounds[N - 1].second.data(),
                       x_bounds[N - 1].second.data() + nx);

    // Build constraint bounds interleaved per stage (matching g_vec structure)
    for (casadi_int i = 0; i < N; ++i) {
      // Dynamics constraint bounds for stage i
      lbg_numeric.insert(lbg_numeric.end(), nx, 0.0);
      ubg_numeric.insert(ubg_numeric.end(), nx, 0.0);
      equality_.insert(equality_.end(), nx, true);

      // Path constraint bounds for stage i
      for (auto &con : prob_->equality_constraints_) {
        auto con_val = con(x_k, u_k);
        lbg_numeric.insert(lbg_numeric.end(), con_val.size1(), 0.0);
        ubg_numeric.insert(ubg_numeric.end(), con_val.size1(), 0.0);
        equality_.insert(equality_.end(), con_val.size1(), true);
      }
      for (auto &con : prob_->inequality_constraints_) {
        auto con_val = con(x_k, u_k);
        lbg_numeric.insert(lbg_numeric.end(), con_val.size1(), -inf);
        ubg_numeric.insert(ubg_numeric.end(), con_val.size1(), 0.0);
        equality_.insert(equality_.end(), con_val.size1(), false);
      }
    }

    lbw_ = casadi::DM(lbw_numeric);
    ubw_ = casadi::DM(ubw_numeric);
    lbg_ = casadi::DM(lbg_numeric);
    ubg_ = casadi::DM(ubg_numeric);

    SymType g_all = vertcat(g_vec);
    SymType p_all = vertcat(params_sym);

    casadi_prob_ = {{"x", w}, {"f", J}, {"g", g_all}, {"p", p_all}};

    w0_ = DM::zeros(w.size1(), 1);
    lam_x0_ = DM::zeros(w.size1(), 1);
    lam_g0_ = DM::zeros(vertcat(g_vec).size1(), 1);
  }

  virtual void build_solver() { solver_ = nlpsol("solver", solver_name_, casadi_prob_, config_); }
};

/**
 * @brief JIT-compiled MPC solver
 * @tparam SymType casadi::SX or casadi::MX
 */
template <typename SymType = casadi::MX> class JITMPC : public MPC<SymType> {
public:
  template <class T>
  JITMPC(const std::string &prob_name, std::shared_ptr<T> prob, std::string solver_name = "ipopt",
         casadi::Dict config = mpc_config::default_ipopt_config<SymType>(), bool verbose = false)
      : MPC<SymType>(prob, solver_name, config), prob_name_(prob_name) {
    static_assert(std::is_base_of_v<Problem<SymType>, T>, "prob must be based on Problem<SymType>");

    if (verbose) {
      std::cout << "Generating and compiling optimized code..." << std::endl;
    }
    build_jit_solver();
    if (verbose) {
      std::cout << "Code generation completed." << std::endl;
    }
  }

private:
  void build_jit_solver() {
    casadi::Dict jit_options = this->config_;
    jit_options["jit"] = true;
    jit_options["jit_options"] = casadi::Dict{
        {"compiler", "ccache gcc"},
        {"flags", "-O3 -march=native"},
        {"verbose", false},
    };
    jit_options["jit_name"] = "jit_" + prob_name_;
    jit_options["jit_temp_suffix"] = false;
    jit_options["jit_cleanup"] = false;

    this->solver_ =
        casadi::nlpsol("compiled_solver", this->solver_name_, this->casadi_prob_, jit_options);
  }

  void build_solver() override {
    // Do nothing here, solver is built via build_jit_solver() after base class construction
  }

  std::string prob_name_;
};

/**
 * @brief AOT-compiled MPC solver
 * @tparam SymType casadi::SX or casadi::MX
 */
template <typename SymType = casadi::MX> class CompiledMPC : public MPC<SymType> {
public:
  struct CompiledLibraryConfig {
    std::string export_solver_name;
    std::string shared_library_path;
  };

  template <class T>
  CompiledMPC(const CompiledLibraryConfig &lib_config, std::shared_ptr<T> prob)
      : MPC<SymType>(prob), lib_config_(lib_config) {
    static_assert(std::is_base_of_v<Problem<SymType>, T>, "prob must be based on Problem<SymType>");
    load_compiled_solver();
  }

  template <class T>
  static void generate_code(std::shared_ptr<T> prob, const std::string &export_solver_name,
                            const std::string &export_dir, const std::string &solver_name = "ipopt",
                            const casadi::Dict &solver_config = mpc_config::default_ipopt_config(),
                            const casadi::Dict &codegen_options = {}) {
    static_assert(std::is_base_of_v<Problem<SymType>, T>,
                  "Problem type must inherit from Problem<SymType>");
    namespace fs = std::filesystem;
    MPC<SymType> mpc(prob, solver_name, solver_config);

    fs::path out_dir = fs::path(export_dir);
    fs::create_directories(out_dir);
    fs::path c_path = out_dir / (export_solver_name + ".c");

    casadi::Dict solver_cfg = solver_config;
    if (MPC<SymType>::equality_required(solver_name, solver_cfg)) {
      solver_cfg["equality"] = mpc.equality_flags();
    }

    casadi::Function solver =
        casadi::nlpsol(export_solver_name, solver_name, mpc.casadi_prob(), solver_cfg);
    casadi::Dict opts = codegen_options;
    if (opts.find("with_header") == opts.end()) {
      opts["with_header"] = true;
    }
    casadi::CodeGenerator cg(export_solver_name, opts);
    cg.add(solver);
    cg.generate(out_dir.string() + "/");
    std::cout << "Generated solver source at: " << c_path << std::endl;
  }

private:
  void load_compiled_solver() {
    this->solver_ =
        casadi::external(lib_config_.export_solver_name, lib_config_.shared_library_path);
  }

  void build_solver() override {
    // Do nothing here, solver is loaded via load_compiled_solver() after base class construction
  }

  CompiledLibraryConfig lib_config_;
};

template <class Problem>
static void generate_compiled_mpc_code(
    std::shared_ptr<Problem> prob, const std::string &export_solver_name,
    const std::string &export_dir, const std::string &solver_name = "ipopt",
    const casadi::Dict &solver_config = mpc_config::default_ipopt_config<typename Problem::Sym>(),
    const casadi::Dict &codegen_options = {}) {
  CompiledMPC<typename Problem::Sym>::template generate_code<Problem>(
      prob, export_solver_name, export_dir, solver_name, solver_config, codegen_options);
}

} // namespace simple_casadi_mpc
