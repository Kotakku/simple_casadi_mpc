#pragma once
#include "casadi_utils.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <memory>
#include <numeric>
#include <vector>

/// Lightweight C++ utilities for building and solving MPC problems with CasADi.
namespace simple_casadi_mpc {

/// @brief Forward Euler one-step integrator: `x_{k+1} = x + dt * f(x, u)`.
/// @tparam T  state / value type (e.g. `casadi::MX` for symbolic build, `Eigen::VectorXd` for
/// numeric eval).
/// @tparam DT step size type — `double` for numeric, `casadi::MX` to plug in symbolic per-stage dt.
/// @param dt step size.
/// @param x state at step k.
/// @param u input applied between k and k+1.
/// @param dynamics continuous-time dynamics function `f(x, u)` returning dx/dt.
template <class T, class DT>
static T integrate_dynamics_forward_euler(DT dt, T x, T u, std::function<T(T, T)> dynamics) {
  return x + dt * dynamics(x, u);
}

/// @brief Modified (Heun) Euler one-step integrator: 2nd-order explicit.
/// @copydetails integrate_dynamics_forward_euler
template <class T, class DT>
static T integrate_dynamics_modified_euler(DT dt, T x, T u, std::function<T(T, T)> dynamics) {
  T k1 = dynamics(x, u);
  T k2 = dynamics(x + dt * k1, u);

  return x + dt * (k1 + k2) / 2;
}

/// @brief Classical 4-stage Runge-Kutta one-step integrator.
/// @copydetails integrate_dynamics_forward_euler
template <class T, class DT>
static T integrate_dynamics_rk4(DT dt, T x, T u, std::function<T(T, T)> dynamics) {
  T k1 = dynamics(x, u);
  T k2 = dynamics(x + dt / 2 * k1, u);
  T k3 = dynamics(x + dt / 2 * k2, u);
  T k4 = dynamics(x + dt * k3, u);
  return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

/// @brief Lie-group forward Euler step: `x_{k+1} = retract(x, dt * f(x, u))`.
///
/// `f(x, u)` returns the tangent (body-frame) velocity of the state, of size
/// `M.ndx()`; `retract` follows the right-perturbation convention of
/// `casadi::Manifold` (e.g. `q * Exp(dt * omega)` for a quaternion).
/// @tparam T `casadi::MX` (symbolic) or `casadi::DM` (numeric).
/// @tparam DT step size type (`double` or `casadi::MX`).
template <class T, class DT>
static T integrate_dynamics_lie_euler(const casadi::Manifold &M, DT dt, T x, T u,
                                      std::function<T(T, T)> dynamics) {
  return M.retract(x, dt * dynamics(x, u));
}

/// @brief Runge-Kutta-Munthe-Kaas 4th-order step on a Lie group.
///
/// Stage increments \f$\theta_i\f$ live in the tangent space of `x`; the
/// derivative of \f$\theta\f$ is \f$J_r(\theta)^{-1} f(x \oplus \theta, u)\f$, so the
/// scheme is fourth-order accurate on the group and reduces to classical RK4
/// for Euclidean states. Requires `M.is_lie_group()`.
/// @copydetails integrate_dynamics_lie_euler
template <class T, class DT>
static T integrate_dynamics_lie_rk4(const casadi::Manifold &M, DT dt, T x, T u,
                                    std::function<T(T, T)> dynamics) {
  auto rhs = [&](const T &theta) {
    return T::mtimes(M.Jr_inv(theta), dynamics(M.retract(x, theta), u));
  };
  T k1 = dynamics(x, u);
  T k2 = rhs(dt / 2 * k1);
  T k3 = rhs(dt / 2 * k2);
  T k4 = rhs(dt * k3);
  return M.retract(x, dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4));
}

/// @brief Projected RK4 step on a manifold.
///
/// Classical RK4 in the ambient (embedding) coordinates with the ambient velocity
/// `jac_retract_dx(x) * f(x, u)`, followed by `M.project`. Much cheaper to differentiate
/// than @ref integrate_dynamics_lie_rk4 (no exp/log/Jr_inv in the stages) and works on any
/// manifold with a projection; fourth-order accurate up to the projection.
/// @copydetails integrate_dynamics_lie_euler
template <class T, class DT>
static T integrate_dynamics_projected_rk4(const casadi::Manifold &M, DT dt, T x, T u,
                                          std::function<T(T, T)> dynamics, bool project = true) {
  auto rhs = [&](const T &y) { return T::mtimes(M.jac_retract_dx(y), dynamics(y, u)); };
  T k1 = rhs(x);
  T k2 = rhs(x + dt / 2 * k1);
  T k3 = rhs(x + dt / 2 * k2);
  T k4 = rhs(x + dt * k3);
  T x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
  return project ? M.project(x_next) : x_next;
}

/// @brief Base class describing an MPC optimal control problem.
///
/// Derive from this class and override @ref dynamics, @ref stage_cost and
/// optionally @ref terminal_cost. Constraints, bounds and time-varying
/// parameters are configured via the public methods on this class.
///
/// The optimization horizon is fixed at construction time. The dynamics may be
/// either continuous (auto-discretized via Forward Euler / Modified Euler /
/// RK4) or already discretized.
class Problem {
public:
  /// @brief Discretization scheme used to advance the dynamics.
  enum class DynamicsType {
    ContinuesForwardEuler,  ///< Continuous-time, integrated by forward Euler.
    ContinuesModifiedEuler, ///< Continuous-time, integrated by Heun's method.
    ContinuesRK4,           ///< Continuous-time, integrated by classical RK4.
    Discretized,            ///< User supplies x_{k+1} directly.
    /// Continuous-time on the state manifold: @ref dynamics returns the tangent velocity
    /// (size `ndx()`), integrated by `retract(x, dt * f)`.
    ContinuesLieEuler,
    /// Continuous-time on the state manifold: @ref dynamics returns the tangent velocity,
    /// integrated by Runge-Kutta-Munthe-Kaas 4 (requires a Lie group).
    ContinuesLieRK4,
    /// Continuous-time on the state manifold: @ref dynamics returns the tangent velocity,
    /// integrated by RK4 in the ambient coordinates followed by a projection onto the
    /// manifold (cheapest to differentiate; any manifold with a projection).
    ContinuesProjectedRK4,
  };

  /// @brief Constraint kind passed to @ref add_constraint and @ref add_soft_constraint.
  enum class ConstraintType {
    Equality,  ///< `g(x, u) = 0`.
    Inequality ///< `g(x, u) <= 0`.
  };

  /// @brief Construct a problem with a fixed prediction horizon and uniform time step.
  /// @param dyn_type discretization scheme used by @ref dynamics.
  /// @param _nx state dimension.
  /// @param _nu control dimension.
  /// @param _horizon number of stages in the optimization (N).
  /// @param _dt uniform time step in seconds (used by continuous dynamics).
  Problem(DynamicsType dyn_type, size_t _nx, size_t _nu, size_t _horizon, double _dt)
      : Problem(dyn_type, _nx, _nu, _horizon, std::vector<double>(_horizon, _dt)) {}

  /// @brief Construct a problem with a per-stage (variable) time step.
  ///
  /// Each entry of `_dts` is the integration step \f$\Delta t_k\f$ used between
  /// stage \f$k\f$ and \f$k+1\f$. Use this overload when the prediction
  /// horizon spans non-uniform time intervals (e.g. coarse-to-fine schedules).
  /// @param dyn_type discretization scheme used by @ref dynamics.
  /// @param _nx state dimension.
  /// @param _nu control dimension.
  /// @param _horizon number of stages in the optimization (N).
  /// @param _dts per-stage time steps, must have size == `_horizon`.
  Problem(DynamicsType dyn_type, size_t _nx, size_t _nu, size_t _horizon, std::vector<double> _dts)
      : dyn_type_(dyn_type), nx_(_nx), nu_(_nu), horizon_(_horizon), dts_(std::move(_dts)),
        state_manifold_(casadi::Manifold::Euclidean(_nx)) {
    assert(dts_.size() == horizon_ && "dts.size() must equal horizon");
    double inf = std::numeric_limits<double>::infinity();

    Eigen::VectorXd uub = Eigen::VectorXd::Constant(nu(), inf);
    Eigen::VectorXd ulb = -uub;
    u_bounds_ = std::vector<LUbound>{horizon(), {ulb, uub}};

    Eigen::VectorXd xub = Eigen::VectorXd::Constant(nx(), inf);
    Eigen::VectorXd xlb = -xub;
    x_bounds_ = std::vector<LUbound>{horizon(), {xlb, xub}};
  }

  /// @brief Construct a problem whose state lives on a manifold (uniform time step).
  ///
  /// The state is stored in the ambient representation of `state_manifold`
  /// (`nx() == state_manifold.nx()`, e.g. a unit quaternion for SO3), while
  /// the dynamics constraints, the initial condition and the solver's search
  /// directions live in its tangent space (`ndx()`). Typical state:
  /// `casadi::Manifold::product({R3, R3, SO3, R3})` for position, velocity,
  /// orientation and angular velocity.
  ///
  /// Use DynamicsType::ContinuesLieEuler / ContinuesLieRK4 / ContinuesProjectedRK4 and return the
  /// tangent velocity from @ref dynamics, or DynamicsType::Discretized and
  /// return the next state. State bounds are only allowed on the Euclidean
  /// entries (`state_manifold.euclidean_mask()`).
  ///
  /// The NLP solver must support manifolds natively ("sqpmethod", "fatrop") or
  /// handles them through `manifold_fallback = "embed"` (e.g. "ipopt"), which
  /// @ref MPC sets by default.
  Problem(DynamicsType dyn_type, casadi::Manifold state_manifold, size_t _nu, size_t _horizon,
          double _dt)
      : Problem(dyn_type, state_manifold, _nu, _horizon, std::vector<double>(_horizon, _dt)) {}

  /// @brief Manifold-state problem with per-stage time steps. See the uniform-step overload.
  Problem(DynamicsType dyn_type, casadi::Manifold state_manifold, size_t _nu, size_t _horizon,
          std::vector<double> _dts)
      : Problem(dyn_type, static_cast<size_t>(state_manifold.nx()), _nu, _horizon,
                std::move(_dts)) {
    state_manifold_ = state_manifold;
    if (dyn_type == DynamicsType::ContinuesLieRK4 && !state_manifold_.is_lie_group()) {
      throw std::invalid_argument("DynamicsType::ContinuesLieRK4 requires a Lie group, got " +
                                  casadi::str(state_manifold_));
    }
  }

  /// @brief Symbolic dynamics, must be overridden by the user.
  ///
  /// For a continuous DynamicsType, return \f$\frac{dx}{dt} = f(x, u)\f$ given
  /// the current \f$(x, u)\f$; the discretization is applied automatically by
  /// the chosen @ref DynamicsType.
  /// For DynamicsType::Discretized, return the next state \f$x_{k+1}\f$ directly.
  ///
  /// @param x state \f$x_k \in \mathbb{R}^{n_x}\f$.
  /// @param u input \f$u_k \in \mathbb{R}^{n_u}\f$.
  /// For DynamicsType::ContinuesLieEuler / ContinuesLieRK4 / ContinuesProjectedRK4, return the tangent
  /// velocity \f$\xi\f$ of size `ndx()` (e.g. body angular velocity for SO3).
  ///
  /// @return either \f$\frac{dx}{dt}\f$ (continuous), \f$\xi\f$ (Lie) or \f$x_{k+1}\f$ (discrete).
  virtual casadi::MX dynamics(casadi::MX x, casadi::MX u) = 0;

  /// @brief Evaluate @ref dynamics at numeric `(x, u)`. Useful for plant simulation.
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

  /// @brief Advance the plant one step using the user-provided discrete dynamics.
  /// @pre `dynamics_type() == DynamicsType::Discretized`.
  Eigen::VectorXd simulate(Eigen::VectorXd x0, Eigen::MatrixXd u) {
    assert(dyn_type_ == DynamicsType::Discretized);
    return dynamics_eval(x0, u);
  }

  /// @brief Advance the plant `dt` seconds using the configured continuous integrator.
  /// @pre `dynamics_type() != DynamicsType::Discretized`.
  /// @param x0 current state, shape (nx,).
  /// @param u  control held over the step.
  /// @param dt simulation step (independent of the MPC discretization step).
  Eigen::VectorXd simulate(Eigen::VectorXd x0, Eigen::MatrixXd u, double dt) {
    assert(dyn_type_ != DynamicsType::Discretized);
    auto dyn =
        std::bind(&Problem::dynamics_eval, this, std::placeholders::_1, std::placeholders::_2);
    if (dyn_type_ == DynamicsType::ContinuesLieEuler ||
        dyn_type_ == DynamicsType::ContinuesLieRK4 ||
        dyn_type_ == DynamicsType::ContinuesProjectedRK4) {
      std::function<casadi::DM(casadi::DM, casadi::DM)> dyn_dm = [&](casadi::DM x, casadi::DM v) {
        return casadi_utils::to_casadi(dynamics_eval(casadi_utils::to_eigen(x),
                                                 casadi_utils::to_eigen(v)));
      };
      casadi::DM x = casadi_utils::to_casadi(x0);
      casadi::DM v = casadi_utils::to_casadi(Eigen::VectorXd(u));
      casadi::DM x_next =
          dyn_type_ == DynamicsType::ContinuesLieEuler
              ? integrate_dynamics_lie_euler<casadi::DM>(state_manifold_, dt, x, v, dyn_dm)
          : dyn_type_ == DynamicsType::ContinuesLieRK4
              ? integrate_dynamics_lie_rk4<casadi::DM>(state_manifold_, dt, x, v, dyn_dm)
              : integrate_dynamics_projected_rk4<casadi::DM>(state_manifold_, dt, x, v, dyn_dm);
      // Remove round-off drift (e.g. renormalize quaternions)
      return casadi_utils::to_eigen(state_manifold_.project(x_next));
    }
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
    case DynamicsType::ContinuesLieEuler:
    case DynamicsType::ContinuesLieRK4:
    case DynamicsType::ContinuesProjectedRK4:
      break;
    }
    return x0;
  }

  /// @brief Set per-stage input bounds \f$u_{\text{lb},k} \le u_k \le u_{\text{ub},k}\f$.
  /// @param lb lower bound, shape (nu,).
  /// @param ub upper bound, shape (nu,).
  /// @param start first stage (inclusive), or -1 for all stages.
  /// @param end one-past-last stage, or -1 (with `start == -1` applies to all stages;
  ///        with `start != -1, end == -1` applies to the single stage `start`).
  void set_input_bound(Eigen::VectorXd lb, Eigen::VectorXd ub, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      u_bounds_[i] = {lb, ub};
    }
  }

  /// @brief Set only the lower bound on the input. See @ref set_input_bound for `start`/`end`.
  void set_input_lower_bound(Eigen::VectorXd lb, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      u_bounds_[i].first = lb;
    }
  }

  /// @brief Set only the upper bound on the input. See @ref set_input_bound for `start`/`end`.
  void set_input_upper_bound(Eigen::VectorXd ub, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      u_bounds_[i].second = ub;
    }
  }

  /// @brief Set per-stage state bounds \f$x_{\text{lb},k} \le x_k \le x_{\text{ub},k}\f$.
  ///        Range semantics match @ref set_input_bound.
  void set_state_bound(Eigen::VectorXd lb, Eigen::VectorXd ub, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      x_bounds_[i] = {lb, ub};
    }
  }

  /// @brief Set only the lower bound on the state. See @ref set_input_bound for `start`/`end`.
  void set_state_lower_bound(Eigen::VectorXd lb, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      x_bounds_[i].first = lb;
    }
  }

  /// @brief Set only the upper bound on the state. See @ref set_input_bound for `start`/`end`.
  void set_state_upper_bound(Eigen::VectorXd ub, int start = -1, int end = -1) {
    std::tie(start, end) = index_range(start, end);
    for (int i = start; i < end; i++) {
      x_bounds_[i].second = ub;
    }
  }

  /// @brief Add a hard path constraint applied at every stage.
  /// @param type ConstraintType::Equality (\f$g(x, u) = 0\f$) or
  ///        ConstraintType::Inequality (\f$g(x, u) \le 0\f$).
  /// @param constrinat callable returning the constraint vector at one stage.
  void add_constraint(ConstraintType type,
                      std::function<casadi::MX(casadi::MX, casadi::MX)> constrinat) {
    add_constraint_at(type, constrinat, /*start=*/-1, /*end=*/-1);
  }

  /// @brief Add a hard path constraint applied only at the specified stage range.
  ///
  /// Range semantics match @ref set_input_bound:
  /// - `start == -1, end == -1`           → all stages \f$[0, N)\f$ (same as @ref add_constraint).
  /// - `start != -1, end == -1`           → only stage `start`.
  /// - `start != -1, end != -1`           → stages \f$[\text{start}, \text{end})\f$.
  ///
  /// Internally the constraint is still attached to the per-stage `g(x, u)`
  /// vector (so the symbolic structure remains uniform across stages); the
  /// path-constraint bound is set to \f$[-\infty, +\infty]\f$ on inactive
  /// stages so it does not influence the solution.
  void add_constraint_at(ConstraintType type,
                         std::function<casadi::MX(casadi::MX, casadi::MX)> constraint, int start,
                         int end = -1) {
    auto mask = stage_mask(start, end);
    if (type == ConstraintType::Equality) {
      equality_constraints_.push_back({constraint, std::move(mask)});
    } else {
      inequality_constraints_.push_back({constraint, std::move(mask)});
    }
  }

  /// @brief Add a soft path constraint with per-stage non-negative slack.
  ///
  /// Introduces slack \f$s \ge 0\f$ and adds the penalty
  /// \f[
  ///   w_1 \mathbf{1}^\top s + \tfrac{1}{2}\, w_2\, s^\top s
  /// \f]
  /// to the cost. The original constraint is relaxed as:
  /// - Inequality: \f$g(x, u) \le 0 \;\to\; g - s \le 0\f$.
  /// - Equality:   \f$h(x, u) = 0 \;\to\; |h| \le s\f$, i.e. \f$h - s \le 0\f$ and \f$-h - s \le
  /// 0\f$.
  ///
  /// \f$w_2 = 0\f$ (default) gives a pure L1 (exact) penalty; \f$w_2 > 0\f$
  /// adds an L2 (smooth) term. Hard `add_constraint` and soft
  /// `add_soft_constraint` may be mixed on the same problem.
  ///
  /// @param type     constraint kind, see @ref ConstraintType.
  /// @param constraint callable returning the constraint vector at one stage.
  /// @param w1       L1 penalty weight \f$w_1\f$ (default 1e3).
  /// @param w2       L2 penalty weight \f$w_2\f$ (default 0.0).
  void add_soft_constraint(ConstraintType type,
                           std::function<casadi::MX(casadi::MX, casadi::MX)> constraint,
                           double w1 = 1e3, double w2 = 0.0) {
    add_soft_constraint_at(type, constraint, w1, w2, /*start=*/-1, /*end=*/-1);
  }

  /// @brief Add a soft path constraint applied only at the specified stage range.
  ///
  /// Range semantics match @ref add_constraint_at. On inactive stages the
  /// associated slack is forced to 0 (so the penalty contributes nothing) and
  /// the relaxed constraint bound is set to \f$[-\infty, +\infty]\f$.
  void add_soft_constraint_at(ConstraintType type,
                              std::function<casadi::MX(casadi::MX, casadi::MX)> constraint,
                              double w1, double w2, int start, int end = -1) {
    soft_constraints_.push_back({type, constraint, w1, w2, stage_mask(start, end)});
  }

  /// @brief Stage cost \f$L(x_k, u_k, k;\,p)\f$ at step \f$k\f$. Default returns 0.
  /// @param x state at stage k, shape (nx, 1).
  /// @param u input at stage k, shape (nu, 1).
  /// @param k stage index in \f$[0, N)\f$.
  virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u, size_t k) {
    (void)x;
    (void)u;
    (void)k;
    return 0;
  }

  /// @brief Terminal cost \f$\Phi(x_N;\,p)\f$ evaluated at \f$x_N\f$. Default returns 0.
  virtual casadi::MX terminal_cost(casadi::MX x) {
    (void)x;
    return 0;
  }

  DynamicsType dynamics_type() const { return dyn_type_; } ///< Discretization scheme.
  size_t nx() const { return nx_; }                        ///< State dimension (ambient).
  /// Tangent dimension of the state (equal to nx() for Euclidean states).
  size_t ndx() const { return static_cast<size_t>(state_manifold_.ndx()); }
  /// Manifold of the state (`Euclidean(nx)` unless constructed with a manifold).
  const casadi::Manifold &state_manifold() const { return state_manifold_; }
  /// True if the state lives on a non-Euclidean manifold.
  bool has_manifold() const { return !state_manifold_.is_euclidean(); }
  size_t nu() const { return nu_; }                        ///< Control dimension.
  size_t horizon() const { return horizon_; }              ///< Number of stages N.

  /// @brief Step size at stage 0 (or the uniform step for problems built with the
  ///        single-`dt` constructor). Use @ref dt(size_t) for per-stage values.
  double dt() const { return dts_.empty() ? 0.0 : dts_.front(); }

  /// @brief Per-stage time step \f$\Delta t_k\f$.
  double dt(size_t k) const { return dts_.at(k); }

  /// @brief Read-only view of all per-stage time steps, length `horizon()`.
  const std::vector<double> &dts() const { return dts_; }

  /// @brief True iff every stage has the same `dt`.
  bool has_uniform_dt() const {
    if (dts_.size() <= 1)
      return true;
    for (size_t i = 1; i < dts_.size(); ++i) {
      if (dts_[i] != dts_[0])
        return false;
    }
    return true;
  }

  /// @brief Declare a runtime-tunable symbolic parameter that can be passed to @ref MPC::solve.
  ///
  /// Use this to inject reference trajectories, weights, obstacle positions, etc.
  /// without rebuilding the solver. Update its numeric value via `solve(x0, {{name, dm}})`.
  /// @param name unique identifier (used as the key in the DMDict passed to solve).
  /// @param rows number of rows of the parameter matrix.
  /// @param cols number of cols. Use `cols == horizon` for per-stage values
  ///        (column k is consumed at stage k); otherwise the parameter is broadcast.
  /// @return symbolic MX you can use inside @ref dynamics, @ref stage_cost, etc.
  casadi::MX parameter(std::string name, size_t rows, size_t cols) {
    auto param = casadi::MX::sym(name, rows, cols);
    param_list_[name] = {param, casadi::DM::zeros(rows, cols)};
    return param;
  }

  /// @brief Convenience wrapper of @ref parameter with shape \f$(n_x, N)\f$.
  ///
  /// Each column is the reference state at the corresponding stage, suitable
  /// for trajectory tracking inside `stage_cost`.
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

  std::vector<bool> stage_mask(int start, int end) {
    auto [s, e] = index_range(start, end);
    std::vector<bool> mask(horizon_, false);
    for (int k = s; k < e; ++k)
      mask[k] = true;
    return mask;
  }

  const DynamicsType dyn_type_;
  const size_t nx_;
  const size_t nu_;
  const size_t horizon_;
  const std::vector<double> dts_;
  casadi::Manifold state_manifold_;

  using ConstraintFunc = std::function<casadi::MX(casadi::MX, casadi::MX)>;

  struct PathConstraint {
    ConstraintFunc func;
    std::vector<bool> stage_mask; // size == horizon; true at stages where the constraint is active.
  };
  std::vector<PathConstraint> equality_constraints_;
  std::vector<PathConstraint> inequality_constraints_;

  struct SoftConstraint {
    ConstraintType type;
    ConstraintFunc func;
    double w1;
    double w2;
    std::vector<bool> stage_mask;
  };
  std::vector<SoftConstraint> soft_constraints_;

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

/// @brief Runtime MPC solver. Builds a CasADi NLP from a @ref Problem and solves it on demand.
///
/// This is the simplest variant: the NLP is constructed once at construction
/// time and solved at runtime using the chosen CasADi nlpsol backend. For
/// faster iteration time after a startup cost, use @ref JITMPC; for
/// build-time AOT compilation use @ref CompiledMPC.
class MPC {
public:
  /// @brief Reasonable defaults for the IPOPT backend (silent, warm-start enabled).
  static casadi::Dict default_ipopt_config() {
    casadi::Dict config = {{"calc_lam_p", true},  {"calc_lam_x", true},
                           {"ipopt.sb", "yes"},   {"ipopt.print_level", 0},
                           {"print_time", false}, {"ipopt.warm_start_init_point", "yes"},
                           {"expand", true}};
    return config;
  }

  /// @brief Reasonable defaults for SQP method with the qpOASES inner QP solver.
  static casadi::Dict default_qpoases_config() {
    casadi::Dict config = {
        {"calc_lam_p", true},
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

  /// @brief Reasonable defaults for the FATROP backend (auto structure detection).
  static casadi::Dict default_fatrop_config() {
    casadi::Dict config = {
        {"calc_lam_p", true},      {"calc_lam_x", true},
        {"expand", true},          {"print_time", false},
        {"fatrop.print_level", 0}, {"fatrop.max_iter", 500},
        {"fatrop.mu_init", 0.1},   {"structure_detection", "auto"},
        {"fatrop.tol", 1e-6},      {"fatrop.tol_acceptable", 5e-3},
        // {"debug", true},
    };
    return config;
  }

  /// @brief Whether the chosen backend needs an `equality` flag vector in the config.
  ///
  /// FATROP with `structure_detection == "auto"` requires it, so the constructor
  /// inserts it automatically when this returns true.
  static bool equality_required(const std::string &solver_name, const casadi::Dict &config) {
    if (solver_name == "fatrop") {
      auto it = config.find("structure_detection");
      if (it != config.end() && it->second == "auto") {
        return true;
      }
    }
    return false;
  }

  /// @brief Build the NLP from `prob` and create the underlying nlpsol.
  /// @param prob the problem to solve.
  /// @param solver_name CasADi nlpsol backend name (e.g. "ipopt", "fatrop", "sqpmethod").
  /// @param config nlpsol options. Two simple-casadi-mpc-specific keys are
  ///        also recognised and consumed before being forwarded to CasADi:
  ///        - `mapsum_stage_cost` (bool, default true): build the stage-cost
  ///          sum via MapSum so AD stays loop-shaped.
  ///        - `expand_inner_functions` (bool, default true): SX-expand
  ///          per-stage F/L/G before mapping for faster JIT compilation.
  template <class T>
  MPC(std::shared_ptr<T> prob, std::string solver_name = "ipopt",
      casadi::Dict config = default_ipopt_config())
      : prob_(prob), solver_name_(solver_name), config_(config) {
    using namespace casadi;
    static_assert(std::is_base_of_v<Problem, T>, "prob must be based SimpleProb");

    const size_t nx = prob_->nx();
    const size_t nu = prob_->nu();
    const size_t N = prob_->horizon();

    // mapsum_stage_cost: build the stage-cost sum as a MapSum so first-order AD
    //   stays loop-shaped (smaller derivative graph).
    // expand_inner_functions: SX-expand per-stage F/L/G before .map(N) so the
    //   inner per-stage AD operates on flat SX.
    bool mapsum_stage_cost = true;
    bool expand_inner_functions = true;
    {
      auto it = config_.find("mapsum_stage_cost");
      if (it != config_.end()) {
        mapsum_stage_cost = static_cast<bool>(it->second);
        config_.erase(it);
      }
      it = config_.find("expand_inner_functions");
      if (it != config_.end()) {
        expand_inner_functions = static_cast<bool>(it->second);
        config_.erase(it);
      }
    }

    build_with_map(nx, nu, N, mapsum_stage_cost, expand_inner_functions);

    if (expand_inner_functions) {
      config_["expand"] = false;
    }

    if (!x_manifold_.is_null()) {
      // Decision variables live on a manifold: solvers with native support optimize in its
      // tangent space; others (e.g. ipopt) add its embedding constraints ("embed").
      config_["x_manifold"] = x_manifold_;
      if (config_.find("manifold_fallback") == config_.end()) {
        config_["manifold_fallback"] = "embed";
      }
    }

    if (equality_required(solver_name_, config_)) {
      // Convert std::vector<bool> to std::vector<casadi_int> for CasADi
      std::vector<casadi_int> equality_int(equality_.begin(), equality_.end());
      config_["equality"] = equality_int;
    }

    build_solver();
  }

  /// @brief Solve the NLP at the current state and return the first optimal control.
  ///
  /// Warm-starts from the previous solve (`x`, `lam_x`, `lam_g` are cached
  /// internally), so calling this repeatedly during closed-loop simulation
  /// benefits from incremental convergence.
  ///
  /// @param x0 current measured state, shape (nx,).
  /// @param new_param_list updates to parameters declared via @ref Problem::parameter.
  ///        Keys are parameter names; values are `casadi::DM` of matching shape.
  /// @return optimal control to apply now, `u_0`, shape (nu,).
  virtual Eigen::VectorXd solve(Eigen::VectorXd x0,
                                casadi::DMDict new_param_list = casadi::DMDict()) {
    using namespace casadi;

    const size_t nx = prob_->nx();
    const size_t nu = prob_->nu();

    DMDict arg = make_solver_args(x0, new_param_list);
    DMDict sol = solver_(arg);
    update_stats(solver_);

    w0_ = sol["x"];
    lam_x0_ = sol["lam_x"];
    lam_g0_ = sol["lam_g"];

    Eigen::VectorXd opt_u(nu);
    std::copy(w0_.ptr() + nx, w0_.ptr() + nx + nu, opt_u.data());

    return opt_u;
  }

  /// @brief Symbolic NLP description `{x, f, g, p}` constructed from the Problem.
  casadi::MXDict casadi_prob() const { return casadi_prob_; }
  /// @brief Backend name passed to CasADi `nlpsol`.
  const std::string &solver_name() const { return solver_name_; }
  /// @brief Effective config forwarded to `nlpsol` (after consuming simple-casadi-mpc keys).
  casadi::Dict solver_config() const { return config_; }
  /// @brief Per-constraint flags marking equality (`true`) vs inequality (`false`).
  std::vector<casadi_int> equality_flags() const {
    return std::vector<casadi_int>(equality_.begin(), equality_.end());
  }

  /// @brief Set the initial guess of the next solve to a constant trajectory.
  ///
  /// All states are set to `x` and all inputs to `u` (slacks to 0); the warm-start
  /// multipliers are reset. By default the first solve starts from zeros for
  /// Euclidean states and from the initial state for manifold states.
  void initialize_trajectory(const Eigen::VectorXd &x, const Eigen::VectorXd &u) {
    const size_t nx = prob_->nx(), nu = prob_->nu(), N = prob_->horizon();
    w0_ = casadi::DM::zeros(w0_.size1(), 1);
    for (size_t k = 0; k <= N; ++k) {
      for (size_t l = 0; l < nx; ++l)
        w0_(k * w_stage_size_ + l) = x[l];
      if (k < N) {
        for (size_t l = 0; l < nu; ++l)
          w0_(k * w_stage_size_ + nx + l) = u[l];
      }
    }
    lam_x0_ = casadi::DM::zeros(lam_x0_.size1(), 1);
    lam_g0_ = casadi::DM::zeros(lam_g0_.size1(), 1);
    initialized_ = true;
  }

  /// @brief Manifold of the decision vector passed as `x_manifold` (null for Euclidean states).
  const casadi::Manifold &decision_manifold() const { return x_manifold_; }

  /// @brief Solver statistics of the last solve (`return_status`, `success`, `iter_count`, ...).
  casadi::Dict stats() const { return last_stats_; }

  /// @brief Full primal solution of the last solve, `[x_0; u_0; (s_0;) x_1; ...; x_N]`.
  const casadi::DM &last_solution() const { return w0_; }

protected:
  std::shared_ptr<Problem> prob_;
  std::string solver_name_;
  casadi::Dict config_;
  casadi::Manifold x_manifold_;
  casadi::MX x_init_sym_;
  bool initialized_ = false;
  casadi::Dict last_stats_;
  size_t w_stage_size_ = 0; // entries of [x_k; u_k; s_k] in w

  /// @brief Store the statistics of the last solve (tolerates plugins that fail to report).
  void update_stats(const casadi::Function &solver) {
    try {
      last_stats_ = solver.stats();
    } catch (std::exception &e) {
      last_stats_ = casadi::Dict{{"return_status", std::string("unknown")}, {"success", false}};
    }
  }

  /// @brief Assemble the nlpsol arguments for the current state and parameters.
  casadi::DMDict make_solver_args(const Eigen::VectorXd &x0, const casadi::DMDict &new_param_list) {
    using namespace casadi;
    for (auto &[param_name, param] : new_param_list) {
      prob_->param_list_[param_name].dm = param;
    }
    const size_t nx = prob_->nx();
    param_vec_.clear();
    param_vec_.reserve(prob_->param_list_.size() + 1);
    for (auto &[param_name, param_pair] : prob_->param_list_) {
      param_vec_.push_back(param_pair.dm);
    }
    if (prob_->has_manifold()) {
      // Initial state enters through the constraint defect(x_0, x_init) == 0
      param_vec_.push_back(casadi_utils::to_casadi(x0));
      if (!initialized_) {
        // Cold start: the whole state trajectory at the initial state
        for (size_t k = 0; k <= prob_->horizon(); ++k) {
          for (size_t l = 0; l < nx; ++l)
            w0_(k * w_stage_size_ + l) = x0[l];
        }
      }
    } else {
      for (size_t l = 0; l < nx; l++) {
        lbw_(l) = x0[l];
        ubw_(l) = x0[l];
      }
    }
    initialized_ = true;
    DMDict arg;
    arg["x0"] = w0_;
    arg["lbx"] = lbw_;
    arg["ubx"] = ubw_;
    arg["lbg"] = lbg_;
    arg["ubg"] = ubg_;
    arg["lam_x0"] = lam_x0_;
    arg["lam_g0"] = lam_g0_;
    arg["p"] = vertcat(param_vec_);
    return arg;
  }
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
  void build_with_map(size_t nx, size_t nu, casadi_int N, bool mapsum_stage_cost = true,
                      bool expand_inner_functions = true) {
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

    // State on a manifold: the initial state is a parameter (last entry of p) and enters
    // through the stage-0 constraint defect(x_0, x_init) == 0 (no bounds on manifold entries)
    const bool manifold = prob_->has_manifold();
    const Manifold &Mx = prob_->state_manifold();
    const casadi_int ndx = static_cast<casadi_int>(prob_->ndx());
    if (manifold) {
      x_init_sym_ = MX::sym("x_init", nx);
      params_mx.push_back(x_init_sym_);
    }

    // 2. CasADi Functions for one step (unchanged)
    // std::function<MX(MX, MX)> dynamics_func;
    // ... same as your code ...
    // For variable per-stage dt, dt_k_sym is plumbed through F as an extra
    // input and bound to a (1, N) row of dts when calling F.map(N).
    const bool variable_dt =
        !prob_->has_uniform_dt() && prob_->dynamics_type() != Problem::DynamicsType::Discretized;
    MX dt_k_sym = MX::sym("dt_k", 1, 1);
    MX x_next;
    switch (prob_->dynamics_type()) {
    case Problem::DynamicsType::ContinuesForwardEuler: {
      std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
          std::bind(&Problem::dynamics, prob_.get(), std::placeholders::_1, std::placeholders::_2);
      x_next = variable_dt
                   ? integrate_dynamics_forward_euler<casadi::MX>(dt_k_sym, x_k, u_k, con_dyn)
                   : integrate_dynamics_forward_euler<casadi::MX>(prob_->dt(), x_k, u_k, con_dyn);
      break;
    }
    case Problem::DynamicsType::ContinuesModifiedEuler: {
      std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
          std::bind(&Problem::dynamics, prob_.get(), std::placeholders::_1, std::placeholders::_2);
      x_next = variable_dt
                   ? integrate_dynamics_modified_euler<casadi::MX>(dt_k_sym, x_k, u_k, con_dyn)
                   : integrate_dynamics_modified_euler<casadi::MX>(prob_->dt(), x_k, u_k, con_dyn);
      break;
    }
    case Problem::DynamicsType::ContinuesRK4: {
      std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
          std::bind(&Problem::dynamics, prob_.get(), std::placeholders::_1, std::placeholders::_2);
      x_next = variable_dt ? integrate_dynamics_rk4<casadi::MX>(dt_k_sym, x_k, u_k, con_dyn)
                           : integrate_dynamics_rk4<casadi::MX>(prob_->dt(), x_k, u_k, con_dyn);
      break;
    }
    case Problem::DynamicsType::Discretized:
      x_next = prob_->dynamics(x_k, u_k);
      break;
    case Problem::DynamicsType::ContinuesLieEuler: {
      std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
          std::bind(&Problem::dynamics, prob_.get(), std::placeholders::_1, std::placeholders::_2);
      x_next = variable_dt
                   ? integrate_dynamics_lie_euler<casadi::MX>(Mx, dt_k_sym, x_k, u_k, con_dyn)
                   : integrate_dynamics_lie_euler<casadi::MX>(Mx, prob_->dt(), x_k, u_k, con_dyn);
      break;
    }
    case Problem::DynamicsType::ContinuesLieRK4: {
      std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
          std::bind(&Problem::dynamics, prob_.get(), std::placeholders::_1, std::placeholders::_2);
      x_next = variable_dt
                   ? integrate_dynamics_lie_rk4<casadi::MX>(Mx, dt_k_sym, x_k, u_k, con_dyn)
                   : integrate_dynamics_lie_rk4<casadi::MX>(Mx, prob_->dt(), x_k, u_k, con_dyn);
      break;
    }
    case Problem::DynamicsType::ContinuesProjectedRK4: {
      std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
          std::bind(&Problem::dynamics, prob_.get(), std::placeholders::_1, std::placeholders::_2);
      x_next = variable_dt ? integrate_dynamics_projected_rk4<casadi::MX>(Mx, dt_k_sym, x_k, u_k,
                                                                          con_dyn)
                           : integrate_dynamics_projected_rk4<casadi::MX>(Mx, prob_->dt(), x_k,
                                                                          u_k, con_dyn);
      break;
    }
    }
    std::vector<MX> F_inputs = {x_k, u_k};
    if (variable_dt)
      F_inputs.push_back(dt_k_sym);
    Function F("F_dynamics", F_inputs, {x_next});
    if (expand_inner_functions)
      F = F.expand(F.name(), {{"cse", true}});
    // On a manifold the dynamics residual defect(x_{k+1}, F(x_k, u_k)) is nonlinear in F:
    // build it per stage so that it is mapped (and expanded) together with F
    Function F_defect;
    if (manifold) {
      MX x_k1 = MX::sym("x_k1", nx);
      std::vector<MX> D_inputs = F_inputs;
      D_inputs.push_back(x_k1);
      MX x_next_res = x_next;
      if (prob_->dyn_type_ == Problem::DynamicsType::ContinuesProjectedRK4 &&
          Mx.is_lie_group()) {
        // For the built-in Lie groups, defect(y, x) vanishes iff y equals x up to the scale
        // of x (e.g. 2*vec(conj(q_x) * q_y)), so the projection can be skipped in the residual
        // (cheaper second derivatives); F itself stays projected
        std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn = std::bind(
            &Problem::dynamics, prob_.get(), std::placeholders::_1, std::placeholders::_2);
        x_next_res = variable_dt
                         ? integrate_dynamics_projected_rk4<casadi::MX>(Mx, dt_k_sym, x_k, u_k,
                                                                        con_dyn, false)
                         : integrate_dynamics_projected_rk4<casadi::MX>(Mx, prob_->dt(), x_k,
                                                                        u_k, con_dyn, false);
      }
      F_defect = Function("F_defect", D_inputs, {Mx.defect(x_k1, x_next_res)});
      if (expand_inner_functions)
        F_defect = F_defect.expand(F_defect.name(), {{"cse", true}});
    }

    std::vector<MX> L_inputs = {x_k, u_k};
    L_inputs.insert(L_inputs.end(), params_mx.begin(), params_mx.end());
    MX stage_cost = prob_->stage_cost(x_k, u_k, 0);
    Function L("L_stage_cost", L_inputs, {stage_cost});
    if (expand_inner_functions)
      L = L.expand(L.name(), {{"cse", true}});

    // 制約一覧
    std::vector<MX> g_k_vec;
    std::vector<casadi_int> equality_sizes; // size of each equality constraint (per stage)
    std::vector<casadi_int> inequality_sizes;
    equality_sizes.reserve(prob_->equality_constraints_.size());
    inequality_sizes.reserve(prob_->inequality_constraints_.size());
    for (auto &con : prob_->equality_constraints_) {
      MX g_part = con.func(x_k, u_k);
      equality_sizes.push_back(g_part.size1());
      g_k_vec.push_back(g_part);
    }
    for (auto &con : prob_->inequality_constraints_) {
      MX g_part = con.func(x_k, u_k);
      inequality_sizes.push_back(g_part.size1());
      g_k_vec.push_back(g_part);
    }

    // ソフト制約用のスラック変数を per-stage に確保し、対応する不等式を g_k に追加する。
    // - 不等式 g(x,u) <= 0 -> g - s <= 0
    // - 等式  h(x,u)  = 0 -> h - s <= 0 かつ -h - s <= 0  (|h| <= s)
    // s >= 0 は w 側の bound で強制し、ペナルティ w1*1^T s + 0.5*w2*s^T s をコストに加算する。
    std::vector<casadi_int> soft_sizes;
    soft_sizes.reserve(prob_->soft_constraints_.size());
    for (auto &sc : prob_->soft_constraints_) {
      soft_sizes.push_back(sc.func(x_k, u_k).size1());
    }
    const casadi_int n_s_total =
        std::accumulate(soft_sizes.begin(), soft_sizes.end(), static_cast<casadi_int>(0));

    std::vector<MX> Ss;
    MX s_k;
    if (n_s_total > 0) {
      s_k = MX::sym("s_k", n_s_total, 1);
      Ss.reserve(N);
      for (casadi_int i = 0; i < N; ++i) {
        Ss.push_back(MX::sym("S_" + std::to_string(i), n_s_total, 1));
      }

      casadi_int s_offset = 0;
      for (size_t i = 0; i < prob_->soft_constraints_.size(); ++i) {
        auto &sc = prob_->soft_constraints_[i];
        const casadi_int m = soft_sizes[i];
        MX s_part = s_k(Slice(s_offset, s_offset + m));
        MX c_val = sc.func(x_k, u_k);
        if (sc.type == Problem::ConstraintType::Inequality) {
          g_k_vec.push_back(c_val - s_part);
        } else {
          g_k_vec.push_back(c_val - s_part);
          g_k_vec.push_back(-c_val - s_part);
        }
        s_offset += m;
      }
    }

    MX g_k = vertcat(g_k_vec);

    std::vector<MX> G_inputs = {x_k, u_k};
    if (n_s_total > 0) {
      G_inputs.push_back(s_k);
    }
    G_inputs.insert(G_inputs.end(), params_mx.begin(), params_mx.end());
    Function G_constraints("G_constraints", G_inputs, {g_k});
    if (expand_inner_functions)
      G_constraints = G_constraints.expand(G_constraints.name(), {{"cse", true}});

    // 3. Map application
    std::vector<MX> F_map_inputs = {X(Slice(), Slice(0, N)), U};
    if (variable_dt) {
      // (1, N) row of per-stage dts, fed as the third input to F across stages.
      F_map_inputs.push_back(DM(prob_->dts()).T());
    }
    MX X_next_cal;
    MX X_defect; // manifold only
    if (manifold) {
      std::vector<MX> D_map_inputs = F_map_inputs;
      D_map_inputs.push_back(X(Slice(), Slice(1, N + 1)));
      X_defect = F_defect.map(N)(D_map_inputs)[0];
    } else {
      X_next_cal = F.map(N)(F_map_inputs)[0];
    }

    // Stage cost: when mapsum_stage_cost is set, replace each per-stage param
    // p (shape rows×N) by repmat(col_sym, 1, N) inside the user's stage_cost
    // so p(:, k) collapses to a single column symbol; build the cost via
    // MapSum. Verify k-independence by comparing the substituted expression at
    // every k against k=0; on mismatch (k-dependent branching beyond param
    // slicing), fall back to the per-stage loop.
    bool mapsum_safe = mapsum_stage_cost;
    std::vector<MX> col_syms;
    std::vector<size_t> per_stage_idx;
    MX cost_substituted;

    if (mapsum_safe) {
      for (size_t p = 0; p < params_mx.size(); ++p) {
        if (params_mx[p].size2() == N) {
          per_stage_idx.push_back(p);
          col_syms.push_back(MX::sym(params_mx[p].name() + "_col", params_mx[p].size1(), 1));
        }
      }

      std::vector<MX> per_stage_orig;
      std::vector<MX> per_stage_magic;
      per_stage_orig.reserve(per_stage_idx.size());
      per_stage_magic.reserve(per_stage_idx.size());
      for (size_t i = 0; i < per_stage_idx.size(); ++i) {
        per_stage_orig.push_back(params_mx[per_stage_idx[i]]);
        per_stage_magic.push_back(repmat(col_syms[i], 1, N));
      }
      cost_substituted =
          per_stage_orig.empty()
              ? stage_cost
              : MX::substitute(std::vector<MX>{stage_cost}, per_stage_orig, per_stage_magic)[0];

      for (casadi_int k = 1; k < N; ++k) {
        MX cost_at_k = prob_->stage_cost(x_k, u_k, k);
        MX cost_k_subst =
            per_stage_orig.empty()
                ? cost_at_k
                : MX::substitute(std::vector<MX>{cost_at_k}, per_stage_orig, per_stage_magic)[0];
        if (!MX::is_equal(cost_substituted, cost_k_subst, /*depth*/ 100)) {
          casadi_warning("mapsum_stage_cost requested but stage_cost depends on k "
                         "beyond per-stage parameter slicing; falling back to "
                         "per-stage loop to preserve correctness.");
          mapsum_safe = false;
          break;
        }
      }
    }

    MX J_stage;
    if (mapsum_safe) {
      // Per-stage params -> col_syms (Map iterates columns); broadcast params
      // are marked via reduce_in (Map repeats them).
      std::vector<MX> L_subst_inputs = {x_k, u_k};
      std::vector<casadi_int> reduce_in;
      size_t next_col = 0;
      for (size_t p = 0; p < params_mx.size(); ++p) {
        casadi_int input_idx = static_cast<casadi_int>(2 + p);
        bool is_per_stage = (next_col < per_stage_idx.size() && per_stage_idx[next_col] == p);
        if (is_per_stage) {
          L_subst_inputs.push_back(col_syms[next_col]);
          ++next_col;
        } else {
          L_subst_inputs.push_back(params_mx[p]);
          reduce_in.push_back(input_idx);
        }
      }
      Function L_subst("L_stage_subst", L_subst_inputs, {cost_substituted});
      if (expand_inner_functions)
        L_subst = L_subst.expand(L_subst.name(), {{"cse", true}});

      std::vector<casadi_int> reduce_out = {0};
      Function L_mapsum = L_subst.map("L_stage_mapsum", "serial", N, reduce_in, reduce_out, Dict());
      std::vector<MX> map_inputs = {X(Slice(), Slice(0, N)), U};
      for (auto &param : params_mx) {
        map_inputs.push_back(param);
      }
      J_stage = L_mapsum(map_inputs)[0];
    } else {
      std::vector<MX> stage_costs;
      stage_costs.reserve(N);
      for (casadi_int i = 0; i < N; ++i) {
        std::vector<MX> stage_cost_inputs = {Xs[i], Us[i]};
        stage_cost_inputs.insert(stage_cost_inputs.end(), params_mx.begin(), params_mx.end());
        MX cost_i = prob_->stage_cost(Xs[i], Us[i], i);
        Function L_i("L_stage_cost_" + std::to_string(i), stage_cost_inputs, {cost_i});
        stage_costs.push_back(L_i(stage_cost_inputs)[0]);
      }
      J_stage = sum(vertcat(stage_costs));
    }

    // Terminal cost
    MX terminal_val = prob_->terminal_cost(Xs[N]);

    // ソフト制約のスラック変数に対するペナルティコスト
    MX penalty_cost = 0;
    if (n_s_total > 0) {
      for (casadi_int k = 0; k < N; ++k) {
        casadi_int off = 0;
        for (size_t i = 0; i < prob_->soft_constraints_.size(); ++i) {
          auto &sc = prob_->soft_constraints_[i];
          const casadi_int m = soft_sizes[i];
          MX sk_part = Ss[k](Slice(off, off + m));
          penalty_cost = penalty_cost + sc.w1 * sum1(sk_part);
          if (sc.w2 != 0.0) {
            penalty_cost = penalty_cost + 0.5 * sc.w2 * dot(sk_part, sk_part);
          }
          off += m;
        }
      }
    }

    MX J = J_stage + terminal_val + penalty_cost;

    // Path constraints
    MX G_path;
    if (!g_k.is_empty()) {
      std::vector<MX> G_map_inputs = {X(Slice(), Slice(0, N)), U};
      if (n_s_total > 0) {
        G_map_inputs.push_back(horzcat(Ss));
      }
      for (auto &param : params_mx) {
        G_map_inputs.push_back(repmat(param, 1, N));
      }
      G_path = G_constraints.map(N)(G_map_inputs)[0];
    }

    // 4. NLP construction
    std::vector<MX> w_vec;
    w_vec.reserve((n_s_total > 0 ? 3 : 2) * N + 1);
    for (casadi_int i = 0; i < N; ++i) {
      w_vec.push_back(Xs[i]);
      w_vec.push_back(Us[i]);
      if (n_s_total > 0) {
        w_vec.push_back(Ss[i]);
      }
    }
    w_vec.push_back(Xs[N]);
    MX w = vertcat(w_vec);

    // [dynamics_k; path_k] interleaved per stage so that structure-exploiting solvers
    // (fatrop) see the stage-wise OCP structure. Always used on a manifold (dynamics in the
    // tangent space) and for fatrop; other solvers keep the historical ordering.
    const bool interleave = manifold || solver_name_ == "fatrop";
    const casadi_int n_defect = manifold ? ndx : static_cast<casadi_int>(nx);
    std::vector<MX> g_vec;
    if (interleave) {
      // defect: same zeros and first-order behavior as minus, but without the logarithm
      MX defect = manifold ? X_defect : X(Slice(), Slice(1, N + 1)) - X_next_cal;
      MX stages = G_path.is_empty() ? defect : vertcat(defect, G_path);
      if (manifold) {
        // Stage 0 additionally carries the initial condition defect(x_0, x_init) == 0 (in the
        // tangent space): [dynamics_0; initial; path_0], then [dynamics_k; path_k]
        g_vec.push_back(defect(Slice(), 0));
        g_vec.push_back(Mx.defect(X(Slice(), 0), x_init_sym_));
        if (!G_path.is_empty())
          g_vec.push_back(G_path(Slice(), 0));
        if (N > 1)
          g_vec.push_back(reshape(stages(Slice(), Slice(1, N)), stages.size1() * (N - 1), 1));
      } else {
        g_vec.push_back(reshape(stages, stages.size1() * N, 1));
      }
    } else {
      g_vec.push_back(reshape(X(Slice(), Slice(1, N + 1)) - X_next_cal, nx * N, 1));
      if (!g_k.is_empty()) {
        g_vec.push_back(reshape(G_path, G_path.size1() * G_path.size2(), 1));
      }
    }

    // --- [FIX] Build bounds in temporary double vectors first ---
    std::vector<double> lbw_numeric, ubw_numeric, lbg_numeric, ubg_numeric;

    auto &u_bounds = prob_->u_bounds_;
    auto &x_bounds = prob_->x_bounds_;

    if (manifold) {
      // Simple bounds are only allowed on Euclidean entries of the state
      std::vector<bool> mask = Mx.euclidean_mask();
      for (size_t k = 0; k < x_bounds.size(); ++k) {
        for (size_t l = 0; l < nx; ++l) {
          if (!mask[l] && (std::isfinite(x_bounds[k].first[l]) ||
                           std::isfinite(x_bounds[k].second[l]))) {
            throw std::invalid_argument(
                "State bounds are only allowed on Euclidean entries of the state manifold " +
                casadi::str(Mx) + "; entry " + std::to_string(l) + " is not Euclidean.");
          }
        }
      }
    }

    // Bounds for w
    for (casadi_int i = 0; i < N; ++i) {
      if (i == 0 && manifold) { // x_0 is fixed by the initial-condition constraint
        lbw_numeric.insert(lbw_numeric.end(), nx, -inf);
        ubw_numeric.insert(ubw_numeric.end(), nx, inf);
      } else if (i == 0) { // Dummy bounds for x_0 (will be overwritten by x0)
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
      if (n_s_total > 0) {
        // Slack bound is [0, inf] on stages where the soft constraint is
        // active and [0, 0] (force s = 0) elsewhere.
        for (size_t s_i = 0; s_i < prob_->soft_constraints_.size(); ++s_i) {
          const casadi_int m = soft_sizes[s_i];
          const bool active = prob_->soft_constraints_[s_i].stage_mask[i];
          lbw_numeric.insert(lbw_numeric.end(), m, 0.0);
          ubw_numeric.insert(ubw_numeric.end(), m, active ? inf : 0.0);
        }
      }
    }
    // Bounds for x_N
    lbw_numeric.insert(lbw_numeric.end(), x_bounds[N - 1].first.data(),
                       x_bounds[N - 1].first.data() + nx);
    ubw_numeric.insert(ubw_numeric.end(), x_bounds[N - 1].second.data(),
                       x_bounds[N - 1].second.data() + nx);

    // Bounds for g
    // Continuity constraints are all zero
    if (!interleave) {
      lbg_numeric.insert(lbg_numeric.end(), nx * N, 0.0);
      ubg_numeric.insert(ubg_numeric.end(), nx * N, 0.0);
      equality_.insert(equality_.end(), nx * N, true);
    }

    // Path constraints bounds. Inactive stages of staged constraints get
    // bounds set to [-inf, +inf] so the constraint is symbolically present
    // (uniform map structure) but does not influence the solution.
    for (casadi_int i = 0; i < N; ++i) {
      if (interleave) {
        // Dynamics of stage i, then its path constraints
        lbg_numeric.insert(lbg_numeric.end(), n_defect, 0.0);
        ubg_numeric.insert(ubg_numeric.end(), n_defect, 0.0);
        equality_.insert(equality_.end(), n_defect, true);
      }
      if (manifold && i == 0) {
        // Initial condition (stage 0 only)
        lbg_numeric.insert(lbg_numeric.end(), ndx, 0.0);
        ubg_numeric.insert(ubg_numeric.end(), ndx, 0.0);
        equality_.insert(equality_.end(), ndx, true);
      }
      for (size_t ci = 0; ci < prob_->equality_constraints_.size(); ++ci) {
        auto &con = prob_->equality_constraints_[ci];
        const casadi_int sz = equality_sizes[ci];
        const bool active = con.stage_mask[i];
        lbg_numeric.insert(lbg_numeric.end(), sz, active ? 0.0 : -inf);
        ubg_numeric.insert(ubg_numeric.end(), sz, active ? 0.0 : inf);
        equality_.insert(equality_.end(), sz, active);
      }
      for (size_t ci = 0; ci < prob_->inequality_constraints_.size(); ++ci) {
        auto &con = prob_->inequality_constraints_[ci];
        const casadi_int sz = inequality_sizes[ci];
        const bool active = con.stage_mask[i];
        lbg_numeric.insert(lbg_numeric.end(), sz, -inf);
        ubg_numeric.insert(ubg_numeric.end(), sz, active ? 0.0 : inf);
        equality_.insert(equality_.end(), sz, false);
      }
      // Soft path constraints (both forms are one-sided inequalities <= 0).
      // On inactive stages the bound is relaxed to [-inf, +inf]; combined
      // with the slack pinned to [0, 0] above, this nulls the constraint and
      // its penalty contribution at those stages.
      for (size_t s_i = 0; s_i < prob_->soft_constraints_.size(); ++s_i) {
        auto &sc = prob_->soft_constraints_[s_i];
        const casadi_int m = soft_sizes[s_i];
        const casadi_int rows = (sc.type == Problem::ConstraintType::Inequality) ? m : 2 * m;
        const bool active = sc.stage_mask[i];
        lbg_numeric.insert(lbg_numeric.end(), rows, -inf);
        ubg_numeric.insert(ubg_numeric.end(), rows, active ? 0.0 : inf);
        equality_.insert(equality_.end(), rows, false);
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

    w_stage_size_ = nx + nu + n_s_total;
    if (manifold) {
      // Manifold of the decision vector [x_0; u_0; (s_0;) x_1; ...; x_N]
      std::vector<Manifold> comps;
      for (casadi_int i = 0; i < N; ++i) {
        comps.push_back(Mx);
        comps.push_back(Manifold::Euclidean(nu + n_s_total));
      }
      comps.push_back(Mx);
      x_manifold_ = Manifold::product(comps);
      // States start at the identity element until the first solve provides x0
      DM e = Mx.identity();
      for (casadi_int k = 0; k <= N; ++k) {
        for (size_t l = 0; l < nx; ++l)
          w0_(k * w_stage_size_ + l) = e(l);
      }
    }
  }

  virtual void build_solver() { solver_ = nlpsol("solver", solver_name_, casadi_prob_, config_); }

private:
};

/// @brief MPC variant that JIT-compiles the solver during construction.
///
/// Pays a one-time compile cost (cache it with ccache via `default_jit_options()`)
/// in exchange for substantially faster per-iteration solve time. Behaves like
/// @ref MPC otherwise.
class JITMPC : public MPC {
public:
  /// @brief Default JIT compile options (compiler / flags / verbose) passed to CasADi.
  ///
  /// The defaults are `ccache gcc`, `-O3 -march=native`, `verbose=false`.
  /// Override individual entries before passing the dict to the constructor:
  /// @code
  /// auto opts = JITMPC::default_jit_options();
  /// opts["compiler"] = "clang";
  /// opts["flags"] = "-O2";
  /// JITMPC mpc("my_prob", prob, "ipopt", config, opts);
  /// @endcode
  static casadi::Dict default_jit_options() {
    return casadi::Dict{
        {"compiler", "ccache gcc"},
        {"flags", "-O3 -march=native"},
        {"verbose", false},
    };
  }

  /// @brief Build the NLP and JIT-compile its solver in one step.
  /// @param prob_name unique identifier embedded in the JIT artifact name.
  /// @param prob the @ref Problem to solve.
  /// @param solver_name CasADi nlpsol backend name.
  /// @param config nlpsol options (same semantics as @ref MPC::MPC).
  /// @param jit_options inner CasADi `jit_options` dict; see @ref default_jit_options.
  /// @param verbose if true, print progress to stdout while compiling.
  template <class T>
  JITMPC(const std::string &prob_name, std::shared_ptr<T> prob, std::string solver_name = "ipopt",
         casadi::Dict config = MPC::default_ipopt_config(),
         casadi::Dict jit_options = JITMPC::default_jit_options(), const bool verbose = false)
      : MPC(prob, solver_name, config), prob_(prob), prob_name_(prob_name),
        jit_options_(std::move(jit_options)) {
    static_assert(std::is_base_of_v<Problem, T>, "prob must be based SimpleProb");

    if (verbose)
      std::cout << "Generating and compiling optimized code..." << std::endl;
    generate_and_compile_code(prob_name);
    if (verbose)
      std::cout << "Code generation completed." << std::endl;
  }

  /// @copydoc MPC::solve
  Eigen::VectorXd solve(Eigen::VectorXd x0,
                        casadi::DMDict new_param_list = casadi::DMDict()) override {
    using namespace casadi;

    const size_t nx = prob_->nx();
    const size_t nu = prob_->nu();

    DMDict arg = make_solver_args(x0, new_param_list);
    DMDict sol = compiled_solver_(arg);
    update_stats(compiled_solver_);

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

    Dict nlpsol_config = config_;
    nlpsol_config["jit"] = true;
    nlpsol_config["jit_options"] = jit_options_;
    nlpsol_config["jit_name"] = "jit_" + prob_name;
    nlpsol_config["jit_temp_suffix"] = false;

    compiled_solver_ = nlpsol("compiled_solver", solver_name_, casadi_prob_, nlpsol_config);
  }

  virtual void build_solver() override {
    // Do nothing, as solver will be built via JIT compilation
  }

  casadi::Function compiled_solver_;
  std::shared_ptr<Problem> prob_;
  std::string prob_name_;
  casadi::Dict jit_options_;
};

/// @brief MPC variant that loads an externally-compiled solver shared library.
///
/// Use the CMake helper `add_simple_casadi_mpc_codegen` to generate and build
/// the shared library at project build time, then pass the resulting
/// @ref CompiledLibraryConfig (provided by a generated `_config.cpp`) here.
/// Trades flexibility (solver backend and its options are baked in) for the
/// best runtime startup and steady-state solve time.
class CompiledMPC : public MPC {
public:
  /// @brief Locator for an AOT-compiled solver shared library.
  ///
  /// Populated by the generated `<export_solver_name>_config.cpp`; users do
  /// not normally fill this struct by hand.
  struct CompiledLibraryConfig {
    std::string export_solver_name;  ///< CasADi function name embedded in the shared library.
    std::string shared_library_path; ///< Filesystem path to the `.so` / `.dylib` / `.dll`.
  };

  /// @brief Load the prebuilt solver from `lib_config` and bind it to `prob`.
  /// @param lib_config locator for the compiled solver, see @ref CompiledLibraryConfig.
  /// @param prob the matching @ref Problem instance (used for shapes and parameters).
  template <class T>
  CompiledMPC(const CompiledLibraryConfig &lib_config, std::shared_ptr<T> prob)
      : MPC(prob), prob_(prob), lib_config_(lib_config) {
    static_assert(std::is_base_of_v<Problem, T>, "prob must be based SimpleProb");
    load_compiled_solver();
  }

  /// @copydoc MPC::solve
  Eigen::VectorXd solve(Eigen::VectorXd x0,
                        casadi::DMDict new_param_list = casadi::DMDict()) override {
    using namespace casadi;

    const size_t nx = prob_->nx();
    const size_t nu = prob_->nu();

    DMDict arg = make_solver_args(x0, new_param_list);
    DMDict sol = compiled_solver_(arg);
    update_stats(compiled_solver_);

    w0_ = sol["x"];
    lam_x0_ = sol["lam_x"];
    lam_g0_ = sol["lam_g"];

    Eigen::VectorXd opt_u(nu);
    std::copy(w0_.ptr() + nx, w0_.ptr() + nx + nu, opt_u.data());

    return opt_u;
  }

  /// @brief Emit the C source for an AOT-compiled solver. Called from a codegen executable.
  ///
  /// The generated `<export_solver_name>.c` (and matching header) is compiled
  /// into a shared library by the `add_simple_casadi_mpc_codegen` CMake helper.
  ///
  /// @tparam T the @ref Problem subclass to instantiate.
  /// @param export_solver_name CasADi function name embedded in the generated source.
  /// @param export_dir output directory for the generated files.
  /// @param solver_name CasADi nlpsol backend baked into the compiled solver.
  /// @param solver_config nlpsol options baked in (cannot be changed at runtime).
  /// @param codegen_options forwarded to `casadi::CodeGenerator`.
  template <class T>
  static void generate_code(const std::string &export_solver_name, const std::string &export_dir,
                            const std::string &solver_name = "ipopt",
                            const casadi::Dict &solver_config = MPC::default_ipopt_config(),
                            const casadi::Dict &codegen_options = {}) {
    static_assert(std::is_base_of_v<Problem, T>, "Problem type must inherit from Problem");
    namespace fs = std::filesystem;
    auto prob = std::make_shared<T>();
    MPC mpc(prob, solver_name, solver_config);

    fs::path out_dir = fs::path(export_dir);
    fs::create_directories(out_dir);
    fs::path c_path = out_dir / (export_solver_name + ".c");

    // Use the MPC-preprocessed config (equality flags applied, custom flags popped).
    casadi::Dict solver_cfg = mpc.solver_config();

    casadi::Function solver =
        casadi::nlpsol(export_solver_name, solver_name, mpc.casadi_prob(), solver_cfg);
    casadi::Dict opts = codegen_options;
    if (opts.find("with_header") == opts.end())
      opts["with_header"] = true;
    casadi::CodeGenerator cg(export_solver_name, opts);
    cg.add(solver);
    cg.generate(out_dir.string() + "/");
    std::cout << "Generated solver source at: " << c_path << std::endl;
  }

private:
  void load_compiled_solver() {
    compiled_solver_ =
        casadi::external(lib_config_.export_solver_name, lib_config_.shared_library_path);
  }
  virtual void build_solver() override {
    // Compiled solver is loaded externally; do not construct a CasADi solver here.
  }

  casadi::Function compiled_solver_;
  std::shared_ptr<Problem> prob_;
  CompiledLibraryConfig lib_config_;
};

} // namespace simple_casadi_mpc
