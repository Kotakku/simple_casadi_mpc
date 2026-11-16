#pragma once
// Quadrotor attitude/position control problem in four state representations.
//
// All formulations describe the same optimal control problem:
//   state  : position p, velocity v (world), orientation, body angular velocity w
//   input  : collective thrust T and body torques tau
//   cost   : tracking of a goal hover state, with the attitude error
//            1 - (q . q_goal)^2 (sign-invariant, ~ theta^2 / 4 for small angles)
//
// Formulations
//   QuadrotorManifoldProb  state on R3 x R3 x SO3 x R3 (casadi::Manifold), Lie RK4
//   QuadrotorQuatNormProb  quaternion in R4, RK4 on q' = q * [0, w] / 2, |q|^2 = 1 as a
//                          path equality constraint (upstream CasADi)
//   QuadrotorQuatFreeProb  quaternion in R4 without any norm constraint; the cost uses
//                          q / |q| (upstream CasADi)
//   QuadrotorEulerProb     ZYX Euler angles (roll, pitch, yaw) in R3, singular at
//                          pitch = +-90 deg (upstream CasADi)

#include "simple_casadi_mpc/simple_casadi_mpc.hpp"
#include <casadi/casadi.hpp>

namespace quadrotor {

using casadi::DM;
using casadi::MX;
using simple_casadi_mpc::Problem;

struct Params {
  double mass = 1.0;
  double g = 9.81;
  DM inertia = DM(std::vector<double>{0.01, 0.01, 0.02}); // diagonal
  size_t horizon = 20;
  double dt = 0.05;
  // cost weights
  double w_pos = 10.0, w_vel = 1.0, w_att = 50.0, w_rate = 0.1;
  double w_thrust = 0.01, w_torque = 1.0;
  double terminal_scale = 10.0;
  // input bounds
  double thrust_max = 25.0, torque_max = 0.5;
};

// ---- quaternion helpers (Hamilton, [w, x, y, z]) usable with plain CasADi ----
inline MX quat_mul(const MX &a, const MX &b) {
  return MX::vertcat({a(0) * b(0) - a(1) * b(1) - a(2) * b(2) - a(3) * b(3),
                      a(0) * b(1) + a(1) * b(0) + a(2) * b(3) - a(3) * b(2),
                      a(0) * b(2) - a(1) * b(3) + a(2) * b(0) + a(3) * b(1),
                      a(0) * b(3) + a(1) * b(2) - a(2) * b(1) + a(3) * b(0)});
}

inline MX quat_to_rot(const MX &q) {
  MX w = q(0), x = q(1), y = q(2), z = q(3);
  MX R = MX::zeros(3, 3);
  R(0, 0) = 1 - 2 * (y * y + z * z);
  R(0, 1) = 2 * (x * y - w * z);
  R(0, 2) = 2 * (x * z + w * y);
  R(1, 0) = 2 * (x * y + w * z);
  R(1, 1) = 1 - 2 * (x * x + z * z);
  R(1, 2) = 2 * (y * z - w * x);
  R(2, 0) = 2 * (x * z - w * y);
  R(2, 1) = 2 * (y * z + w * x);
  R(2, 2) = 1 - 2 * (x * x + y * y);
  return R;
}

// ZYX Euler angles (roll, pitch, yaw) -> quaternion
inline MX euler_to_quat(const MX &e) {
  MX cr = cos(e(0) / 2), sr = sin(e(0) / 2);
  MX cp = cos(e(1) / 2), sp = sin(e(1) / 2);
  MX cy = cos(e(2) / 2), sy = sin(e(2) / 2);
  return MX::vertcat({cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                      cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy});
}

inline DM euler_to_quat(const DM &e) { return MX::evalf(euler_to_quat(MX(e))); }

// quaternion -> ZYX Euler angles (numeric)
inline DM quat_to_euler(const DM &q) {
  double w = q(0).scalar(), x = q(1).scalar(), y = q(2).scalar(), z = q(3).scalar();
  double roll = std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  double sp = std::clamp(2 * (w * y - z * x), -1.0, 1.0);
  double pitch = std::asin(sp);
  double yaw = std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return DM(std::vector<double>{roll, pitch, yaw});
}

// Rigid-body accelerations shared by all formulations
inline MX lin_acc(const Params &P, const MX &R, const MX &u) {
  return MX::mtimes(R, MX::vertcat({0, 0, u(0) / P.mass})) - MX(DM(std::vector<double>{0, 0, P.g}));
}
inline MX ang_acc(const Params &P, const MX &w, const MX &u) {
  MX Jw = MX(P.inertia) * w;
  return (u(casadi::Slice(1, 4)) - MX::cross(w, Jw)) / MX(P.inertia);
}

// Attitude cost shared by all formulations: 1 - (q . q_goal)^2
inline MX attitude_cost(const MX &q_unit, const MX &q_goal) {
  MX c = MX::dot(q_unit, q_goal);
  return 1 - c * c;
}

/// Stage cost shared by all formulations, in terms of physical quantities.
inline MX state_cost(const Params &P, const MX &pos, const MX &vel, const MX &q_unit, const MX &rate,
                     const MX &p_goal, const MX &q_goal) {
  return P.w_pos * MX::sumsqr(pos - p_goal) + P.w_vel * MX::sumsqr(vel) +
         P.w_att * attitude_cost(q_unit, q_goal) + P.w_rate * MX::sumsqr(rate);
}
inline MX input_cost(const Params &P, const MX &u) {
  return P.w_thrust * MX::sumsqr(u(0) - P.mass * P.g) +
         P.w_torque * MX::sumsqr(u(casadi::Slice(1, 4)));
}

/// Common goal / cost handling. Derived classes provide the quaternion of a state.
template <class Base> class QuadrotorCost : public Base {
public:
  template <class... A>
  QuadrotorCost(const Params &P, A &&...args) : Base(std::forward<A>(args)...), P_(P) {
    // goal: p_goal (3), q_goal (4) as runtime parameters
    p_goal_ = this->parameter("p_goal", 3, 1);
    q_goal_ = this->parameter("q_goal", 4, 1);
    Eigen::VectorXd ulb(4), uub(4);
    ulb << 0, -P.torque_max, -P.torque_max, -P.torque_max;
    uub << P.thrust_max, P.torque_max, P.torque_max, P.torque_max;
    this->set_input_bound(ulb, uub);
  }

  MX stage_cost(MX x, MX u, size_t) override { return cost(x) + input_cost(u); }
  MX terminal_cost(MX x) override { return P_.terminal_scale * cost(x); }

  virtual MX position(const MX &x) = 0;
  virtual MX velocity(const MX &x) = 0;
  virtual MX quaternion(const MX &x) = 0; // unit quaternion of the state
  virtual MX rate(const MX &x) = 0;

protected:
  MX cost(const MX &x) {
    return state_cost(P_, position(x), velocity(x), quaternion(x), rate(x), p_goal_, q_goal_);
  }
  MX input_cost(const MX &u) { return quadrotor::input_cost(P_, u); }
  Params P_;
  MX p_goal_, q_goal_;
};

/// State [p; v; q; w] on R3 x R3 x SO3 x R3; dynamics return the tangent velocity.
class QuadrotorManifoldProb : public QuadrotorCost<Problem> {
public:
  static casadi::Manifold manifold() {
    auto R3 = casadi::Manifold::Euclidean(3);
    return casadi::Manifold::product({R3, R3, casadi::Manifold::SO3(), R3});
  }
  explicit QuadrotorManifoldProb(const Params &P = Params(),
                                 DynamicsType type = DynamicsType::ContinuesLieRK4)
      : QuadrotorCost<Problem>(P, type, manifold(), 4, P.horizon, P.dt) {}

  MX dynamics(MX x, MX u) override {
    MX q = x(casadi::Slice(6, 10)), w = x(casadi::Slice(10, 13));
    // tangent of R3 x R3 x SO3 x R3: [p'; v'; w (body); w']
    return MX::vertcat({x(casadi::Slice(3, 6)), lin_acc(P_, quat_to_rot(q), u), w,
                        ang_acc(P_, w, u)});
  }
  MX position(const MX &x) override { return x(casadi::Slice(0, 3)); }
  MX velocity(const MX &x) override { return x(casadi::Slice(3, 6)); }
  MX quaternion(const MX &x) override { return x(casadi::Slice(6, 10)); }
  MX rate(const MX &x) override { return x(casadi::Slice(10, 13)); }
};

/// State [p; v; q; w] in R13, q' = q * [0, w] / 2.
class QuadrotorQuatProbBase : public QuadrotorCost<Problem> {
public:
  explicit QuadrotorQuatProbBase(const Params &P)
      : QuadrotorCost<Problem>(P, DynamicsType::ContinuesRK4, 13, 4, P.horizon, P.dt) {}
  MX dynamics(MX x, MX u) override {
    MX q = x(casadi::Slice(6, 10)), w = x(casadi::Slice(10, 13));
    MX qdot = 0.5 * quat_mul(q, MX::vertcat({0, w}));
    return MX::vertcat({x(casadi::Slice(3, 6)), lin_acc(P_, quat_to_rot(q), u), qdot,
                        ang_acc(P_, w, u)});
  }
  MX position(const MX &x) override { return x(casadi::Slice(0, 3)); }
  MX velocity(const MX &x) override { return x(casadi::Slice(3, 6)); }
  MX rate(const MX &x) override { return x(casadi::Slice(10, 13)); }
};

/// Quaternion with the unit-norm equality constraint |q|^2 - 1 = 0 at every stage.
class QuadrotorQuatNormProb : public QuadrotorQuatProbBase {
public:
  explicit QuadrotorQuatNormProb(const Params &P = Params()) : QuadrotorQuatProbBase(P) {
    add_constraint(ConstraintType::Equality, [](MX x, MX) {
      MX q = x(casadi::Slice(6, 10));
      return MX::dot(q, q) - 1;
    });
  }
  MX quaternion(const MX &x) override { return x(casadi::Slice(6, 10)); }
};

/// Quaternion without a norm constraint; the cost uses the normalized quaternion.
class QuadrotorQuatFreeProb : public QuadrotorQuatProbBase {
public:
  explicit QuadrotorQuatFreeProb(const Params &P = Params()) : QuadrotorQuatProbBase(P) {}
  MX quaternion(const MX &x) override {
    MX q = x(casadi::Slice(6, 10));
    return q / MX::norm_2(q);
  }
};

/// State [p; v; (roll, pitch, yaw); w] in R12, ZYX Euler angles.
class QuadrotorEulerProb : public QuadrotorCost<Problem> {
public:
  explicit QuadrotorEulerProb(const Params &P = Params())
      : QuadrotorCost<Problem>(P, DynamicsType::ContinuesRK4, 12, 4, P.horizon, P.dt) {}
  MX dynamics(MX x, MX u) override {
    MX e = x(casadi::Slice(6, 9)), w = x(casadi::Slice(9, 12));
    MX phi = e(0), theta = e(1);
    // Euler angle rates from body rates (singular at cos(pitch) = 0)
    MX edot = MX::vertcat({w(0) + tan(theta) * (sin(phi) * w(1) + cos(phi) * w(2)),
                           cos(phi) * w(1) - sin(phi) * w(2),
                           (sin(phi) * w(1) + cos(phi) * w(2)) / cos(theta)});
    return MX::vertcat({x(casadi::Slice(3, 6)), lin_acc(P_, quat_to_rot(euler_to_quat(e)), u),
                        edot, ang_acc(P_, w, u)});
  }
  MX position(const MX &x) override { return x(casadi::Slice(0, 3)); }
  MX velocity(const MX &x) override { return x(casadi::Slice(3, 6)); }
  MX quaternion(const MX &x) override { return euler_to_quat(x(casadi::Slice(6, 9))); }
  MX rate(const MX &x) override { return x(casadi::Slice(9, 12)); }
};

} // namespace quadrotor
