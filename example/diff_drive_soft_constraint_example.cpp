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

class DiffDriveSoftProb : public simple_casadi_mpc::Problem {
public:
  // soft=true で障害物制約を soft_add_constraint で追加する。
  DiffDriveSoftProb(bool soft, double w1, double w2)
      : Problem(DynamicsType::ContinuesRK4, 5, 2, 40, 0.1) {
    using namespace casadi;
    x_ref = parameter("x_ref", 5, 1);

    Q = DM::diag({10, 10, 6, 0.5, 0.1});
    R = DM::diag({0.01, 0.01});
    Qf = DM::diag({10, 10, 6, 0.5, 0.1});

    Eigen::VectorXd u_ub = (Eigen::VectorXd(2) << 2.0, 2.0).finished();
    set_input_bound(-u_ub, u_ub);
    Eigen::VectorXd x_ub = (Eigen::VectorXd(5) << inf, inf, inf, 2.0, 1.5).finished();
    set_state_bound(-x_ub, x_ub);

    auto obs =
        std::bind(&DiffDriveSoftProb::obstacle, this, std::placeholders::_1, std::placeholders::_2);
    if (soft) {
      soft_add_constraint(ConstraintType::Inequality, obs, w1, w2);
    } else {
      add_constraint(ConstraintType::Inequality, obs);
    }
  }

  virtual casadi::MX dynamics(casadi::MX x, casadi::MX u) override {
    using namespace casadi;
    auto theta = x(2);
    auto v = x(3);
    auto omega = x(4);
    return vertcat(v * cos(theta), v * sin(theta), omega, u(0), u(1));
  }

  // 直線経路にわずかに被る位置に障害物を 1 つ置く。hard なら避けるが、soft の重みが
  // 小さければ突っ切るほうが最適になる、という挙動差が見える。
  // (xy - center)^T (xy - center) >= radius^2 を g(x,u) <= 0 形式で表現
  static constexpr double obs_cx = 0.0;
  static constexpr double obs_cy = 0.15;
  static constexpr double obs_r = 0.3;

  casadi::MX obstacle(casadi::MX x, casadi::MX u) {
    (void)u;
    using namespace casadi;
    MX xy = x(Slice(0, 2));
    DM center = DM::zeros(2);
    center(0) = obs_cx;
    center(1) = obs_cy;
    return -(mtimes((xy - center).T(), (xy - center)) - obs_r * obs_r);
  }

  virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u, size_t k) override {
    (void)k;
    using namespace casadi;
    MX e = x - x_ref;
    return dt() * (0.5 * mtimes(e.T(), mtimes(Q, e)) + 0.5 * mtimes(u.T(), mtimes(R, u)));
  }

  virtual casadi::MX terminal_cost(casadi::MX x) override {
    using namespace casadi;
    MX e = x - x_ref;
    return 0.5 * mtimes(e.T(), mtimes(Qf, e));
  }

  casadi::MX x_ref;
  casadi::DM Q, R, Qf;
};

struct Trajectory {
  std::vector<double> x, y;
  double min_clearance = std::numeric_limits<double>::infinity();
};

// 障害物中心からの距離 - radius の最小値（負だと衝突）
static double clearance(double x, double y, double cx, double cy, double radius) {
  return std::hypot(x - cx, y - cy) - radius;
}

static Trajectory rollout(simple_casadi_mpc::MPC &mpc, simple_casadi_mpc::Problem &prob,
                          const casadi::DMDict &param_list, Eigen::VectorXd x0, size_t sim_len,
                          double dt) {
  Trajectory traj;
  traj.x.reserve(sim_len);
  traj.y.reserve(sim_len);
  for (size_t i = 0; i < sim_len; ++i) {
    Eigen::VectorXd u = mpc.solve(x0, param_list);
    x0 = prob.simulate(x0, u, dt);
    traj.x.push_back(x0[0]);
    traj.y.push_back(x0[1]);
    traj.min_clearance = std::min(traj.min_clearance,
                                  clearance(x0[0], x0[1], DiffDriveSoftProb::obs_cx,
                                            DiffDriveSoftProb::obs_cy, DiffDriveSoftProb::obs_r));
  }
  return traj;
}

int main() {
  using namespace simple_casadi_mpc;
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();

  casadi::DMDict param_list;
  // 2 つの障害物のあいだを通り抜ける必要があるゴール
  param_list["x_ref"] = std::vector<double>{1, 0.0, 0, 0, 0};
  Eigen::VectorXd x0(5);
  x0 << -1, 0.0, 0, 0, 0;

  const double dt = 0.05;
  const size_t sim_len = 80;

  auto run = [&](const std::string &label, bool soft, double w1, double w2) {
    auto prob = std::make_shared<DiffDriveSoftProb>(soft, w1, w2);
    MPC mpc(prob);
    auto t0 = std::chrono::steady_clock::now();
    auto traj = rollout(mpc, *prob, param_list, x0, sim_len, dt);
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() * 1e-3;
    std::cout << "[" << label << "] min_clearance=" << traj.min_clearance << " final=("
              << traj.x.back() << ", " << traj.y.back() << ")"
              << " elapsed=" << sec << " s" << std::endl;
    return traj;
  };

  // hard: 障害物を確実に避けて遠回りする
  auto traj_hard = run("hard", false, 0.0, 0.0);
  // soft (重み大): 実質 hard と同じ挙動
  auto traj_soft_high = run("soft w1=1e4", true, 1e4, 0.0);
  // soft (重み小): 障害物を切り抜けるルートが最適になるはず
  auto traj_soft_low = run("soft w1=1e0", true, 1e0, 0.0);

  // Plot
  plt.figure();
  plt.gca().set_aspect(pybind11::make_tuple(1.0));
  auto draw_circle = [&](double cx, double cy, double r) {
    std::vector<double> xs(64), ys(64);
    for (size_t i = 0; i < xs.size(); ++i) {
      double a = 2 * M_PI * i / (xs.size() - 1);
      xs[i] = cx + r * std::cos(a);
      ys[i] = cy + r * std::sin(a);
    }
    plt.plot(pybind11::make_tuple(xs, ys, "k-"));
  };
  draw_circle(DiffDriveSoftProb::obs_cx, DiffDriveSoftProb::obs_cy, DiffDriveSoftProb::obs_r);
  plt.plot(pybind11::make_tuple(traj_hard.x, traj_hard.y), pybind11::dict("label"_a = "hard"));
  plt.plot(pybind11::make_tuple(traj_soft_high.x, traj_soft_high.y),
           pybind11::dict("label"_a = "soft w1=1e4"));
  plt.plot(pybind11::make_tuple(traj_soft_low.x, traj_soft_low.y),
           pybind11::dict("label"_a = "soft w1=1e0"));
  plt.legend();
  plt.show();
  return 0;
}
