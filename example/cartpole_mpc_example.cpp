#include <iostream>
#include <string>
#include <casadi/casadi.hpp>
#include "simple_casadi_mpc.hpp"
#include "thirdparty/matplotlib-cpp/matplotlibcpp.h"

class CartpoleProb : public simple_casadi_mpc::Problem
{
public:
    using LUbound = Problem::LUbound;
    CartpoleProb():
        Problem(4, 1, 20, 0.05)
    {
        using namespace casadi;
        x_ref = {0, M_PI, 0, 0};
        Q = DM::diag({2.5, 10.0, 0.01, 0.01});
        R = DM::diag({0.1});
        Qf = DM::diag({2.5, 10.0, 0.01, 0.01});
    }

    casadi::MX dynamics(casadi::MX x, casadi::MX u)
    {
        using namespace casadi;
        
        auto y = x(0);     // cart の水平位置[m]
        auto th = x(1);    // pole の傾き角[rad]
        auto dy = x(2);    // cart の水平速度[m/s]
        auto dth = x(3);   // pole の傾き角速度[rad/s]
        auto f = u(0);     // cart を押す力[N]（制御入力）
        // cart の水平加速度
        casadi::MX ddy = (f+mp*sin(th)*(l*dth*dth+g*cos(th))) / (mc+mp*sin(th)*sin(th));
        // pole の傾き角加速度
        casadi::MX ddth = (-f*cos(th)-mp*l*dth*dth*cos(th)*sin(th)-(mc+mp)*g*sin(th)) / (l * (mc+mp*sin(th)*sin(th)));
        
        return vertcat(dy, dth, ddy, ddth);
    }

    virtual casadi::MX discretized_dynamics(casadi::MX x, casadi::MX u) override
    {
        auto dynamics = std::bind(&CartpoleProb::dynamics, this, std::placeholders::_1, std::placeholders::_2);
        return simple_casadi_mpc::integrate_dynamics_rk4<casadi::MX>(dt(), x, u, dynamics);
    }

    // シミュレーション用
    Eigen::VectorXd discretized_dynamics_sim(double dt, Eigen::VectorXd x, Eigen::VectorXd u)
    {
        auto dynamics = [&](Eigen::VectorXd x, Eigen::VectorXd u) -> Eigen::VectorXd
        {
            auto y = x(0);     // cart の水平位置[m]
            auto th = x(1);    // pole の傾き角[rad]
            auto dy = x(2);    // cart の水平速度[m/s]
            auto dth = x(3);   // pole の傾き角速度[rad/s]
            auto f = u(0);     // cart を押す力[N]（制御入力）
            // cart の水平加速度
            double ddy = (f+mp*sin(th)*(l*dth*dth+g*cos(th))) / (mc+mp*sin(th)*sin(th));
            // pole の傾き角加速度
            double ddth = (-f*cos(th)-mp*l*dth*dth*cos(th)*sin(th)-(mc+mp)*g*sin(th)) / (l * (mc+mp*sin(th)*sin(th)));  

            return (Eigen::VectorXd(4) << dy, dth, ddy, ddth).finished();
        };

        return simple_casadi_mpc::integrate_dynamics_rk4<Eigen::VectorXd>(dt, x, u, dynamics);
    }

    virtual std::vector<LUbound> u_bounds()
    {
        Eigen::VectorXd ub = Eigen::VectorXd::Constant(nu(), 15.0);
        Eigen::VectorXd lb = -ub;
        
        return std::vector<LUbound>(horizon(), {lb, ub});
    }

    virtual std::vector<LUbound> x_bounds()
    {
        double inf = std::numeric_limits<double>::infinity();
        Eigen::VectorXd ub = Eigen::VectorXd::Constant(nx(), inf);
        Eigen::VectorXd lb = -ub;

        return std::vector<LUbound>{horizon(), {lb, ub}};
    }

    virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u) override
    {
        using namespace casadi;
        MX L = 0;
        auto e = x - x_ref;
        L += 0.5 * mtimes(e.T(), mtimes(Q, e));
        L += 0.5 * mtimes(u.T(), mtimes(R, u));
        return dt() * L;
    }

    virtual casadi::MX terminal_cost(casadi::MX x)
    {
        using namespace casadi;
        auto e = x - x_ref;
        return 0.5 * mtimes(e.T(), mtimes(Qf, e));
    }

    const double mc = 2.0;
    const double mp = 0.2;
    const double l = 0.5;
    const double g = 9.8;

    casadi::DM x_ref;
    casadi::DM Q, R, Qf;
};

int main() {
    using namespace simple_casadi_mpc;
    std::cout << "cartpole mpc example" << std::endl;
    auto prob = std::make_shared<CartpoleProb>();
    // MPC ocp(prob); // デフォルトではIPOPTを使う
    MPC ocp(prob, "sqpmethod", MPC::default_qpoases_config()); // sqpmethod + qpOASESで解く
    
    Eigen::VectorXd x = Eigen::VectorXd::Constant(prob->nx(), 0.0);

    std::cout << "simulation" << x << std::endl;
    const double dt = 0.01;
    const size_t sim_len = 1000;
    std::vector<double> t_log(sim_len), x_log(sim_len), angle_log(sim_len), u_log(sim_len);
    
    for(size_t i = 0; i < sim_len; i++)
    {
        Eigen::VectorXd u = ocp.solve(x);
        x = prob->discretized_dynamics_sim(dt, x, u);
        std::cout << u << ", " << x[1] << std::endl; // 操作量とpoleの傾き角
        t_log[i] = i * dt;
        x_log[i] = x[0];
        angle_log[i] = x[1];
        u_log[i] = u[0];
    }

    namespace plt = matplotlibcpp;
    plt::figure();
    plt::plot(t_log, u_log);
    plt::plot(t_log, x_log);
    plt::plot(t_log, angle_log);
    plt::show();

    return 0;
}