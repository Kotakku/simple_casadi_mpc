#include <iostream>
#include <string>
#include <filesystem>
#include <casadi/casadi.hpp>
#include "simple_casadi_mpc.hpp"
#include "thirdparty/matplotlib-cpp/matplotlibcpp.h"

class DoubleIntegratorProb : public simple_casadi_mpc::Problem
{
public:
    DoubleIntegratorProb():
        Problem(DynamicsType::ContinuesRK4, 2, 1, 20, 0.05)
    {
        set_input_bound(Eigen::VectorXd::Constant(1, -1.0), Eigen::VectorXd::Constant(1, 1.0));
    }

    virtual casadi::MX dynamics(casadi::MX x, casadi::MX u) override
    {
        return vertcat(x(1), u);
    }

    // シミュレーション用
    Eigen::VectorXd discretized_dynamics_sim(double dt, Eigen::VectorXd x, Eigen::VectorXd u)
    {
        auto dynamics = [&](Eigen::VectorXd x, Eigen::VectorXd u) -> Eigen::VectorXd
        {
            return (Eigen::VectorXd(2) << x(1), u(0)).finished();
        };
        return simple_casadi_mpc::integrate_dynamics_rk4<Eigen::VectorXd>(dt, x, u, dynamics);
    }

    virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u) override
    {
        return mtimes(x.T(), x);
    }

    virtual casadi::MX terminal_cost(casadi::MX x)
    {
        return mtimes(x.T(), x);
    }
};

int main() {
    using namespace simple_casadi_mpc;
    std::cout << "double integrator mpc example" << std::endl;
    auto prob = std::make_shared<DoubleIntegratorProb>();
    MPC mpc(prob);
    
    Eigen::VectorXd x = Eigen::VectorXd::Constant(prob->nx(), 1.0);

    std::cout << "simulation" << x << std::endl;
    const double dt = 0.01;
    const size_t sim_len = 1000;
    std::vector<double> t_log(sim_len), x0_log(sim_len), x1_log(sim_len), u_log(sim_len);
    for(size_t i = 0; i < sim_len; i++)
    {
        Eigen::VectorXd u = mpc.solve(x);
        x = prob->discretized_dynamics_sim(dt, x, u);
        std::cout << u << ", " << x.transpose() << std::endl;
        t_log[i] = i * dt;
        x0_log[i] = x[0];
        x1_log[i] = x[1];
        u_log[i] = u[0];
    }

    // plot
    namespace plt = matplotlibcpp;
    plt::figure();
    plt::named_plot("u", t_log, u_log);
    plt::named_plot("vel", t_log, x0_log);
    plt::named_plot("pos", t_log, x1_log);
    plt::legend();
    plt::show();

    return 0;
}