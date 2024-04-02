#include "simple_casadi_mpc.hpp"
#include "thirdparty/matplotlib-cpp/matplotlibcpp.h"
#include <casadi/casadi.hpp>
#include <iostream>
#include <string>

class DoubleIntegratorProb : public simple_casadi_mpc::Problem {
public:
    DoubleIntegratorProb() : Problem(DynamicsType::ContinuesRK4, 2, 1, 20, 0.05) {
        set_input_bound(Eigen::VectorXd::Constant(1, -1.0), Eigen::VectorXd::Constant(1, 1.0));
    }

    virtual casadi::MX dynamics(casadi::MX x, casadi::MX u) override {
        (void)u;
        return vertcat(x(1), u);
    }

    virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u) override {
        (void)u;
        return mtimes(x.T(), x);
    }

    virtual casadi::MX terminal_cost(casadi::MX x) { return mtimes(x.T(), x); }
};

int main() {
    using namespace simple_casadi_mpc;

    std::cout << "double integrator mpc example" << std::endl;
    auto prob = std::make_shared<DoubleIntegratorProb>();
    MPC mpc(prob);

    Eigen::VectorXd x = Eigen::VectorXd::Constant(prob->nx(), 1.0);

    std::cout << "simulation" << std::endl;
    const double dt = 0.01;
    const size_t sim_len = 1000;
    std::vector<double> t_log(sim_len), x_log(sim_len), v_log(sim_len), u_log(sim_len);
    for (size_t i = 0; i < sim_len; i++) {
        Eigen::VectorXd u = mpc.solve(x);
        x = prob->simulate(x, u, dt);
        // std::cout << u << ", " << x.transpose() << std::endl;
        t_log[i] = i * dt;
        x_log[i] = x[0];
        v_log[i] = x[1];
        u_log[i] = u[0];
    }

    // plot
    namespace plt = matplotlibcpp;
    plt::figure();
    plt::named_plot("u", t_log, u_log);
    plt::named_plot("pos", t_log, x_log);
    plt::named_plot("vel", t_log, v_log);
    plt::legend();
    plt::show();

    return 0;
}