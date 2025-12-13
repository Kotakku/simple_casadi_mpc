#include "double_integrator_problem.hpp"
#include "double_integrator_solver_config.hpp"
#include <casadi/casadi.hpp>
#include <iostream>
#include <string>

int main() {
    using namespace simple_casadi_mpc;

    auto prob = std::make_shared<DoubleIntegratorProb>();

    auto solver_config = get_double_integrator_solver_compiled_library_options();
    std::cout << "Loading pre-compiled solver from: " << solver_config.shared_library_path << std::endl;
    CompiledMPC mpc(solver_config, prob);

    Eigen::VectorXd x = Eigen::VectorXd::Constant(prob->nx(), 1.0);

    const double dt = 0.01;
    const size_t sim_len = 1000;

    std::cout << "Simulating with compiled solver..." << std::endl;
    for (size_t i = 0; i < sim_len; i++) {
        Eigen::VectorXd u = mpc.solve(x);
        x = prob->simulate(x, u, dt);
        if (i % 50 == 0) {
            std::cout << "step " << i << ": u=" << u.transpose() << " x=" << x.transpose() << std::endl;
        }
    }

    std::cout << "Final state: " << x.transpose() << std::endl;
    return 0;
}
