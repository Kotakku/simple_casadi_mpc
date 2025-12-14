#include "double_integrator_problem.hpp"
#include "simple_casadi_mpc/simple_casadi_mpc.hpp"

int main() {
    using namespace simple_casadi_mpc;

    // solver name
    const std::string solver_name = "fatrop";

    // solver option
    auto solver_config = MPC::default_fatrop_config();

    // generated solver name
    const std::string export_solver_name = "double_integrator_solver";

    // output directory
    const std::string export_dir = "./";

    CompiledMPC::generate_code<DoubleIntegratorProb>(export_solver_name, export_dir, solver_name, solver_config);
}
