#include "cartpole_problem.hpp"
#include "simple_casadi_mpc/simple_casadi_mpc.hpp"

int main() {
  using namespace simple_casadi_mpc;

  // solver name
  const std::string solver_name = "fatrop";

  // solver option
  auto solver_config = mpc_config::default_fatrop_config();
  solver_config["fatrop.tol"] = 1e-4;
  solver_config["fatrop.acceptable_tol"] = 5e-4;

  // generated solver name
  const std::string export_solver_name = "cartpole_solver";

  // output directory
  const std::string export_dir = "./";

  CompiledMPC<casadi::MX>::generate_code<CartpoleProb>(export_solver_name, export_dir, solver_name,
                                                       solver_config);
}
