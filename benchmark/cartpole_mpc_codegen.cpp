#include "cartpole_problem.hpp"
#include "simple_casadi_mpc/simple_casadi_mpc.hpp"

int main() {
  using namespace simple_casadi_mpc;

  auto prob = std::make_shared<CartpoleProb>();

  // solver name
  const std::string solver_name = "fatrop";

  // solver option
  auto solver_config = mpc_config::default_fatrop_config();

  // generated solver name
  const std::string export_solver_name = "cartpole_solver_bench";

  // output directory
  const std::string export_dir = "./";

  generate_compiled_mpc_code(prob, export_solver_name, export_dir, solver_name, solver_config);
}
