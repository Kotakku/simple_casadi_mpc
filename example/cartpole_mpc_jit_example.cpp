#include "simple_casadi_mpc/simple_casadi_mpc.hpp"
#include "thirdparty/matplotlib-cpp/matplotlibcpp.h"
#include <casadi/casadi.hpp>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>

class CartpoleProb : public simple_casadi_mpc::Problem {
public:
    CartpoleProb() : Problem(DynamicsType::ContinuesRK4, 4, 1, 30, 0.05) {
        using namespace casadi;
        x_ref = parameter("x_ref", 4, 1);
        Q = DM::diag({5, 10.0, 0.01, 0.01});
        R = DM::diag({0.1});
        Qf = DM::diag({10, 10.0, 0.01, 0.01});

        set_input_bound(Eigen::VectorXd::Constant(1, -15.0), Eigen::VectorXd::Constant(1, 15.0));
    }

    virtual casadi::MX dynamics(casadi::MX x, casadi::MX u) override {
        using namespace casadi;

        // auto y = x(0);     // cart の水平位置[m]
        auto th = x(1);  // pole の傾き角[rad]
        auto dy = x(2);  // cart の水平速度[m/s]
        auto dth = x(3); // pole の傾き角速度[rad/s]
        auto f = u(0);   // cart を押す力[N]（制御入力）
        // cart の水平加速度
        casadi::MX ddy = (f + mp * sin(th) * (l * dth * dth + g * cos(th))) / (mc + mp * sin(th) * sin(th));
        // pole の傾き角加速度
        casadi::MX ddth = (-f * cos(th) - mp * l * dth * dth * cos(th) * sin(th) - (mc + mp) * g * sin(th)) /
                          (l * (mc + mp * sin(th) * sin(th)));

        return vertcat(dy, dth, ddy, ddth);
    }

    virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u, size_t k) override {
        (void)k;
        using namespace casadi;
        MX L = 0;
        auto e = x - x_ref;
        L += 0.5 * mtimes(e.T(), mtimes(Q, e));
        L += 0.5 * mtimes(u.T(), mtimes(R, u));
        return dt() * L;
    }

    virtual casadi::MX terminal_cost(casadi::MX x) {
        using namespace casadi;
        auto e = x - x_ref;
        return 0.5 * mtimes(e.T(), mtimes(Qf, e));
    }

    const double mc = 2.0;
    const double mp = 0.2;
    const double l = 0.5;
    const double g = 9.8;

    casadi::MX x_ref;
    casadi::DM Q, R, Qf;
};

void animate(const std::vector<double> &x, const std::vector<double> &angle, const std::vector<double> &u, const std::vector<double> &target,
             size_t skip = 4) {
    (void)u;
    namespace plt = matplotlibcpp;

    double cart_width = 0.5;
    double cart_height = 0.2;
    double pole_length = 0.5;

    double x_max = *std::max_element(x.begin(), x.end());
    double x_min = *std::min_element(x.begin(), x.end());
    double range_max = std::max(std::abs(x_max), std::abs(x_min)) + cart_width;
    for (size_t i = 0; i < x.size(); i += skip) {
        plt::clf();

        double cart_x = x[i];
        double cart_y = 0.0;
        double pole_x = cart_x + pole_length * sin(angle[i]);
        double pole_y = cart_y - pole_length * cos(angle[i]);

        std::vector<double> cart_x_data = {cart_x - cart_width / 2, cart_x + cart_width / 2, cart_x + cart_width / 2,
                                           cart_x - cart_width / 2, cart_x - cart_width / 2};
        std::vector<double> cart_y_data = {cart_y - cart_height / 2, cart_y - cart_height / 2, cart_y + cart_height / 2,
                                           cart_y + cart_height / 2, cart_y - cart_height / 2};
        std::vector<double> pole_x_data = {cart_x, pole_x};
        std::vector<double> pole_y_data = {cart_y, pole_y};
        std::vector<double> target_x_data = {target[i], target[i]};
        std::vector<double> target_y_data = {-range_max, range_max};

        plt::set_aspect(1.0);
        plt::plot(target_x_data, target_y_data, "k--");
        plt::plot(cart_x_data, cart_y_data, "b-");
        plt::plot(pole_x_data, pole_y_data, "r-");

        plt::xlim(-range_max, range_max);
        plt::ylim(-range_max, range_max);
        plt::pause(0.01);
        // std::cout << i+1 << "/" << x.size() << std::endl;
    }
}

int main() {
    using namespace simple_casadi_mpc;
    std::cout << "Cartpole MPC vs JITMPC Comparison" << std::endl;
    auto prob = std::make_shared<CartpoleProb>();

    // Create both regular MPC and JITMPC
    // Use FATROP for better performance
    auto fatrop_config = MPC::default_fatrop_config();
    fatrop_config["fatrop.tol"] = 1e-2;
    fatrop_config["fatrop.acceptable_tol"] = 5e-2;
    fatrop_config["print_time"] = false;

    std::cout << "\n=== Creating Regular MPC ===" << std::endl;
    MPC mpc_regular(prob, "fatrop", fatrop_config);

    std::cout << "\n=== Creating JITMPC ===" << std::endl;
    JITMPC mpc_codegen("cartpole", prob, "fatrop", fatrop_config);

    casadi::DMDict param_list;
    double target_pos = -0.5;
    param_list["x_ref"] = {target_pos, M_PI, 0, 0};

    // Simulation for both MPCs
    const double dt = 0.01;
    const size_t sim_len = 600; // Shorter simulation for comparison

    std::cout << "\n=== Running Regular MPC ===" << std::endl;
    Eigen::VectorXd x_regular = Eigen::VectorXd::Zero(prob->nx());
    std::vector<double> i_log(sim_len), target_log(sim_len), t_log(sim_len);
    std::vector<double> x_log_regular(sim_len), angle_log_regular(sim_len), u_log_regular(sim_len);
    std::vector<double> dt_log_regular(sim_len);

    auto t_all_start = std::chrono::system_clock::now();
    for (size_t i = 0; i < sim_len; i++) {
        if (i == 300) {
            target_pos = 0.5;
            param_list["x_ref"] = {target_pos, M_PI, 0, 0};
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd u = mpc_regular.solve(x_regular, param_list);
        auto t_end = std::chrono::high_resolution_clock::now();
        x_regular = prob->simulate(x_regular, u, dt);
        double solve_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() * 1e-6;

        target_log[i] = target_pos;
        t_log[i] = i * dt;
        i_log[i] = i;
        x_log_regular[i] = x_regular[0];
        angle_log_regular[i] = x_regular[1];
        u_log_regular[i] = u[0];
        dt_log_regular[i] = solve_time * 1e3;
    }
    auto t_all_end = std::chrono::system_clock::now();
    double regular_time = std::chrono::duration_cast<std::chrono::microseconds>(t_all_end - t_all_start).count() * 1e-6;
    std::cout << "Regular MPC total time: " << regular_time << " sec" << std::endl;
    std::cout << "Regular MPC average: " << regular_time / sim_len * 1000 << " ms/iter" << std::endl;

    std::cout << "\n=== Running JITMPC ===" << std::endl;
    std::cout << "First solve will generate and compile code..." << std::endl;

    target_pos = -0.5; // Reset target
    param_list["x_ref"] = {target_pos, M_PI, 0, 0};
    Eigen::VectorXd x_codegen = Eigen::VectorXd::Zero(prob->nx());
    std::vector<double> x_log_codegen(sim_len), angle_log_codegen(sim_len), u_log_codegen(sim_len);
    std::vector<double> dt_log_codegen(sim_len);

    t_all_start = std::chrono::system_clock::now();
    for (size_t i = 0; i < sim_len; i++) {
        if (i == 300) {
            target_pos = 0.5;
            param_list["x_ref"] = {target_pos, M_PI, 0, 0};
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd u = mpc_codegen.solve(x_codegen, param_list);
        auto t_end = std::chrono::high_resolution_clock::now();
        x_codegen = prob->simulate(x_codegen, u, dt);
        double solve_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() * 1e-6;

        if (i == 0) {
            std::cout << "First solve (with code generation): " << solve_time * 1000 << " ms" << std::endl;
        } else if (i == 1) {
            std::cout << "Second solve (using generated code): " << solve_time * 1000 << " ms" << std::endl;
        }

        x_log_codegen[i] = x_codegen[0];
        angle_log_codegen[i] = x_codegen[1];
        u_log_codegen[i] = u[0];
        dt_log_codegen[i] = solve_time * 1e3;
    }
    t_all_end = std::chrono::system_clock::now();
    double codegen_time = std::chrono::duration_cast<std::chrono::microseconds>(t_all_end - t_all_start).count() * 1e-6;
    std::cout << "JITMPC total time: " << codegen_time << " sec" << std::endl;
    std::cout << "JITMPC average: " << codegen_time / sim_len * 1000 << " ms/iter" << std::endl;
    std::cout << "\nSpeedup factor: " << regular_time / codegen_time << "x" << std::endl;

    namespace plt = matplotlibcpp;

    // Plot state trajectories
    plt::figure();
    plt::subplot(3, 1, 1);
    plt::named_plot("Regular MPC", t_log, x_log_regular);
    plt::named_plot("JITMPC", t_log, x_log_codegen);
    plt::ylabel("Position [m]");
    plt::legend();
    plt::grid(true);

    plt::subplot(3, 1, 2);
    plt::named_plot("Regular MPC", t_log, angle_log_regular);
    plt::named_plot("JITMPC", t_log, angle_log_codegen);
    plt::ylabel("Angle [rad]");
    plt::legend();
    plt::grid(true);

    plt::subplot(3, 1, 3);
    plt::named_plot("Regular MPC", t_log, u_log_regular);
    plt::named_plot("JITMPC", t_log, u_log_codegen);
    plt::xlabel("Time [s]");
    plt::ylabel("Control [N]");
    plt::legend();
    plt::grid(true);

    plt::suptitle("State Trajectories Comparison");
    plt::show();

    // Plot timing comparison
    plt::figure();
    plt::named_plot("MPC", i_log, dt_log_regular, "b-");
    plt::named_plot("MPC (with CodeGen)", i_log, dt_log_codegen, "r-");
    plt::xlabel("Iteration");
    plt::ylabel("Solve time [ms]");
    plt::title("MPC Performance Comparison");
    plt::legend();
    plt::ylim(0.0, *std::max_element(dt_log_regular.begin(), dt_log_regular.end()) * 1.1);
    plt::grid(true);

    // Add average lines
    double avg_regular = std::accumulate(dt_log_regular.begin() + 10, dt_log_regular.end(), 0.0) / (sim_len - 10);
    double avg_codegen = std::accumulate(dt_log_codegen.begin() + 10, dt_log_codegen.end(), 0.0) / (sim_len - 10);
    std::vector<double> avg_regular_line(sim_len, avg_regular);
    std::vector<double> avg_codegen_line(sim_len, avg_codegen);

    // Add text annotations
    std::string text_regular = "Avg Regular: " + std::to_string(avg_regular).substr(0, 5) + " ms";
    std::string text_codegen = "Avg CodeGen: " + std::to_string(avg_codegen).substr(0, 5) + " ms";
    std::string text_speedup = "Speedup: " + std::to_string(avg_regular / avg_codegen).substr(0, 4) + "x";

    plt::text(sim_len * 0.6, avg_regular * 1.1, text_regular);
    plt::text(sim_len * 0.6, avg_codegen * 1.1, text_codegen);
    plt::text(sim_len * 0.6, avg_regular + (avg_regular + avg_codegen) / 2, text_speedup);

    plt::show();

    // Animate only JITMPC result
    animate(x_log_codegen, angle_log_codegen, u_log_codegen, target_log);

    return 0;
}