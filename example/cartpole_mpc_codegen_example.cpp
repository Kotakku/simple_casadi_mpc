#include "cartpole_problem.hpp"
#include "cartpole_solver_config.hpp"
#include "simple_casadi_mpc/simple_casadi_mpc.hpp"
#include <casadi/casadi.hpp>
#include <chrono>
#include <iostream>
#include <matplotlibcpp17/animation.h>
#include <matplotlibcpp17/patches.h>
#include <matplotlibcpp17/pyplot.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include <pybind11/embed.h>

using namespace pybind11::literals;

void animate(matplotlibcpp17::pyplot::PyPlot &plt, const std::vector<double> &x, const std::vector<double> &angle,
             const std::vector<double> &u, const std::vector<double> &target, const double dt) {
    (void)u;

    const double cart_width = 0.5;
    const double cart_height = 0.2;
    const double pole_length = 0.5;

    const double x_max = *std::max_element(x.begin(), x.end());
    const double x_min = *std::min_element(x.begin(), x.end());
    const double range_max = std::max(std::abs(x_max), std::abs(x_min)) + cart_width;

    const double frame_interval = 1.0 / 30.0;
    const size_t skip = std::ceil(frame_interval / dt);
    for (size_t i = 0; i < x.size(); i += skip) {
        plt.clf();

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

        plt.gca().set_aspect(pybind11::make_tuple(1.0));
        plt.plot(pybind11::make_tuple(target_x_data, target_y_data, "k--"));
        plt.plot(pybind11::make_tuple(cart_x_data, cart_y_data, "b-"));
        plt.plot(pybind11::make_tuple(pole_x_data, pole_y_data, "r-"));

        plt.xlim(pybind11::make_tuple(-range_max, range_max));
        plt.ylim(pybind11::make_tuple(-range_max, range_max));
        plt.pause(pybind11::make_tuple(0.01));
        // std::cout << i+1 << "/" << x.size() << std::endl;
    }
}

int main() {
    using namespace simple_casadi_mpc;
    std::cout << "cartpole mpc example" << std::endl;
    pybind11::scoped_interpreter guard{};
    auto plt = matplotlibcpp17::pyplot::import();
    auto prob = std::make_shared<CartpoleProb>();
    auto lib_config = get_cartpole_solver_compiled_library_options();
    CompiledMPC mpc(lib_config, prob);

    casadi::DMDict param_list;
    double target_pos = -0.5;
    param_list["x_ref"] = {target_pos, M_PI, 0, 0};

    Eigen::VectorXd x = Eigen::VectorXd::Zero(prob->nx());

    std::cout << "simulation" << std::endl;
    const double dt = 0.01;
    const size_t sim_len = 1200;
    std::vector<double> i_log(sim_len), target_log(sim_len), t_log(sim_len), x_log(sim_len), angle_log(sim_len), u_log(sim_len),
        dt_log(sim_len);

    auto t_all_start = std::chrono::system_clock::now();
    for (size_t i = 0; i < sim_len; i++) {
        if (i == 700) {
            // change target
            target_pos = 0.5;
            param_list["x_ref"] = {target_pos, M_PI, 0, 0};
        }

        auto t_start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd u = mpc.solve(x, param_list);
        auto t_end = std::chrono::high_resolution_clock::now();
        x = prob->simulate(x, u, dt);
        double solve_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() * 1e-6;
        // std::cout << "solve time: " << solve_time << std::endl;
        target_log[i] = target_pos;
        t_log[i] = i * dt;
        i_log[i] = i;
        x_log[i] = x[0];
        angle_log[i] = x[1];
        u_log[i] = u[0];
        dt_log[i] = solve_time * 1e3;
    }
    auto t_all_end = std::chrono::system_clock::now();
    double all_time = std::chrono::duration_cast<std::chrono::microseconds>(t_all_end - t_all_start).count() * 1e-6;
    std::cout << "all time: " << all_time << std::endl;

    plt.figure();
    plt.plot(pybind11::make_tuple(t_log, u_log), pybind11::dict("label"_a = "u"));
    plt.plot(pybind11::make_tuple(t_log, x_log), pybind11::dict("label"_a = "x"));
    plt.plot(pybind11::make_tuple(t_log, angle_log), pybind11::dict("label"_a = "angle"));
    plt.legend();
    plt.show();

    plt.figure();
    plt.plot(pybind11::make_tuple(i_log, dt_log));
    plt.xlabel(pybind11::make_tuple("iteration"));
    plt.ylabel(pybind11::make_tuple("MPC solve time [ms]"));
    plt.show();

    animate(plt, x_log, angle_log, u_log, target_log, dt);

    return 0;
}
