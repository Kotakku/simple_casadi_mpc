#include "simple_casadi_mpc/simple_casadi_mpc.hpp"
#include "thirdparty/matplotlib-cpp/matplotlibcpp.h"
#include <casadi/casadi.hpp>
#include <chrono>
#include <iostream>
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
    std::cout << "cartpole mpc example" << std::endl;
    auto prob = std::make_shared<CartpoleProb>();
    MPC mpc(prob);

    // MUMPSじゃなくてHSLのMA97とかを使うと速くなる
    // auto ipopt_dict = MPC::default_ipopt_config();
    // ipopt_dict["ipopt.linear_solver"] = "ma27";
    // MPC mpc(prob, "ipopt", ipopt_dict);

    // FATROPを使用する場合
    // auto fatrop_config = MPC::default_fatrop_config();
    // MPC mpc(prob, "fatrop", fatrop_config);

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

    namespace plt = matplotlibcpp;
    plt::figure();
    plt::named_plot("u", t_log, u_log);
    plt::named_plot("x", t_log, x_log);
    plt::named_plot("angle", t_log, angle_log);
    plt::legend();
    plt::show();

    plt::figure();
    plt::plot(i_log, dt_log);
    plt::xlabel("iteration");
    plt::ylabel("MPC solve time [ms]");
    plt::show();

    animate(x_log, angle_log, u_log, target_log);

    return 0;
}