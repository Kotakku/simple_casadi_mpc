#include "simple_casadi_mpc.hpp"
#include "thirdparty/matplotlib-cpp/matplotlibcpp.h"
#include <casadi/casadi.hpp>
#include <chrono>
#include <iostream>
#include <string>

class DiffDriveProb : public simple_casadi_mpc::Problem {
public:
    DiffDriveProb() : Problem(DynamicsType::ContinuesRK4, 5, 2, 40, 0.1) {
        using namespace casadi;
        x_ref = parameter("x_ref", 5, 1);

        Q = DM::diag({10, 10, 6, 0.5, 0.1});
        R = DM::diag({0.01, 0.01});
        Qf = DM::diag({10, 10, 6, 0.5, 0.1});

        // 入力のボックス制約(加速度制約相当)
        Eigen::VectorXd u_ub = (Eigen::VectorXd(2) << 2.0, 2.0).finished();
        Eigen::VectorXd u_lb = -u_ub;
        set_input_bound(u_lb, u_ub);

        // 状態料のボックス制約(速度制約)
        Eigen::VectorXd x_ub = (Eigen::VectorXd(5) << inf, inf, inf, 2.0, 1.5).finished();
        Eigen::VectorXd x_lb = -x_ub;
        set_state_bound(x_lb, x_ub);

        //
        add_constraint(ConstraintType::Inequality,
                       std::bind(&DiffDriveProb::obstacle1, this, std::placeholders::_1, std::placeholders::_2));
        add_constraint(ConstraintType::Inequality,
                       std::bind(&DiffDriveProb::obstacle2, this, std::placeholders::_1, std::placeholders::_2));
    }

    virtual casadi::MX dynamics(casadi::MX x, casadi::MX u) override {
        using namespace casadi;

        auto lacc = u(0);
        auto racc = u(1);
        // auto _x = x(0);
        // auto _y = x(1);
        auto theta = x(2);
        auto v = x(3);
        auto omega = x(4);
        auto vx = v * cos(theta);
        auto vy = v * sin(theta);

        return vertcat(vx, vy, omega, lacc, racc);
    }

    casadi::MX obstacle1(casadi::MX x, casadi::MX u) {
        (void)u;
        using namespace casadi;
        casadi::MX xy = x(Slice(0, 2));
        casadi::DM center = casadi::DM::zeros(2);
        center(0) = 0.0;
        center(1) = 0.5;
        casadi::MX radius = 0.4;

        return -(mtimes((xy - center).T(), (xy - center)) - radius * radius);
    }

    casadi::MX obstacle2(casadi::MX x, casadi::MX u) {
        (void)u;
        using namespace casadi;
        casadi::MX xy = x(Slice(0, 2));
        casadi::DM center = casadi::DM::zeros(2);
        center(0) = 0.0;
        center(1) = -0.5;
        casadi::MX radius = 0.4;

        return -(mtimes((xy - center).T(), (xy - center)) - radius * radius);
    }

    virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u) override {
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

    casadi::MX x_ref;
    casadi::DM Q, R, Qf;
};

std::tuple<std::vector<double>, std::vector<double>> get_rectangle_vertices(double x, double y, double theta,
                                                                            double width, double height) {
    double x1 = x + cos(theta) * (-width / 2) - sin(theta) * (-height / 2);
    double y1 = y + sin(theta) * (-width / 2) + cos(theta) * (-height / 2);

    double x2 = x + cos(theta) * (width / 2) - sin(theta) * (-height / 2);
    double y2 = y + sin(theta) * (width / 2) + cos(theta) * (-height / 2);

    double x3 = x + cos(theta) * (width / 2) - sin(theta) * (height / 2);
    double y3 = y + sin(theta) * (width / 2) + cos(theta) * (height / 2);

    double x4 = x + cos(theta) * (-width / 2) - sin(theta) * (height / 2);
    double y4 = y + sin(theta) * (-width / 2) + cos(theta) * (height / 2);

    std::vector<double> x_data = {x1, x2, x3, x4, x1};
    std::vector<double> y_data = {y1, y2, y3, y4, y1};

    return {x_data, y_data};
}

void draw_circle(double x, double y, double radius) {
    namespace plt = matplotlibcpp;
    std::vector<double> x_data(100), y_data(100);
    for (size_t i = 0; i < 100; i++) {
        x_data[i] = radius * cos(2 * M_PI / 100 * i) + x;
        y_data[i] = radius * sin(2 * M_PI / 100 * i) + y;
    }
    plt::plot(x_data, y_data, "k-");
}

void animate(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &theta,
             Eigen::Vector3d start, Eigen::Vector3d goal, size_t skip = 4) {
    namespace plt = matplotlibcpp;
    for (size_t i = 0; i < x.size(); i += skip) {
        plt::clf();
        auto [paint_x_data, paint_y_data] = get_rectangle_vertices(x[i], y[i], theta[i], 0.2, 0.3);
        plt::set_aspect(1.0);
        plt::plot(paint_x_data, paint_y_data, "b-");

        std::vector<double> dir_x_data = {x[i], x[i] + cos(theta[i]) * 0.2};
        std::vector<double> dir_y_data = {y[i], y[i] + sin(theta[i]) * 0.2};
        plt::plot(dir_x_data, dir_y_data, "r-");

        double arrow_length = 0.3;
        // plt::arrow(start(0), start(1), arrow_length*cos(start(2)), arrow_length*sin(start(2)), "k", "k", 0.1, 0.1);
        // plt::arrow(goal(0), goal(1), arrow_length*cos(goal(2)), arrow_length*sin(goal(2)), "k", "k", 0.1, 0.1);

        draw_circle(0, 0.5, 0.2);
        draw_circle(0, -0.5, 0.2);

        double x_max = *std::max_element(x.begin(), x.end());
        double x_min = *std::min_element(x.begin(), x.end());
        double y_max = *std::max_element(y.begin(), y.end());
        double y_min = *std::min_element(y.begin(), y.end());
        double range_max_x = std::max(std::abs(x_max), std::abs(x_min));
        double range_max_y = std::max(std::abs(y_max), std::abs(y_min));
        double range_max = std::max(range_max_x, range_max_y) + 0.5;
        plt::xlim(-range_max, range_max);
        plt::ylim(-range_max, range_max);
        plt::pause(0.01);
        // std::cout << i+1 << "/" << x.size() << std::endl;
    }
}

int main() {
    using namespace simple_casadi_mpc;
    std::cout << "diff drive mpc example" << std::endl;
    auto prob = std::make_shared<DiffDriveProb>();
    MPC mpc(prob);

    casadi::DMDict param_list;
    param_list["x_ref"] = {1, -1.0, -M_PI / 2, 0, 0};

    Eigen::VectorXd x(5);
    x << -1, 1, 0, 0, 0;

    std::cout << "simulation" << std::endl;
    const double dt = 0.01;
    const size_t sim_len = 600;
    std::vector<double> i_log(sim_len), t_log(sim_len), x_log(sim_len), y_log(sim_len), theta_log(sim_len),
        v_log(sim_len), omega_log(sim_len), dt_log(sim_len);

    auto t_all_start = std::chrono::system_clock::now();
    for (size_t i = 0; i < sim_len; i++) {
        auto t_start = std::chrono::system_clock::now();
        Eigen::VectorXd u = mpc.solve(x, param_list);
        auto t_end = std::chrono::system_clock::now();
        double solve_time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() * 1e-6;
        // std::cout << "solve time: " << solve_time << std::endl;

        x = prob->simulate(x, u, dt);
        t_log[i] = i * dt;
        i_log[i] = i;
        x_log[i] = x[0];
        y_log[i] = x[1];
        theta_log[i] = x[2];
        v_log[i] = x[3];
        omega_log[i] = x[4];
        dt_log[i] = solve_time * 1e3;
    }
    auto t_all_end = std::chrono::system_clock::now();
    double all_time = std::chrono::duration_cast<std::chrono::microseconds>(t_all_end - t_all_start).count() * 1e-6;
    std::cout << "all time: " << all_time << std::endl;

    namespace plt = matplotlibcpp;
    plt::figure();
    plt::named_plot("x", t_log, x_log);
    plt::named_plot("y", t_log, y_log);
    plt::named_plot("theta", t_log, theta_log);
    plt::legend();
    plt::show();

    plt::figure();
    plt::named_plot("vel", t_log, v_log);
    plt::named_plot("omega", t_log, omega_log);
    plt::legend();
    plt::show();

    plt::figure();
    plt::plot(i_log, dt_log);
    plt::xlabel("iteration");
    plt::ylabel("MPC solve time [ms]");
    plt::show();

    plt::figure();
    Eigen::Vector3d start(-1, 0, 0);
    Eigen::Vector3d goal(1, 0, 0);
    animate(x_log, y_log, theta_log, start, goal);

    return 0;
}