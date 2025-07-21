#pragma once
#include "casadi_utils.hpp"
#include <Eigen/Dense>
#include <casadi/casadi.hpp>
#include <map>
#include <memory>
#include <vector>

namespace simple_casadi_mpc {

template <class T>
static T integrate_dynamics_forward_euler(double dt, T x, T u, std::function<T(T, T)> dynamics) {
    return x + dt * dynamics(x, u);
}

template <class T>
static T integrate_dynamics_modified_euler(double dt, T x, T u, std::function<T(T, T)> dynamics) {
    T k1 = dynamics(x, u);
    T k2 = dynamics(x + dt * k1, u);

    return x + dt * (k1 + k2) / 2;
}

template <class T>
static T integrate_dynamics_rk4(double dt, T x, T u, std::function<T(T, T)> dynamics) {
    T k1 = dynamics(x, u);
    T k2 = dynamics(x + dt / 2 * k1, u);
    T k3 = dynamics(x + dt / 2 * k2, u);
    T k4 = dynamics(x + dt * k3, u);
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

class Problem {
public:
    enum class DynamicsType {
        ContinuesForwardEuler,
        ContinuesModifiedEuler,
        ContinuesRK4,
        Discretized,
    };

    enum class ConstraintType {
        Equality,
        Inequality
    };

    Problem(DynamicsType dyn_type, size_t _nx, size_t _nu, size_t _horizon, double _dt)
        : dyn_type_(dyn_type), nx_(_nx), nu_(_nu), horizon_(_horizon), dt_(_dt) {
        double inf = std::numeric_limits<double>::infinity();

        Eigen::VectorXd uub = Eigen::VectorXd::Constant(nu(), inf);
        Eigen::VectorXd ulb = -uub;
        u_bounds_ = std::vector<LUbound>{horizon(), {ulb, uub}};

        Eigen::VectorXd xub = Eigen::VectorXd::Constant(nx(), inf);
        Eigen::VectorXd xlb = -xub;
        x_bounds_ = std::vector<LUbound>{horizon(), {xlb, xub}};
    }

    // ダイナミクス
    // 連続系の場合は、xとuを引数にとり、dxを返す、コンストラクタで離散化手法を{ContinuesForwardEuler,
    // ContinuesModifiedEuler, ContinuesRK4}から選択する
    // 離散系の場合は、xとuを引数にとり、x(k+1)を返す、コンストラクタでDiscretizedと指定する
    virtual casadi::MX dynamics(casadi::MX x, casadi::MX u) = 0;

    Eigen::VectorXd dynamics_eval(Eigen::VectorXd x, Eigen::VectorXd u) {
        casadi::DM x_dm = casadi::DM::zeros(nx(), 1);
        casadi::DM u_dm = casadi::DM::zeros(nu(), 1);
        for (size_t i = 0; i < nx(); i++) {
            x_dm(i) = x[i];
        }
        for (size_t i = 0; i < nu(); i++) {
            u_dm(i) = u[i];
        }
        casadi::MX dx_mx = dynamics(x_dm, u_dm);
        casadi::DM dx_dm = casadi::MX::evalf(dx_mx);
        Eigen::VectorXd dx = casadi_utils::to_eigen(dx_dm);
        return dx;
    }

    Eigen::VectorXd simulate(Eigen::VectorXd x0, Eigen::MatrixXd u) {
        assert(dyn_type_ == DynamicsType::Discretized);
        return dynamics_eval(x0, u);
    }

    Eigen::VectorXd simulate(Eigen::VectorXd x0, Eigen::MatrixXd u, double dt) {
        assert(dyn_type_ != DynamicsType::Discretized);
        auto dyn = std::bind(&Problem::dynamics_eval, this, std::placeholders::_1, std::placeholders::_2);
        switch (dyn_type_) {
        case DynamicsType::ContinuesForwardEuler:
            return integrate_dynamics_forward_euler<Eigen::VectorXd>(dt, x0, u, dyn);
            break;
        case DynamicsType::ContinuesModifiedEuler:
            return integrate_dynamics_modified_euler<Eigen::VectorXd>(dt, x0, u, dyn);
            break;
        case DynamicsType::ContinuesRK4:
            return integrate_dynamics_rk4<Eigen::VectorXd>(dt, x0, u, dyn);
            break;
        case DynamicsType::Discretized:
            break;
        }
        return x0;
    }

    // 操作量と状態量の上下限
    void set_input_bound(Eigen::VectorXd lb, Eigen::VectorXd ub, int start = -1, int end = -1) {
        std::tie(start, end) = index_range(start, end);
        for (int i = start; i < end; i++) {
            u_bounds_[i] = {lb, ub};
        }
    }

    void set_input_lower_bound(Eigen::VectorXd lb, int start = -1, int end = -1) {
        std::tie(start, end) = index_range(start, end);
        for (int i = start; i < end; i++) {
            u_bounds_[i].first = lb;
        }
    }

    void set_input_upper_bound(Eigen::VectorXd ub, int start = -1, int end = -1) {
        std::tie(start, end) = index_range(start, end);
        for (int i = start; i < end; i++) {
            u_bounds_[i].second = ub;
        }
    }

    void set_state_bound(Eigen::VectorXd lb, Eigen::VectorXd ub, int start = -1, int end = -1) {
        std::tie(start, end) = index_range(start, end);
        for (int i = start; i < end; i++) {
            x_bounds_[i] = {lb, ub};
        }
    }

    void set_state_lower_bound(Eigen::VectorXd lb, int start = -1, int end = -1) {
        std::tie(start, end) = index_range(start, end);
        for (int i = start; i < end; i++) {
            x_bounds_[i].first = lb;
        }
    }

    void set_state_upper_bound(Eigen::VectorXd ub, int start = -1, int end = -1) {
        std::tie(start, end) = index_range(start, end);
        for (int i = start; i < end; i++) {
            x_bounds_[i].second = ub;
        }
    }

    void add_constraint(ConstraintType type, std::function<casadi::MX(casadi::MX, casadi::MX)> constrinat) {
        if (type == ConstraintType::Equality) {
            equality_constrinats_.push_back(constrinat);
        } else {
            inequality_constrinats_.push_back(constrinat);
        }
    }

    // k番目のステージコスト
    virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u, size_t k) {
        (void)x;
        (void)u;
        (void)k;
        return 0;
    }

    // 終端のコスト
    virtual casadi::MX terminal_cost(casadi::MX x) {
        (void)x;
        return 0;
    }

    DynamicsType dynamics_type() const { return dyn_type_; }
    size_t nx() const { return nx_; }
    size_t nu() const { return nu_; }
    size_t horizon() const { return horizon_; }
    double dt() const { return dt_; }

    casadi::MX parameter(std::string name, size_t rows, size_t cols) {
        // std::string param_name = "p" + std::to_string(param_list_.size());
        auto param = casadi::MX::sym(name, rows, cols);
        param_list_[name] = {param, casadi::DM::zeros(rows, cols)};
        return param;
    }

private:
    std::pair<int, int> index_range(int start, int end) {
        if (start == -1 && end == -1) {
            return {0, horizon_};
        }
        if (start != -1 && end == -1) {
            return {start, start + 1};
        }
        return {start, end};
    }

    const DynamicsType dyn_type_;
    const size_t nx_;
    const size_t nu_;
    const size_t horizon_;
    const double dt_;

    using ConstraintFunc = std::function<casadi::MX(casadi::MX, casadi::MX)>;
    std::vector<ConstraintFunc> equality_constrinats_;
    std::vector<ConstraintFunc> inequality_constrinats_;

    using LUbound = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
    std::vector<LUbound> u_bounds_;
    std::vector<LUbound> x_bounds_;

    struct MXDMPair {
        casadi::MX mx;
        casadi::DM dm;
    };
    std::map<std::string, MXDMPair> param_list_;

    friend class MPC;
};

class MPC {
public:
    static casadi::Dict default_ipopt_config() {
        casadi::Dict config = {
            {"calc_lam_p", true},
            {"calc_lam_x", true},
            {"ipopt.sb", "yes"},
            {"ipopt.print_level", 0},
            {"print_time", false},
            {"ipopt.warm_start_init_point", "yes"},
            {"expand", true}};
        return config;
    }

    static casadi::Dict default_qpoases_config() {
        casadi::Dict config = {{"calc_lam_p", true},
                               {"calc_lam_x", true},
                               {"max_iter", 100},
                               {"print_header", false},
                               {"print_iteration", false},
                               {"print_status", false},
                               {"print_time", false},
                               {"qpsol", "qpoases"},
                               {"qpsol_options", casadi::Dict{{"enableRegularisation", true}, {"printLevel", "none"}}},
                               {"expand", true}};
        return config;
    }

    static casadi::Dict default_fatrop_config() {
        casadi::Dict config = {
            {"calc_lam_p", true},
            {"calc_lam_x", true},
            {"expand", true},
            {"print_time", false},
            {"fatrop.print_level", 0},
            {"fatrop.max_iter", 500},
            {"fatrop.mu_init", 0.1},
            {"structure_detection", "auto"},
            {"fatrop.warm_start_init_point", true},
            {"fatrop.tol", 1e-6},
            {"fatrop.acceptable_tol", 5e-3},

            // {"debug", true}, // 問題構造の.mtxファイルへの出力の有無
        };
        return config;
    }

    // static casadi::Dict default_hpipm_config() {
    //     casadi::Dict config = {{"calc_lam_p", true},
    //                            {"calc_lam_x", true},
    //                            {"max_iter", 100},
    //                            // {"print_header", false},
    //                            // {"print_iteration", false},
    //                            // {"print_status", false},
    //                            // {"print_time", false},
    //                            {"qpsol", "hpipm"},
    //                            {"qpsol_options", casadi::Dict{{"hpipm.iter_max", 100}, {"hpipm.warm_start", true}}},
    //                            {"expand", true}};
    //     return config;
    // }

    template <class T>
    MPC(std::shared_ptr<T> prob, std::string solver_name = "ipopt", casadi::Dict config = default_ipopt_config())
        : prob_(prob), solver_name_(solver_name), config_(config) {
        using namespace casadi;
        static_assert(std::is_base_of_v<Problem, T>, "prob must be based SimpleProb");

        const size_t nx = prob_->nx();
        const size_t nu = prob_->nu();
        const size_t N = prob_->horizon();

        Xs.reserve(N + 1);
        Us.reserve(N);

        for (size_t i = 0; i < N; i++) {
            Xs.push_back(MX::sym("X_" + std::to_string(i), nx, 1));
            Us.push_back(MX::sym("U_" + std::to_string(i), nu, 1));
        }
        Xs.push_back(MX::sym("X_" + std::to_string(N), nx, 1));

        std::vector<MX> w, g;
        std::vector<DM> w0;
        MX J = 0;

        std::function<casadi::MX(casadi::MX, casadi::MX)> dynamics;
        switch (prob_->dynamics_type()) {
        case Problem::DynamicsType::ContinuesForwardEuler: {
            std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
                std::bind(&Problem::dynamics, prob_, std::placeholders::_1, std::placeholders::_2);
            dynamics = std::bind(integrate_dynamics_forward_euler<casadi::MX>, prob_->dt(), std::placeholders::_1, std::placeholders::_2, con_dyn);
            break;
        }
        case Problem::DynamicsType::ContinuesModifiedEuler: {
            std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
                std::bind(&Problem::dynamics, prob_, std::placeholders::_1, std::placeholders::_2);
            dynamics = std::bind(integrate_dynamics_modified_euler<casadi::MX>, prob_->dt(), std::placeholders::_1, std::placeholders::_2, con_dyn);
            break;
        }
        case Problem::DynamicsType::ContinuesRK4: {
            std::function<casadi::MX(casadi::MX, casadi::MX)> con_dyn =
                std::bind(&Problem::dynamics, prob_, std::placeholders::_1, std::placeholders::_2);
            dynamics = std::bind(integrate_dynamics_rk4<casadi::MX>, prob_->dt(), std::placeholders::_1, std::placeholders::_2, con_dyn);
            break;
        }
        case Problem::DynamicsType::Discretized:
            dynamics = std::bind(&Problem::dynamics, prob_, std::placeholders::_1, std::placeholders::_2);
            break;
        }

        auto &u_bounds = prob_->u_bounds_;
        auto &x_bounds = prob_->x_bounds_;
        for (size_t i = 0; i < N; i++) {
            w.push_back(Xs[i]);

            if (i != 0) {
                for (size_t l = 0; l < nx; l++) {
                    lbw_.push_back(x_bounds[i - 1].first[l]);
                    ubw_.push_back(x_bounds[i - 1].second[l]);
                }
            } else {
                for (size_t l = 0; l < nx; l++) {
                    lbw_.push_back(0); // dummy
                    ubw_.push_back(0); // dummy
                }
            }

            w.push_back(Us[i]);
            for (size_t l = 0; l < nu; l++) {
                lbw_.push_back(u_bounds[i].first[l]);
                ubw_.push_back(u_bounds[i].second[l]);
            }
            MX xplus = dynamics(Xs[i], Us[i]);
            J += prob_->stage_cost(Xs[i], Us[i], i);

            g.push_back((Xs[i + 1] - xplus));
            for (size_t l = 0; l < nx; l++) {
                lbg_.push_back(0);
                ubg_.push_back(0);
                equality_.push_back(true);
            }

            for (auto &con : prob_->equality_constrinats_) {
                auto con_val = con(Xs[i], Us[i]);
                g.push_back(con_val);
                for (auto l = 0; l < con_val.size1(); l++) {
                    lbg_.push_back(0);
                    ubg_.push_back(0);
                    equality_.push_back(true);
                }
            }
            for (auto &con : prob_->inequality_constrinats_) {
                auto con_val = con(Xs[i], Us[i]);
                g.push_back(con_val);
                for (auto l = 0; l < con_val.size1(); l++) {
                    lbg_.push_back(-inf);
                    ubg_.push_back(0);
                    equality_.push_back(false);
                }
            }
        }
        J += prob_->terminal_cost(Xs[N]);

        w.push_back(Xs[N]);
        for (size_t l = 0; l < nx; l++) {
            lbw_.push_back(x_bounds[N - 1].first[l]);
            ubw_.push_back(x_bounds[N - 1].second[l]);
        }

        std::vector<MX> params;
        for (auto &[param_name, param_pair] : prob_->param_list_)
            params.push_back(param_pair.mx);
        casadi_prob_ = {{"x", vertcat(w)}, {"f", J}, {"g", vertcat(g)}, {"p", vertcat(params)}};
        if (solver_name_ == "fatrop" && config_["structure_detection"] == "auto") {
            config_["equality"] = equality_;
        }

        solver_ = nlpsol("solver", solver_name_, casadi_prob_, config_);
    }

    Eigen::VectorXd solve(Eigen::VectorXd x0, casadi::DMDict new_param_list = casadi::DMDict()) {
        using namespace casadi;

        // Set new parameter
        for (auto &[param_name, param] : new_param_list) {
            prob_->param_list_[param_name].dm = param;
        }

        const size_t nx = prob_->nx();
        const size_t nu = prob_->nu();
        for (size_t l = 0; l < nx; l++) {
            lbw_[l] = x0[l];
            ubw_[l] = x0[l];
        }

        DMDict arg;
        arg["x0"] = w0_;
        arg["lbx"] = vertcat(lbw_);
        arg["ubx"] = vertcat(ubw_);
        arg["lbg"] = vertcat(lbg_);
        arg["ubg"] = vertcat(ubg_);
        arg["lam_x0"] = lam_x0_;
        arg["lam_g0"] = lam_g0_;
        param_vec_.clear();
        param_vec_.reserve(prob_->param_list_.size());
        for (auto &[param_name, param_pair] : prob_->param_list_) {
            param_vec_.push_back(param_pair.dm);
        }
        arg["p"] = vertcat(param_vec_);
        DMDict sol = solver_(arg);

        w0_ = sol["x"];
        lam_x0_ = sol["lam_x"];
        lam_g0_ = sol["lam_g"];

        Eigen::VectorXd opt_u(nu);
        std::copy(w0_.ptr() + nx, w0_.ptr() + nx + nu, opt_u.data());

        return opt_u;
    }

    casadi::MXDict casadi_prob() const { return casadi_prob_; }

private:
    std::shared_ptr<Problem> prob_;
    std::string solver_name_;
    casadi::Dict config_;
    casadi::MXDict casadi_prob_;
    casadi::Function solver_;
    std::vector<casadi::MX> Xs;
    std::vector<casadi::MX> Us;

    std::vector<casadi::DM> lbw_;
    std::vector<casadi::DM> ubw_;
    std::vector<casadi::DM> lbg_;
    std::vector<casadi::DM> ubg_;
    std::vector<casadi::DM> param_vec_;

    std::vector<bool> equality_; // ダイナミクスと追加の制約が等式か不等式か

    casadi::DM w0_;
    casadi::DM lam_x0_;
    casadi::DM lam_g0_;
};

} // namespace simple_casadi_mpc