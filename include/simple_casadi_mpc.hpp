#pragma once
#include <casadi/casadi.hpp>
#include <Eigen/Dense>
#include <memory>
#include <vector>

namespace simple_casadi_mpc
{

template<class T>
static T integrate_dynamics_forward_euler(double dt, T x, T u, std::function<T(T,T)> dynamics)
{
    return x + dt*dynamics(x, u);
}

template<class T>
static T integrate_dynamics_modified_euler(double dt, T x, T u, std::function<T(T,T)> dynamics)
{
    T k1 = dynamics(x, u);
    T k2 = dynamics(x+dt*k1, u);
    
    return x + dt*(k1+k2)/2;
}

template<class T>
static T integrate_dynamics_rk4(double dt, T x, T u, std::function<T(T,T)> dynamics)
{
    T k1 = dynamics(x, u);
    T k2 = dynamics(x + dt / 2 * k1, u);
    T k3 = dynamics(x + dt / 2 * k2, u);
    T k4 = dynamics(x + dt * k3, u);
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

class Problem
{
public:
    Problem() = default;
    Problem(size_t _nx, size_t _nu, size_t _horizon, double _dt):
        nx_(_nx), nu_(_nu), horizon_(_horizon), dt_(_dt) {}

    // dt_で離散化されたダイナミクス
    // 微分方程式の場合はintegrate_xxx関数で積分された形にする
    virtual casadi::MX discretized_dynamics(casadi::MX x, casadi::MX u) = 0;

    // 操作量と状態量の上下限
    using LUbound = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
    // size (nu * horizon) u0 u1 u2 ... uN-1
    virtual std::vector<LUbound> u_bounds() = 0;

    // size (nx * horizon) x1 x2 x3 ... xN
    // x0はsolveで与える状態量
    virtual std::vector<LUbound> x_bounds() = 0;

    // k番目のステージコスト
    virtual casadi::MX stage_cost(casadi::MX x, casadi::MX u)
    {
        return 0;
    }

    // 終端のコスト
    virtual casadi::MX terminal_cost(casadi::MX x)
    {
        return 0;
    }

    size_t nx() const { return nx_; }
    size_t nu() const { return nu_; }
    size_t horizon() const { return horizon_; }
    double dt() const { return dt_; }

private:
    const size_t nx_;
    const size_t nu_;
    const size_t horizon_;
    const double dt_;
};

class MPC
{
public:
    static casadi::Dict default_config()
    {
        casadi::Dict config = 
        {
            {"calc_lam_p", true},
            {"calc_lam_x", true},
            {"ipopt.sb", "yes"}, 
            {"ipopt.print_level", 0},
            {"print_time", false},
            {"ipopt.warm_start_init_point", "yes"},
        };
        return config;
    }

    static casadi::Dict default_qpoases_config()
    {
        casadi::Dict config = 
        {
            {"calc_lam_p", true},
            {"calc_lam_x", true},
            {"max_iter", 100},
            {"print_header", false},
            {"print_iteration", false},
            {"print_status", false},
            {"print_time", false},
            {"qpsol", "qpoases"},
            {"qpsol_options", casadi::Dict{{"enableRegularisation", true}, {"printLevel", "none"}}},
        };
        return config;
    }

    static casadi::Dict default_hpipm_config()
    {
        casadi::Dict config = 
        {
            {"calc_lam_p", true},
            {"calc_lam_x", true},
            {"max_iter", 100},
            {"print_header", false},
            {"print_iteration", false},
            {"print_status", false},
            {"print_time", false},
            {"qpsol", "hpipm"},
            {"qpsol_options", casadi::Dict{{"hpipm.iter_max", 100}, {"hpipm.warm_start", true}}},
        };
        return config;
    }

    template<class T>
    MPC(std::shared_ptr<T> prob, std::string solver_name = "ipopt", casadi::Dict config = default_config()):
        prob_(prob), solver_name_(solver_name), config_(config)
    {
        using namespace casadi;
        static_assert(std::is_base_of_v<Problem, T>, "prob must be based SimpleProb");

        const size_t nx = prob_->nx();
        const size_t nu = prob_->nu();
        const size_t N = prob_->horizon();

        Xs.reserve(N+1);
        Us.reserve(N);

        for(size_t i = 0; i < N; i++)
        {
            Xs.push_back(MX::sym("X_"+std::to_string(i), nx, 1));
            Us.push_back(MX::sym("U_"+std::to_string(i), nu, 1));
        }
        Xs.push_back(MX::sym("X_"+std::to_string(N), nx, 1));

        std::vector<MX> w, g;
        std::vector<DM> w0;
        MX J = 0;

        auto u_bounds = prob_->u_bounds();
        auto x_bounds = prob_->x_bounds();
        for(size_t i = 0; i < N; i++)
        {
            w.push_back(Xs[i]);

            if(i != 0)
            {
                for(auto l = 0; l < nx; l++)
                {
                    lbw_.push_back(x_bounds[i-1].first[l]);
                    ubw_.push_back(x_bounds[i-1].second[l]);
                }
            }
            else
            {
                for(auto l = 0; l < nx; l++)
                {
                    lbw_.push_back(0); // dummy
                    ubw_.push_back(0); // dummy
                }
            }

            w.push_back(Us[i]);
            for(auto l = 0; l < nu; l++)
            {
                lbw_.push_back(u_bounds[i].first[l]);
                ubw_.push_back(u_bounds[i].second[l]);
            }
            MX xplus = prob_->discretized_dynamics(Xs[i], Us[i]);
            J += prob_->stage_cost(Xs[i], Us[i]);

            g.push_back((xplus - Xs[i+1]));
            for(auto l = 0; l < nx; l++)
            {
                lbg_.push_back(0);
                ubg_.push_back(0);
            }
        }
        J += prob_->terminal_cost(Xs[N]);
        
        w.push_back(Xs[N]);
        for(auto l = 0; l < nx; l++)
        {
            lbw_.push_back(x_bounds[nx-1].first[l]);
            ubw_.push_back(x_bounds[nx-1].second[l]);
        }

        casadi_prob_ = {{"x", vertcat(w)}, {"f", J}, {"g", vertcat(g)}};
        solver_ = nlpsol("solver", solver_name_, casadi_prob_, config_);
    }

    Eigen::VectorXd solve(Eigen::VectorXd x0)
    {
        using namespace casadi;
        const size_t nx = prob_->nx();
        const size_t nu = prob_->nu();
        const size_t N = prob_->horizon();
        for(auto l = 0; l < nx; l++)
        {
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
        DMDict sol = solver_(arg);

        w0_ = sol["x"];
        lam_x0_ = sol["lam_x"];
        lam_g0_ = sol["lam_g"];

        Eigen::VectorXd opt_u(nu);
        std::copy(w0_.ptr()+nx, w0_.ptr()+nx+nu, opt_u.data());

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

    casadi::DM w0_;
    casadi::DM lam_x0_;
    casadi::DM lam_g0_;
};

}