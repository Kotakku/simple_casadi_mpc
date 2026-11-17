#pragma once
// Stage-wise Hessian of the Lagrangian and constraint Jacobian of the MPC NLP
// (config option "structured_derivatives" of simple_casadi_mpc::MPC).
#include <casadi/casadi.hpp>
#include <functional>
#include <vector>

namespace simple_casadi_mpc {

/// @brief Stage-wise description of the MPC NLP, input of build_structured_derivatives.
///
/// The decision vector is w = [x_0; u_0; s_0; x_1; ...; u_{N-1}; s_{N-1}; x_N] and the
/// constraint rows of stage k are [dynamics_k; (init); path_k] (manifold or interleaved
/// layout) or all dynamics followed by all path constraints (otherwise).
struct StructuredStageData {
  casadi_int N, nx, nu, ns, np, n_defect, ndx, ng;
  bool manifold, interleave, variable_dt;
  casadi::Manifold Mx;                ///< state manifold
  casadi::Function F, F_defect, L, G; ///< discrete dynamics, its residual, stage cost, path g
  std::vector<casadi::MX> params;
  std::vector<size_t> per_stage_idx; ///< parameters with one column per stage
  casadi::MX X, U, x0, xN, terminal, w, p;
  casadi::MX x_init; ///< initial-state parameter (manifold only)
  std::vector<casadi::MX> Ss;
  std::vector<double> dts; ///< per-stage time steps (variable_dt only)
  struct Soft {
    casadi_int m;
    double w1, w2;
  };
  std::vector<Soft> soft; ///< soft constraints, in the order of s_k
};

/// @brief Custom derivative functions for the nlpsol options custom_hess_lag / custom_jac_g.
struct StructuredDerivatives {
  casadi::Function hess_lag; ///< (x, p, lam_f, lam_g) -> (hess, grad)
  casadi::Function jac_g;    ///< (x, p) -> (jac)
};

/// @brief Hessian of the Lagrangian and constraint Jacobian assembled stage by stage.
///
/// Automatic differentiation of the whole NLP colors the Hessian globally, which needs
/// many directions when a stage couples (x_k, u_k) with x_{k+1} nonlinearly (e.g. the
/// manifold dynamics residual). Here the derivatives of each stage are computed as
/// expanded SX functions of (x_k, u_k, s_k, x_{k+1}), mapped over the stages and
/// scattered into the global matrices with a constant sparse matrix. They are passed to
/// CasADi through the nlpsol options `custom_hess_lag` and `custom_jac_g`.
inline StructuredDerivatives build_structured_derivatives(const StructuredStageData &d) {
  using namespace casadi;
  const casadi_int N = d.N, nx = d.nx, nu = d.nu, ns = d.ns, np = d.np;
  const casadi_int ws = nx + nu + ns;    // entries of [x_k; u_k; s_k] in w
  const casadi_int nr = d.n_defect + np; // rows of [dynamics_k; path_k]
  const Manifold &Mx = d.Mx;
  StructuredDerivatives ret;

  // Global row indices of the dynamics and path constraints of each stage (stage major)
  std::vector<casadi_int> dyn_rows, path_rows, init_rows;
  for (casadi_int k = 0; k < N; ++k) {
    casadi_int off;
    if (d.manifold) {
      off = k == 0 ? 0 : (d.n_defect + d.ndx + np) + (k - 1) * nr;
    } else if (d.interleave) {
      off = k * nr;
    } else {
      off = -1;
    }
    for (casadi_int i = 0; i < d.n_defect; ++i)
      dyn_rows.push_back(off >= 0 ? off + i : k * nx + i);
    if (d.manifold && k == 0)
      for (casadi_int i = 0; i < d.ndx; ++i)
        init_rows.push_back(d.n_defect + i);
    casadi_int poff =
        off >= 0 ? off + d.n_defect + (d.manifold && k == 0 ? d.ndx : 0) : N * nx + k * np;
    for (casadi_int i = 0; i < np; ++i)
      path_rows.push_back(poff + i);
  }
  // Global indices of z_k in w
  auto z_index = [&](casadi_int k) {
    std::vector<casadi_int> r;
    for (casadi_int i = 0; i < nx + nu + ns; ++i)
      r.push_back(k * ws + i);
    for (casadi_int i = 0; i < nx; ++i)
      r.push_back((k + 1) * ws + i);
    return r;
  };

  // Stage Lagrangian lam_f * (L + penalty) + lam_d' r + lam_p' G
  MX xs = MX::sym("x", nx), us = MX::sym("u", nu), ss = MX::sym("s", ns), x1s = MX::sym("x1", nx),
     dts = MX::sym("dt"), lfs = MX::sym("lam_f"), lds = MX::sym("lam_d", d.n_defect),
     lps = MX::sym("lam_p", np);
  std::vector<MX> pfull, pcol;
  for (auto &p : d.params)
    pfull.push_back(MX::sym(p.name() + "_full", p.sparsity()));
  for (size_t i : d.per_stage_idx)
    pcol.push_back(MX::sym(d.params[i].name() + "_col", d.params[i].size1(), 1));
  std::vector<MX> f_args = {xs, us};
  if (d.variable_dt)
    f_args.push_back(dts);
  MX r;
  if (d.manifold) {
    std::vector<MX> a = f_args;
    a.push_back(x1s);
    r = d.F_defect(a)[0];
  } else {
    r = x1s - d.F(f_args)[0];
  }
  std::vector<MX> l_args = {xs, us};
  for (size_t p = 0, c = 0; p < d.params.size(); ++p) {
    bool col = c < d.per_stage_idx.size() && d.per_stage_idx[c] == p;
    l_args.push_back(col ? pcol[c++] : pfull[p]);
  }
  MX cost = d.L(l_args)[0];
  casadi_int off = 0;
  for (auto &sc : d.soft) {
    MX sp = ss(Slice(off, off + sc.m));
    cost += sc.w1 * sum1(sp);
    if (sc.w2 != 0.0)
      cost += 0.5 * sc.w2 * dot(sp, sp);
    off += sc.m;
  }

  MX g = MX(0, 1);
  if (np > 0) {
    std::vector<MX> g_args = {xs, us};
    if (ns > 0)
      g_args.push_back(ss);
    g_args.insert(g_args.end(), pfull.begin(), pfull.end());
    g = d.G(g_args)[0];
  }
  MX z = vertcat(std::vector<MX>{xs, us, ss, x1s});
  MX gamma = lfs * cost + dot(lds, r);
  if (np > 0)
    gamma += dot(lps, g);
  std::vector<MX> s_in = {xs, us, ss, x1s, dts};
  s_in.insert(s_in.end(), pfull.begin(), pfull.end());
  s_in.insert(s_in.end(), pcol.begin(), pcol.end());
  s_in.push_back(lfs);
  s_in.push_back(lds);
  s_in.push_back(lps);
  Dict ex = {{"cse", true}};
  Function H_stage("H_stage", s_in, {tril(MX::hessian(gamma, z)), MX::gradient(gamma, z)});
  H_stage = H_stage.expand(H_stage.name(), ex);
  Function J_stage("J_stage", s_in, {MX::jacobian(vertcat(r, g), z)});
  J_stage = J_stage.expand(J_stage.name(), ex);

  // Mapped arguments (in terms of the NLP symbols)
  MX lam_f = MX::sym("lam_f"), lam_g = MX::sym("lam_g", d.ng);
  std::vector<MX> m_args = {d.X(Slice(), Slice(0, N)), d.U, ns > 0 ? horzcat(d.Ss) : MX(0, N),
                            d.X(Slice(), Slice(1, N + 1)),
                            d.variable_dt ? MX(DM(d.dts).T()) : MX::zeros(1, N)};
  for (auto &p : d.params)
    m_args.push_back(repmat(p, 1, N));
  for (size_t i : d.per_stage_idx)
    m_args.push_back(d.params[i]);
  m_args.push_back(repmat(lam_f, 1, N));
  m_args.push_back(reshape(lam_g(dyn_rows), d.n_defect, N));
  m_args.push_back(np > 0 ? reshape(lam_g(path_rows), np, N) : MX(0, N));

  // Scatter the nonzeros of blocks into a matrix with the given global indices
  struct Part {
    MX nz;                                   // nonzeros (column)
    std::vector<casadi_int> rows, cols, src; // global (row, col) of entries from nz[src]
  };
  auto assemble = [&](const std::vector<Part> &parts, casadi_int nrow, casadi_int ncol) {
    std::vector<casadi_int> rr, cc, src;
    std::vector<MX> nzs;
    casadi_int n_src = 0;
    for (auto &pt : parts) {
      for (size_t e = 0; e < pt.rows.size(); ++e) {
        rr.push_back(pt.rows[e]);
        cc.push_back(pt.cols[e]);
        src.push_back(n_src + pt.src[e]);
      }
      n_src += pt.nz.size1();
      nzs.push_back(pt.nz);
    }
    std::vector<casadi_int> mapping;
    Sparsity sp = Sparsity::triplet(nrow, ncol, rr, cc, mapping, true); // entry -> nonzero
    DM A = DM::triplet(mapping, src, std::vector<double>(src.size(), 1.0), sp.nnz(), n_src);
    return MX::sparsity_cast(mtimes(A, vertcat(nzs)), sp);
  };
  // symmetric: sp_block is lower triangular; also scatter the mirrored entries
  auto block_part = [&](const MX &blocks, const Sparsity &sp_block,
                        const std::function<casadi_int(casadi_int, casadi_int)> &row_of,
                        const std::function<casadi_int(casadi_int, casadi_int)> &col_of,
                        casadi_int n_blocks, bool symmetric = false) {
    Part pt;
    MX v;
    blocks.get_nz(v, false, Slice());
    pt.nz = v;
    std::vector<casadi_int> br, bc;
    sp_block.get_triplet(br, bc);
    for (casadi_int k = 0; k < n_blocks; ++k)
      for (size_t e = 0; e < br.size(); ++e) {
        casadi_int r = row_of(k, br[e]), c = col_of(k, bc[e]), i = k * br.size() + e;
        pt.rows.push_back(r);
        pt.cols.push_back(c);
        pt.src.push_back(i);
        if (symmetric && r != c) {
          pt.rows.push_back(c);
          pt.cols.push_back(r);
          pt.src.push_back(i);
        }
      }
    return pt;
  };
  std::vector<std::vector<casadi_int>> zi(N);
  for (casadi_int k = 0; k < N; ++k)
    zi[k] = z_index(k);

  // Hessian (full symmetric, from the lower triangular blocks) and gradient of the Lagrangian:
  // stages, terminal cost, initial condition
  std::vector<Part> h_parts, g_parts;
  std::vector<MX> Hres = H_stage.map(N)(m_args);
  h_parts.push_back(block_part(
      Hres[0], H_stage.sparsity_out(0), [&](casadi_int k, casadi_int i) { return zi[k][i]; },
      [&](casadi_int k, casadi_int j) { return zi[k][j]; }, N, true));
  g_parts.push_back(block_part(
      Hres[1], H_stage.sparsity_out(1), [&](casadi_int k, casadi_int i) { return zi[k][i]; },
      [&](casadi_int, casadi_int) { return casadi_int(0); }, N));
  const MX &xN = d.xN;
  {
    MX xt = MX::sym("xN", nx);
    MX term = MX::substitute(d.terminal, xN, xt);
    Function Ht("H_terminal", {xt, d.p, lam_f},
                {tril(MX::hessian(lam_f * term, xt)), MX::gradient(lam_f * term, xt)});
    Ht = Ht.expand(Ht.name(), ex);
    std::vector<MX> rt = Ht(std::vector<MX>{xN, d.p, lam_f});
    h_parts.push_back(block_part(
        rt[0], Ht.sparsity_out(0), [&](casadi_int, casadi_int i) { return N * ws + i; },
        [&](casadi_int, casadi_int j) { return N * ws + j; }, 1, true));
    g_parts.push_back(block_part(
        rt[1], Ht.sparsity_out(1), [&](casadi_int, casadi_int i) { return N * ws + i; },
        [&](casadi_int, casadi_int) { return casadi_int(0); }, 1));
  }
  const MX &x0 = d.x0;
  MX x_init = d.manifold ? d.x_init : MX();
  Function Hi, Ji;
  if (d.manifold) {
    MX xt = MX::sym("x0", nx), xi = MX::sym("x_init", nx), li = MX::sym("lam_i", d.ndx);
    MX ri = Mx.defect(xt, xi);
    Hi = Function("H_init", {xt, xi, li},
                  {tril(MX::hessian(dot(li, ri), xt)), MX::gradient(dot(li, ri), xt)})
             .expand("H_init", ex);
    Ji = Function("J_init", {xt, xi}, {MX::jacobian(ri, xt)}).expand("J_init", ex);
    std::vector<MX> hi = Hi(std::vector<MX>{x0, x_init, lam_g(init_rows)});
    h_parts.push_back(block_part(
        hi[0], Hi.sparsity_out(0), [&](casadi_int, casadi_int i) { return i; },
        [&](casadi_int, casadi_int j) { return j; }, 1, true));
    g_parts.push_back(block_part(
        hi[1], Hi.sparsity_out(1), [&](casadi_int, casadi_int i) { return i; },
        [&](casadi_int, casadi_int) { return casadi_int(0); }, 1));
  }
  const casadi_int nw = d.w.size1();
  MX H = assemble(h_parts, nw, nw);
  MX grad = densify(assemble(g_parts, nw, 1));
  ret.hess_lag = Function("structured_hess_lag", {d.w, d.p, lam_f, lam_g}, {H, grad},
                          {"x", "p", "lam_f", "lam_g"}, {"hess", "grad"});

  // Jacobian of g
  // (the multiplier inputs do not enter the Jacobian)
  std::vector<MX> j_args(m_args.begin(), m_args.end() - 3);
  j_args.push_back(MX::zeros(1, N));
  j_args.push_back(MX::zeros(d.n_defect, N));
  j_args.push_back(MX::zeros(np, N));
  std::vector<Part> j_parts;
  MX Jb = J_stage.map(N)(j_args)[0];
  j_parts.push_back(block_part(
      Jb, J_stage.sparsity_out(0),
      [&](casadi_int k, casadi_int i) {
        return i < d.n_defect ? dyn_rows[k * d.n_defect + i] : path_rows[k * np + (i - d.n_defect)];
      },
      [&](casadi_int k, casadi_int j) { return zi[k][j]; }, N));
  if (d.manifold) {
    j_parts.push_back(block_part(
        Ji(std::vector<MX>{x0, x_init})[0], Ji.sparsity_out(0),
        [&](casadi_int, casadi_int i) { return init_rows[i]; },
        [&](casadi_int, casadi_int j) { return j; }, 1));
  }
  MX Jg = assemble(j_parts, d.ng, nw);
  ret.jac_g = Function("structured_jac_g", {d.w, d.p}, {Jg}, {"x", "p"}, {"jac"});
  return ret;
}

} // namespace simple_casadi_mpc
