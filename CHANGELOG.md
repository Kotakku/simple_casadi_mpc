# Changelog

## [1.0.0](https://github.com/Kotakku/simple_casadi_mpc/compare/v0.1.0...v1.0.0) (2026-05-09)


### ⚠ BREAKING CHANGES

* Problem::soft_add_constraint has been renamed to Problem::add_soft_constraint. Existing callers must update the call site; signature is otherwise unchanged.

### Features

* add pre-commit hooks ([0310d4d](https://github.com/Kotakku/simple_casadi_mpc/commit/0310d4d83ac8f26f5b3af8a6a5c13f13ebd63431))
* add soft_add_constraint with slack variables ([03bbdb6](https://github.com/Kotakku/simple_casadi_mpc/commit/03bbdb6ac907969d03f5030620add16c9d7e48e2))
* add stage-specific add_constraint_at / add_soft_constraint_at ([#36](https://github.com/Kotakku/simple_casadi_mpc/issues/36)) ([3882d64](https://github.com/Kotakku/simple_casadi_mpc/commit/3882d6457ef5ac896c9d2b767e2089f6d8693a84)), closes [#25](https://github.com/Kotakku/simple_casadi_mpc/issues/25)
* allow customizing JIT compile options in JITMPC ([6aba197](https://github.com/Kotakku/simple_casadi_mpc/commit/6aba19791ba5ca812ed4c8d0a7352e9fe54dd11d))
* allow customizing JIT compile options in JITMPC ([f4598d7](https://github.com/Kotakku/simple_casadi_mpc/commit/f4598d7159dc9c23915b05246c3299cd727a0faa)), closes [#22](https://github.com/Kotakku/simple_casadi_mpc/issues/22)
* migrate examples to matplotlibcpp17 ([888decf](https://github.com/Kotakku/simple_casadi_mpc/commit/888decf4284bd59785ae4b14942c3d68a5c0a682))
* soft constraints via slack variables ([1815176](https://github.com/Kotakku/simple_casadi_mpc/commit/1815176813ce27d988c3477945f71f64c1217434))
* support variable per-stage time step in Problem ([#35](https://github.com/Kotakku/simple_casadi_mpc/issues/35)) ([61518b2](https://github.com/Kotakku/simple_casadi_mpc/commit/61518b2f7491e1bcf8eb1676116b8f230d4aaea2)), closes [#16](https://github.com/Kotakku/simple_casadi_mpc/issues/16)


### Bug Fixes

* Add .nojekyll to prevent Jekyll processing ([6b4c03e](https://github.com/Kotakku/simple_casadi_mpc/commit/6b4c03e1d58bd13ae0fe6f6a3b1f681bd40aedfc))
* remove fatrop.warm_start_init_point from default config ([e8bb83b](https://github.com/Kotakku/simple_casadi_mpc/commit/e8bb83b627442009842f65e3f59c2c0d24671333))
* rename fatrop.acceptable_tol to fatrop.tol_acceptable ([2a424a2](https://github.com/Kotakku/simple_casadi_mpc/commit/2a424a283079722d285eae067839462147e19f6c))


### Performance Improvements

* **JIT/AOT:** improve compile time ([6989d60](https://github.com/Kotakku/simple_casadi_mpc/commit/6989d60c6330df6df6456e9ccafe673307def620))


### Code Refactoring

* rename soft_add_constraint to add_soft_constraint ([#31](https://github.com/Kotakku/simple_casadi_mpc/issues/31)) ([cf7f2b9](https://github.com/Kotakku/simple_casadi_mpc/commit/cf7f2b937ecde4450997607611b01fdf686a1e66))
