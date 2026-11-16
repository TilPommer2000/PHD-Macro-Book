# Chapter 10 in Julia: value function iteration

Julia translations of the MATLAB codes for Section 10.5 ("Dynamic Programming via Value Function
Iteration"), plus the building blocks they rely on. They follow the MATLAB files closely, so you
can read the two side by side, and they add checks the originals lack: a comparison with the
closed-form solution, convergence plots, and a cross-check between methods.

| Julia | MATLAB original | What it does |
|---|---|---|
| `vfi_tools.jl` | `Ex1_GS.m`, `Ex2_tauchen.m`, `Ex2_rouwenhorst.m`, `Ex4_location.m`, `hpfilt.m` | Golden-section search, linear interpolation, Tauchen and Rouwenhorst, HP filter |
| `ex3_cake.jl` | `Ex3_gridsearch.m`, `Ex3_GS.m` | Deterministic cake eating, both methods, checked against the exact solution |
| `ex4_brock_mirman.jl` | `Ex4_gridsearch.m`, `Ex4_GS.m`, `Ex4_labor.m` | Stochastic growth with labor: VFI, Howard improvement, simulation, HP-filtered moments |

## Running

Only `Plots` is needed beyond the standard library (`using Pkg; Pkg.add("Plots")` if it is
missing). From this folder:

```sh
julia ex3_cake.jl
julia ex4_brock_mirman.jl
```

Or open a script in VS Code and step through the `##` cells with Shift+Enter, which is the better
way to learn from them. Figures go to `Results/`.

## Suggested path

1. **The algorithm.** Read the slide frames "The Bellman Equation" through "Example 4" in
   `Slides/PDF-Slides/slides_Ch10_Computation.pdf` (Section 10.5). The contraction mapping
   argument is in Chapter 4.
2. **Cake eating, grid search.** In `ex3_cake.jl`, read `vfi_gridsearch` first. It is the
   algorithm on the slide with nothing added: a guess `V`, a loop over states, a max over choices,
   a sup-norm stopping rule.
3. **Cake eating, off-grid.** `vfi_golden` replaces the inner `max` with `golden_max` and reads
   `V` between grid points with `interp`. Compare the accuracy lines the script prints.
4. **Two things the cake problem shows:**
   - The contraction theorem guarantees convergence at rate β or faster (convergence plot).
   - The remaining error comes from the grid, not from stopping early. It is concentrated near
     a = 0, where V has an infinite slope, and a grid that is denser there cuts it by 5-10×.
5. **Brock-Mirman.** `ex4_brock_mirman.jl` adds a Markov shock (`EV = V * P'`) and a static
   choice (labor) solved inside the period return. Note how precomputing that return array makes
   grid search fast, and how Howard steps cut 927 iterations to 46.

## Differences from the MATLAB codes

- Ex4 Method A computes the period return F(K, K', z), including the labor bisection, once
  before iterating, rather than in every iteration. The result is the same and much faster.
- Bisection for labor uses tolerance 1e-8 (MATLAB: 1e-4), and `golden_max` returns the best point
  in the final bracket rather than the last point tried. Differences are within tolerance.
- Random draws differ from MATLAB, so the simulated moments will not match the book to the digit.
- In `Ex3_GS.m` the end-point values `tempva`/`tempvc` pair each end point with the other end's
  continuation value (`v(i)` and `v(1)` are swapped). These values are only used in the
  corner-solution check, which cannot trigger in this problem, so the results are unaffected.
- Parameters follow the MATLAB code (α = 1/3, δ = 0.0233, disutility −ξH). The slides state
  α = 0.36, δ = 0.025 and utility (1−φ) log C + φ log(1−H).

## Exercises

1. **Stochastic cake eating** (slides, "Stochastic Case"; there is no code for it). Adapt
   `vfi_gridsearch` from `ex3_cake.jl` using the `EV = V * P'` pattern from `ex4_brock_mirman.jl`.
   Think about the upper bound of the grid, since the cake can now grow.
2. **The slides' version of Example 4.** Replace the two-state chain with
   `y, P = rouwenhorst(7, 0.95, 0.007); zgrid = exp.(y)`, then switch to
   (1−φ) log C + φ log(1−H). Only `labor`, `period_return` and the calibration of ξ (now φ)
   change.
3. **Speed.** The policy K'(K, z) is increasing in K, so the search over j can start at the
   previous state's optimum. Add this, then add Howard steps to Method B.
4. **Accuracy.** Compute Euler-equation errors along the simulated path, and compare the VFI
   policy with the linearized one from `Ex5_BK.m`.
5. **A faster method.** `Codes-and-Data/Chapter-07/main_figure_4.jl` solves a savings problem
   with the endogenous grid method, which avoids the maximization step entirely. Compare it
   with VFI.
