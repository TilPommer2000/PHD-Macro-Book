# Example 3 (Section 10.5): deterministic cake eating by value function iteration.
#
#     V(a) = max_{0 ≤ a' ≤ a}  √(a - a') + β V(a')
#
# Julia version of Ex3_gridsearch.m (Method A) and Ex3_GS.m (Method B). Unlike the MATLAB codes,
# it compares both answers to the closed-form solution, so you can see how accurate each method is.
#
# Run from this folder with `julia ex3_cake.jl`, or step through the `##` cells in VS Code
# (Shift+Enter). Figures are saved in Results/.

include("vfi_tools.jl")
using Plots

results_dir = joinpath(@__DIR__, "Results")
mkpath(results_dir)

## Parameters (as in the MATLAB codes)
β = 0.98
amin, amax = 0.0, 10.0

## Closed form, to check the numerical answers
# This problem has an exact solution, so we can measure how accurate the numerical answer is.
# (Most models have no closed form, and then this check is not available.)
#
# Derivation by "guess and verify":
#   1. Guess that the value function has the form V(a) = A·√a, with A an unknown number.
#   2. Put the guess into the right-hand side of the Bellman equation:
#          max over a' of  √(a - a') + β·A·√a'
#   3. First-order condition for a':  1/(2√(a - a')) = β·A/(2√a'),
#      which gives  a' = β²A²·a / (1 + β²A²).
#   4. Substitute this a' back. The result equals A·√a (the guess) only if A² = 1 + β²A²,
#      that is, A² = 1/(1 - β²).
#   5. So V(a) = √(a/(1 - β²)) and the policy is a' = β²·a: each period you keep the share β²
#      of the cake and eat the share 1 - β² (about 4% when β = 0.98).
V_true(a) = sqrt(a / (1 - β^2))     # exact value of having a cake of size a
g_true(a) = β^2 * a                 # exact policy: cake left for tomorrow

## Method A: a' restricted to the grid, maximized by grid search (Ex3_gridsearch.m)
# Inputs
#   agrid  the grid: possible cake sizes, sorted from small to large, with agrid[1] = 0.
#          Choosing it is step 1 of the algorithm on the slides; it happens before the call.
#   β      discount factor
#   tol    stopping tolerance: stop once no value changes by more than tol in a round
#   maxit  safety cap on the number of rounds, so a bug cannot make the loop run forever
# Output (a named tuple)
#   V      value of each grid point: V[i] = V(agrid[i])
#   pol    policy: cake left for tomorrow, pol[i] = a'(agrid[i])
#   errs   largest change in V in each round, for the convergence plot
#
# Small example (the 5-point grid 0, 2.5, 5, 7.5, 10 of the "VFI by Hand" page): in the round
# where the old guess is V(a) = √a, the row for cake 10 (i = 5) scores the five choices
# 3.16, 4.29, 4.43, 4.27, 3.10. The best is j = 3 (leave 5), so Vnew[5] = 4.43 and pol[5] = 3.
function vfi_gridsearch(agrid, β; tol = 1e-6, maxit = 100_000)
    N = length(agrid)                # number of grid points (cake sizes)

    # Step 2: start from a guess. V = 0 means "leftover cake is worth nothing", as if the world
    # ended after today. Any other starting guess also converges (contraction mapping theorem).
    # V holds the current guess; Vnew collects the improved values during a round. Two separate
    # arrays are needed because every new value must be computed from the *old* V.
    V, Vnew = zeros(N), zeros(N)

    # The policy is stored as grid indices: pol[i] = j means "with cake agrid[i], leave agrid[j]".
    pol = ones(Int, N)
    errs = Float64[]                 # empty list; one number is added per round

    for _ in 1:maxit                 # one pass through this loop = one round (one application of T)

        # Step 3: compute the new value of every cake size, using the old V for tomorrow.
        for i in 1:N                 # i indexes today's cake, agrid[i] (a row of the table)
            best, jbest = -Inf, 1    # highest score found so far in this row, and its column

            # Try every amount you could leave for tomorrow (the columns of the table).
            # You cannot leave more than you have, so only j ≤ i is feasible. (The MATLAB code
            # tries every j and gives the infeasible ones a score of -10000 instead.)
            for j in 1:i
                # Score of this choice = utility from eating today + discounted value of the leftover
                #   eaten today:          agrid[i] - agrid[j]
                #   utility today:        √(eaten today)
                #   value of leftover:    V[j], from the previous round. No interpolation is needed
                #                         because the leftover agrid[j] is itself a grid point.
                val = sqrt(agrid[i] - agrid[j]) + β * V[j]

                # Keep the highest score in the row. Using ">" (not ">=") keeps the first, i.e.
                # smallest, j when two choices tie, as MATLAB's max does.
                if val > best
                    best, jbest = val, j
                end
            end

            # The best score becomes the new value of this cake size, and the choice that
            # achieved it becomes the policy for this cake size.
            Vnew[i], pol[i] = best, jbest
        end

        # Step 4: has the guess stopped changing? Record the largest change over all cake sizes,
        # the sup norm ||Vnew - V||. The dots in abs. and .- apply the operation element by
        # element ("broadcasting").
        push!(errs, maximum(abs.(Vnew .- V)))

        # The new values become the guess for the next round. ".=" copies the numbers into V.
        # Writing V = Vnew instead would make both names point to the same array: the next round
        # would mix old and new values, and the stopping test would compare the array with itself
        # (difference 0) and stop too early.
        V .= Vnew

        # Stop once the largest change is below tol. "condition && return x" is short for
        # "if condition, return x". agrid[pol] turns the grid indices into cake sizes, so the
        # policy is returned as amounts of cake, like Method B below.
        errs[end] < tol && return (; V, pol = agrid[pol], errs)
    end

    # Only reached if maxit rounds pass without meeting the stopping rule.
    error("VFI did not converge in $maxit iterations")
end

## Method B: a' anywhere in [0, a], V interpolated off the grid (Ex3_GS.m)
function vfi_golden(agrid, β; tol = 1e-6, maxit = 10_000)
    N = length(agrid)
    V, Vnew = zeros(N), zeros(N)
    pol = zeros(N)                   # pol[i] = optimal a' itself (no longer a grid index)
    errs = Float64[]
    for _ in 1:maxit
        for i in 1:N
            a = agrid[i]
            objective(ap) = sqrt(a - ap) + β * interp(ap, agrid, V)
            pol[i], Vnew[i] = golden_max(objective, agrid[1], a)
        end
        push!(errs, maximum(abs.(Vnew .- V)))
        V .= Vnew
        errs[end] < tol && return (; V, pol, errs)
    end
    error("VFI did not converge in $maxit iterations")
end

## Solve
gridA = collect(range(amin, amax, length = 1000))    # Ngrids = 1000 in Ex3_gridsearch.m
gridB = collect(range(amin, amax, length = 100))     # Ngrids = 100 in Ex3_GS.m

vfi_gridsearch(gridA[1:10], β); vfi_golden(gridB[1:10], β)   # compile first, so timings are fair
tA = @elapsed solA = vfi_gridsearch(gridA, β)
tB = @elapsed solB = vfi_golden(gridB, β)

## Accuracy against the closed form
function report(name, grid, sol, t = nothing)
    errV = maximum(abs.(sol.V .- V_true.(grid)))
    errg = maximum(abs.(sol.pol .- g_true.(grid)))
    timing = isnothing(t) ? "" : "   time = $(round(t, digits = 2)) s"
    println(rpad(name, 40), "iterations = ", lpad(length(sol.errs), 3), timing,
            "   max|V - V*| = ", round(errV, sigdigits = 3), "   max|a' - a'*| = ", round(errg, sigdigits = 3))
end
println()
report("A: grid search, 1000 points", gridA, solA, tA)
report("B: golden section, 100 points", gridB, solB, tB)

# Why VFI converges, and why it can be slow: T is a contraction with modulus β, so
# ||V_{i+1} - V_i|| shrinks by at least the factor β each iteration. The same property bounds the
# distance to the true fixed point: ||V_i - V|| ≤ β/(1-β) · ||V_i - V_{i-1}||.
println("\nContraction bound on the remaining error in V: β/(1-β)·tol = ", round(β / (1 - β) * 1e-6, sigdigits = 3))

## Where the remaining error comes from: grid placement
# That bound is tiny, so the errors above are approximation errors from the grid, not from
# stopping too early. They are largest near a = 0, where V(a) = A√a has an infinite slope: on a
# uniform grid the smallest positive cake is 0.01 (or 0.1), so the agent eats it in one go and the
# cake runs out in finite time (which is also why grid search stops after so few iterations).
# Placing more points near zero helps far more than adding points everywhere:
println("\nSame methods on a grid a = 10·u², u uniform on [0, 1] (denser near zero):")
curved(N) = collect(amax .* range(0, 1, length = N) .^ 2)
report("A: grid search, 1000 curved points", curved(1000), vfi_gridsearch(curved(1000), β))
report("B: golden section, 100 curved points", curved(100), vfi_golden(curved(100), β))

## Simulate 100 periods starting from the whole cake (as in the MATLAB codes)
function simulate_cake(policy, a0; T = 100)
    a = zeros(T + 1)
    a[1] = a0
    for t in 1:T
        a[t+1] = policy(a[t])
    end
    c = a[1:T] .- a[2:T+1]
    return a, c
end
aA, cA = simulate_cake(a -> interp(a, gridA, solA.pol), amax)  # on the grid, interp is exact
aB, cB = simulate_cake(a -> interp(a, gridB, solB.pol), amax)
aT, cT = simulate_cake(g_true, amax)

## Plots: the four figures of the MATLAB codes, with the closed form added
style = (linewidth = 2, legend = :best)
p1 = plot(0:100, [aA aB aT]; title = "Leftover cake", label = ["grid search" "golden section" "exact"],
          ls = [:solid :dash :dot], style...)
p2 = plot(0:99, [cA cB cT]; title = "Cake consumption", label = "", ls = [:solid :dash :dot], style...)
p3 = plot(gridA, solA.V; title = "Value function V(a)", label = "grid search", style...)
plot!(p3, gridB, solB.V; label = "golden section", ls = :dash, style...)
plot!(p3, gridA, V_true.(gridA); label = "exact", ls = :dot, style...)
p4 = plot(gridA, solA.pol; title = "Policy function a'(a)", label = "grid search", style...)
plot!(p4, gridB, solB.pol; label = "golden section", ls = :dash, style...)
plot!(p4, gridA, g_true.(gridA); label = "exact", ls = :dot, style...)
fig = plot(p1, p2, p3, p4; layout = (2, 2), size = (900, 700), xlabel = ["t" "t" "a" "a"])
savefig(fig, joinpath(results_dir, "ex3_cake.png"))

# Convergence. The contraction theorem guarantees that the sup-norm distance falls at least as fast
# as β^i (dotted line). Here it falls faster, at rate β²: the error is proportional to √a, and the
# optimal policy a' = β²a shrinks the cake, so each application of T multiplies the error by
# β·√(β²) = β². Grid search drops to zero abruptly because on the grid the cake runs out in
# finite time, after which V stops changing.
it = 1:length(solB.errs)
fig = plot(it, log10.(solB.errs); label = "golden section", linewidth = 2,
           xlabel = "iteration", ylabel = "log10 ||V_{i+1} - V_i||", title = "Distance between successive iterates")
plot!(fig, 1:length(solA.errs), log10.(solA.errs); label = "grid search", ls = :dash, linewidth = 2)
plot!(fig, it, log10(solB.errs[1]) .+ (it .- 1) .* log10(β); label = "β^i (contraction bound)", ls = :dot, color = :black)
plot!(fig, it, log10(solB.errs[1]) .+ (it .- 1) .* log10(β^2); label = "β^2i", ls = :dashdot, color = :gray)
savefig(fig, joinpath(results_dir, "ex3_convergence.png"))
println("Figures saved in ", results_dir)
