# Example 4 (Section 10.5): stochastic neoclassical growth model with labor, solved by VFI.
#
#     V(K, z) = max_{K', H}  log C - ξ H + β Σ_{z'} π(z'|z) V(K', z')
#     s.t.      C = z K^α H^(1-α) + (1-δ) K - K'
#
# Julia version of Ex4_gridsearch.m (Method A) and Ex4_GS.m (Method B), with Ex4_labor.m and
# Ex4_location.m folded in. Parameters follow the MATLAB codes (α = 1/3, δ = 0.0233); the slides
# list α = 0.36, δ = 0.025 and a log(1-H) utility term instead.
#
# Run from this folder with `julia ex4_brock_mirman.jl`, or step through the `##` cells in VS Code.

include("vfi_tools.jl")
using Random, Statistics, Printf, Plots

results_dir = joinpath(@__DIR__, "Results")
mkpath(results_dir)

## Parameters
β, δ, α = 0.99, 0.0233, 1/3
Hss = 1/3
Kstar = ((1/β - 1 + δ) / α)^(1 / (α - 1)) * Hss        # steady-state K when H = 1/3
ξ = (1 - α) * Kstar^α * Hss^(-α) / (Kstar^α * Hss^(1 - α) - δ * Kstar)   # makes H = 1/3 in steady state

zgrid = [0.985, 1.015]              # Z_L, Z_H
P = [0.95 0.05                      # P[z, z'] = probability of moving from z to z'
     0.05 0.95]
# Exercise: replace the two lines above with `y, P = rouwenhorst(7, 0.95, 0.007); zgrid = exp.(y)`.
# Nothing else in the script has to change.

par = (; β, δ, α, ξ, zgrid, P)

## The static labor choice (Ex4_labor.m)
# For given (K, K', z), H solves the first-order condition (1-α) z K^α H^(-α) / C = ξ.
# It does not involve V, so this inner problem is separate from the dynamic one.
function labor(K, Kp, z, par; lo = 0.2, hi = 0.5, tol = 1e-8)
    (; α, δ, ξ) = par
    consumption(H) = max(z * K^α * H^(1 - α) + (1 - δ) * K - Kp, 1e-5)
    foc(H) = (1 - α) * z * K^α * H^(-α) / consumption(H) - ξ
    flo = foc(lo)
    while hi - lo > tol             # bisection (Section 10.2); MATLAB uses tol = 1e-4
        mid = (lo + hi) / 2
        fmid = foc(mid)
        if sign(fmid) == sign(flo)
            lo, flo = mid, fmid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

# Period return F(K, K', z) with H chosen optimally; also returns H.
function period_return(K, Kp, z, par)
    (; α, δ, ξ) = par
    H = labor(K, Kp, z, par)
    C = max(z * K^α * H^(1 - α) + (1 - δ) * K - Kp, 1e-5)
    return log(C) - ξ * H, H
end

## Method A: K' on the grid, grid search (Ex4_gridsearch.m)
# Ex4_gridsearch.m re-solves the labor bisection for every (K, K', z) in every iteration. But the
# period return does not depend on V, so we can compute it once, before iterating. This is the
# main reason the Julia version takes about a second rather than many minutes.
#
# R[j, i, z] = F(Kgrid[i], Kgrid[j], zgrid[z]). The choice index j comes first because Julia
# arrays are stored column by column, which keeps the inner loop over j fast.
function return_array(Kgrid, par)
    (; zgrid) = par
    N, Nz = length(Kgrid), length(zgrid)
    R, Hchoice = zeros(N, N, Nz), zeros(N, N, Nz)
    for z in 1:Nz, i in 1:N, j in 1:N
        R[j, i, z], Hchoice[j, i, z] = period_return(Kgrid[i], Kgrid[j], zgrid[z], par)
    end
    return R, Hchoice
end

# `howard = m` adds m Howard improvement steps per iteration: holding the policy fixed, update
# V ← F_g + β P V_g, which is cheap because there is no maximization.
function vfi_gridsearch(R, Hchoice, Kgrid, par; tol = 1e-4, howard = 0, maxit = 10_000)
    (; β, zgrid, P) = par
    N, Nz = length(Kgrid), length(zgrid)
    V, Vnew = zeros(N, Nz), zeros(N, Nz)
    pol = ones(Int, N, Nz)
    errs = Float64[]
    for _ in 1:maxit
        EV = V * P'                 # EV[j, z] = Σ_{z'} P[z, z'] V[j, z'] = E[V(K'_j, z') | z]
        for z in 1:Nz, i in 1:N
            best, jbest = -Inf, 1
            for j in 1:N
                val = R[j, i, z] + β * EV[j, z]
                if val > best
                    best, jbest = val, j
                end
            end
            Vnew[i, z], pol[i, z] = best, jbest
        end
        push!(errs, maximum(abs.(Vnew .- V)))
        V .= Vnew
        if errs[end] < tol
            Kpol = Kgrid[pol]
            Hpol = [Hchoice[pol[i, z], i, z] for i in 1:N, z in 1:Nz]
            return (; V, Kpol, Hpol, errs)
        end
        for _ in 1:howard
            EV = V * P'
            for z in 1:Nz, i in 1:N
                V[i, z] = R[pol[i, z], i, z] + β * EV[pol[i, z], z]
            end
        end
    end
    error("VFI did not converge in $maxit iterations")
end

## Method B: K' anywhere in [Kmin, Kmax], golden section + interpolation (Ex4_GS.m)
# Ex4_GS.m interpolates V(·, z_L) and V(·, z_H) separately and then takes the expectation.
# Because both operations are linear, interpolating the expectation EV gives the same number.
function vfi_golden(Kgrid, par; tol = 1e-4, maxit = 10_000)
    (; β, zgrid, P) = par
    N, Nz = length(Kgrid), length(zgrid)
    V, Vnew = zeros(N, Nz), zeros(N, Nz)
    Kpol = zeros(N, Nz)
    errs = Float64[]
    for _ in 1:maxit
        EV = V * P'
        for z in 1:Nz
            EVz = view(EV, :, z)
            for i in 1:N
                K = Kgrid[i]
                objective(Kp) = period_return(K, Kp, zgrid[z], par)[1] + β * interp(Kp, Kgrid, EVz)
                Kpol[i, z], Vnew[i, z] = golden_max(objective, Kgrid[1], Kgrid[end])
            end
        end
        push!(errs, maximum(abs.(Vnew .- V)))
        V .= Vnew
        if errs[end] < tol
            Hpol = [labor(Kgrid[i], Kpol[i, z], zgrid[z], par) for i in 1:N, z in 1:Nz]
            return (; V, Kpol, Hpol, errs)
        end
    end
    error("VFI did not converge in $maxit iterations")
end

## Solve
Kmin, Kmax = 0.9Kstar, 1.1Kstar
gridA = collect(range(Kmin, Kmax, length = 400))   # Ngrids = 400 in Ex4_gridsearch.m
gridB = collect(range(Kmin, Kmax, length = 50))    # Ngrids = 50 in Ex4_GS.m

vfi_gridsearch(return_array(gridA[1:10], par)..., gridA[1:10], par)   # compile first, so
vfi_golden(gridB[1:10], par)                                          # timings are fair
tR = @elapsed R, Hchoice = return_array(gridA, par)
tA = @elapsed solA = vfi_gridsearch(R, Hchoice, gridA, par)
tH = @elapsed solH = vfi_gridsearch(R, Hchoice, gridA, par; howard = 20)
tB = @elapsed solB = vfi_golden(gridB, par)

println()
@printf("K* = %.3f, ξ = %.3f\n", Kstar, ξ)
@printf("A: precompute F(K, K', z) on 400×400×2                   %5.2f s\n", tR)
@printf("A: grid search, 400 points             %4d iterations  %5.2f s\n", length(solA.errs), tA)
@printf("A + 20 Howard steps per iteration      %4d iterations  %5.2f s   max|ΔV| vs A = %.1e\n",
        length(solH.errs), tH, maximum(abs.(solH.V .- solA.V)))
@printf("B: golden section, 50 points           %4d iterations  %5.2f s\n", length(solB.errs), tB)

# Why 927 iterations: the distance shrinks by about β = 0.99 per step, and 0.99^927 ≈ 1e-4.
# The contraction bound then says ||V_i - V|| ≤ β/(1-β)·1e-4 = 0.01, which is why the Howard
# solution (a different path to the same fixed point) differs from plain VFI by about 1e-3.

# The two methods should agree up to the grid spacing of Method A.
KpolB_on_A = [interp(K, gridB, view(solB.Kpol, :, z)) for K in gridA, z in 1:2]
@printf("max |K'_A - K'_B| on the 400-point grid = %.4f (grid step %.4f)\n",
        maximum(abs.(KpolB_on_A .- solA.Kpol)), gridA[2] - gridA[1])

## Simulate 3000 periods (as in the MATLAB codes)
# The random draws differ from MATLAB, so the numbers will not match the book exactly.
function simulate(Kpolicy, par; T = 3000, K0, z0 = 1, seed = 1234)
    (; α, δ, zgrid, P) = par
    rng = Xoshiro(seed)
    K = zeros(T + 1); Y, C, I, H, Z = (zeros(T) for _ in 1:5)
    K[1], zi = K0, z0
    for t in 1:T
        K[t+1] = Kpolicy(K[t], zi)
        H[t] = labor(K[t], K[t+1], zgrid[zi], par)
        Y[t] = zgrid[zi] * K[t]^α * H[t]^(1 - α)
        C[t] = Y[t] + (1 - δ) * K[t] - K[t+1]
        I[t] = K[t+1] - (1 - δ) * K[t]
        Z[t] = zgrid[zi]
        zi = something(findfirst(cumsum(P[zi, :]) .> rand(rng)), size(P, 1))   # draw z'
    end
    return (; Y, C, I, H, K = K[1:T], Z)
end

# Policies as functions of (K, z): linear interpolation between grid points.
# For Method A, K stays on the grid, where interpolation returns the grid-search choice exactly.
KpolicyA(K, z) = interp(K, gridA, view(solA.Kpol, :, z))
KpolicyB(K, z) = interp(K, gridB, view(solB.Kpol, :, z))
simA = simulate(KpolicyA, par; K0 = gridA[200])   # Ex4_gridsearch.m starts at grid point 200
simB = simulate(KpolicyB, par; K0 = Kstar)        # Ex4_GS.m starts at K*

## Business-cycle statistics: log, HP-filter (λ = 1600), drop the first 99 periods
function cycle_stats(sim; burn = 99, λ = 1600)
    cyc = map(x -> (lx = log.(x[burn+1:end]); lx .- hp_trend(lx, λ)), sim)
    return (; (name => (sd = 100std(c), corrY = cor(c, cyc.Y)) for (name, c) in pairs(cyc))...)
end
statsA, statsB = cycle_stats(simA), cycle_stats(simB)
println("\nHP-filtered cyclical components (std in %, correlation with Y)")
println("          Method A            Method B")
for v in keys(statsA)
    @printf("  %-2s   %5.2f%%  %5.2f      %5.2f%%  %5.2f\n", v,
            statsA[v].sd, statsA[v].corrY, statsB[v].sd, statsB[v].corrY)
end

## Plots
style = (linewidth = 2,)
lbl = ["z_L" "z_H"]
p1 = plot(gridA, solA.V; title = "Value function V(K, z)", label = lbl, style...)
p2 = plot(gridA, solA.Kpol .- gridA; title = "Net investment K'(K, z) - K", label = lbl, style...)
plot!(p2, gridB, solB.Kpol .- gridB; label = ["z_L (golden)" "z_H (golden)"], ls = :dash, style...)
hline!(p2, [0]; color = :black, label = "")
p3 = plot(gridA, solA.Hpol; title = "Hours H(K, z)", label = lbl, style...)
p4 = plot(1:length(solA.errs), log10.(solA.errs); title = "log10 ||V_{i+1} - V_i||",
          label = "plain VFI", style...)
plot!(p4, 1:length(solH.errs), log10.(solH.errs); label = "with Howard steps", style...)
fig = plot(p1, p2, p3, p4; layout = (2, 2), size = (900, 700), xlabel = ["K" "K" "K" "iteration"])
savefig(fig, joinpath(results_dir, "ex4_brock_mirman.png"))

t = 101:400
fig = plot(t, [simB.Y[t] ./ mean(simB.Y) simB.C[t] ./ mean(simB.C) simB.H[t] ./ mean(simB.H)];
           label = ["Y" "C" "H"], title = "Simulated path (relative to sample mean)", xlabel = "quarter",
           linewidth = 1.5)
savefig(fig, joinpath(results_dir, "ex4_simulation.png"))
println("\nFigures saved in ", results_dir)
