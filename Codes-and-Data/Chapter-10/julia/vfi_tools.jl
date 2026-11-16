# Building blocks from Sections 10.1-10.4, translated from the MATLAB files in ../
#
#   golden_max   <- Ex1_GS.m   (golden-section search, Section 10.3)
#   locate       <- Ex4_location.m
#   interp       <- linear interpolation used inside Ex3_GS.m / Ex4_GS.m (Section 10.1)
#   tauchen      <- Ex2_tauchen.m      (Section 10.4)
#   rouwenhorst  <- Ex2_rouwenhorst.m  (Section 10.4)
#   hp_trend     <- hpfilt.m
#
# Only the standard library is used, so `include("vfi_tools.jl")` works without installing anything.

using LinearAlgebra, SparseArrays

"""
    golden_max(f, a, c; tol = 1e-5) -> (xmax, fmax)

Maximize `f` on `[a, c]` by golden-section search (Ex1_GS.m). Each step discards the part of the
bracket that cannot contain the maximum of a unimodal function, shrinking it by about 38%.
"""
function golden_max(f, a, c; tol = 1e-5)
    gold = (3 - sqrt(5)) / 2          # 0.38197, as in the MATLAB codes
    lo, hi = a, c
    flo, fhi = f(lo), f(hi)
    b = lo + gold * (hi - lo)
    fb = f(b)
    while hi - lo > tol
        if b - lo > hi - b            # try a new point in the longer segment
            m = b - gold * (b - lo)
            fm = f(m)
            if fm > fb
                hi, b, fb = b, m, fm
            else
                lo = m
            end
        else
            m = b + gold * (hi - b)
            fm = f(m)
            if fm > fb
                lo, b, fb = b, m, fm
            else
                hi = m
            end
        end
    end
    # Corner solutions: the MATLAB codes check the end points before searching; checking them
    # afterwards does the same job and also catches a maximum sitting exactly on a bound.
    flo > fb && flo >= fhi && return (a, flo)
    fhi > fb && return (c, fhi)
    return (b, fb)
end

"""
    locate(x, grid) -> j

Index `j` with `grid[j] <= x <= grid[j+1]`, clamped to `1:length(grid)-1` so that points outside
the grid are extrapolated from the nearest segment. Ex4_location.m scans the grid linearly;
`searchsortedlast` does a binary search.
"""
locate(x, grid) = clamp(searchsortedlast(grid, x), 1, length(grid) - 1)

"""
    interp(x, grid, v)

Linear interpolation of the values `v` (defined on `grid`) at the point `x`.
"""
function interp(x, grid, v)
    j = locate(x, grid)
    w = (x - grid[j]) / (grid[j+1] - grid[j])
    return (1 - w) * v[j] + w * v[j+1]
end

# Standard normal CDF via the complementary error function (Numerical Recipes `erfcc`,
# relative error < 1.2e-7). With Distributions.jl installed you would use `cdf(Normal(), x)`.
function _erfc(x)
    z = abs(x)
    t = 1 / (1 + z / 2)
    r = t * exp(-z^2 - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
        t * (-0.82215223 + t * 0.17087277)))))))))
    return x >= 0 ? r : 2 - r
end
normcdf(x) = _erfc(-x / sqrt(2)) / 2

"""
    tauchen(n, ρ, σ; m = 3) -> (y, P)

Discretize y' = ρ y + ε, ε ~ N(0, σ²), into `n` states spanning ±m unconditional standard
deviations (Ex2_tauchen.m). `P[i, j]` is the probability of moving from state i to state j.
"""
function tauchen(n, ρ, σ; m = 3)
    σy = σ / sqrt(1 - ρ^2)
    y = collect(range(-m * σy, m * σy, length = n))
    step = y[2] - y[1]
    P = zeros(n, n)
    for i in 1:n
        P[i, 1] = normcdf((y[1] - ρ * y[i] + step / 2) / σ)
        for j in 2:n-1
            P[i, j] = normcdf((y[j] - ρ * y[i] + step / 2) / σ) -
                      normcdf((y[j] - ρ * y[i] - step / 2) / σ)
        end
        P[i, n] = 1 - normcdf((y[n] - ρ * y[i] - step / 2) / σ)
    end
    return y, P
end

"""
    rouwenhorst(n, ρ, σ) -> (y, P)

Same task with the Rouwenhorst method (Ex2_rouwenhorst.m), which matches the persistence and
the unconditional variance exactly and is preferred when ρ is close to 1.
"""
function rouwenhorst(n, ρ, σ)
    σy = σ / sqrt(1 - ρ^2)
    p = (1 + ρ) / 2
    ψ = sqrt(n - 1) * σy
    y = collect(range(-ψ, ψ, length = n))
    P = [p 1-p; 1-p p]
    for k in 2:n-1                     # grow the matrix from k×k to (k+1)×(k+1)
        Pn = zeros(k + 1, k + 1)
        Pn[1:k, 1:k]         .+= p .* P
        Pn[1:k, 2:k+1]       .+= (1 - p) .* P
        Pn[2:k+1, 1:k]       .+= (1 - p) .* P
        Pn[2:k+1, 2:k+1]     .+= p .* P
        Pn[2:k, :] ./= 2               # interior rows were counted twice
        P = Pn
    end
    return y, P
end

"""
    hp_trend(y, λ = 1600)

Hodrick-Prescott trend of the series `y` (hpfilt.m): solves (I + λ D'D) τ = y, where D is the
second-difference matrix. The cyclical component is `y - hp_trend(y, λ)`.
"""
function hp_trend(y, λ = 1600)
    T = length(y)
    D = spdiagm(T - 2, T, 0 => ones(T - 2), 1 => -2ones(T - 2), 2 => ones(T - 2))
    return (I + λ * (D' * D)) \ y
end
