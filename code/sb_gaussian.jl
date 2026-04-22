# sb_gaussian.jl — Bit-flip MH experiments on the Gaussian Schrödinger bridge.
#
# Setup: source N(0, Iᵈ), target N(e₁, Σ_AR1) with ρ=0.9.
# Cost c(x,y) = ‖x−y‖², temperature ε.
#
# Analytic cross-covariance (derived from Sinkhorn fixed-point for Gaussians):
#   C² + (ε/2)C = Σ  →  C = √(Σ + (ε/4)²I) − (ε/4)I
#
# Run from project root:
#   julia --project=. code/sb_gaussian.jl

using LinearAlgebra, Statistics, Random, Printf
using DataFrames, CSV

const AR_RHO  = 0.9
const N_BURN  = 1000   # burn-in iterations
const N_MEAS  = 1000   # thinned measurements
const THIN    = 10     # iters between measurements

# ── Distributions ─────────────────────────────────────────────────────────────

ar1_cov(d) = Float64[AR_RHO^abs(i - j) for i in 1:d, j in 1:d]

function matsqrt(A::AbstractMatrix)
    F = eigen(Symmetric(A))
    F.vectors * Diagonal(sqrt.(max.(F.values, 0.0))) * F.vectors'
end

# C = √(Σ + α²I) − αI  where α = ε/4.
# Derived from the fixed-point P₁₂ = −2/ε I of the joint Gaussian precision,
# giving C² + (ε/2)C = Σ.
function analytic_cross_cov(Σ::AbstractMatrix, ε::Float64)
    α = ε / 4.0
    matsqrt(Σ .+ α^2 .* I(size(Σ, 1))) .- α .* I(size(Σ, 1))
end

# ── Ground-truth moments ───────────────────────────────────────────────────────

function analytic_gt(d::Int, ε::Float64)
    Σ1 = ar1_cov(d)
    m1 = zeros(d); m1[1] = 1.0
    C  = analytic_cross_cov(Σ1, ε)

    # Joint: w = [X; Y] ~ N([0; m1], [[I, C]; [C, Σ1]])
    Id  = Matrix{Float64}(I, d, d)
    Z0  = zeros(d, d)
    Σ_w = [Id C; C Σ1]        # 2d × 2d
    μ_w = [zeros(d); m1]      # 2d

    # X·Y = wᵀ B w,  B = [[0, I/2]; [I/2, 0]]
    B   = [Z0 Id/2; Id/2 Z0]
    BΣ  = B * Σ_w
    Bμ  = B * μ_w              # = [m1/2; 0]

    # First moment: E[Z] = tr(BΣ) + μᵀBμ = tr(C)  (since μ_x = 0)
    EZ   = tr(BΣ) + dot(μ_w, Bμ)

    # Second moment: Var(Z) = 2tr((BΣ)²) + 4μᵀ BΣ Bμ
    BΣ2  = BΣ * BΣ
    VarZ = 2tr(BΣ2) + 4dot(μ_w, BΣ * Bμ)
    EZ2  = VarZ + EZ^2

    # Third moment via third central moment:
    #   E[(Z−EZ)³] = 8(tr((BΣ)³) + 3 μᵀ(BΣ)² Bμ)
    tcm  = 8(tr(BΣ2 * BΣ) + 3dot(μ_w, BΣ2 * Bμ))
    EZ3  = tcm + 3EZ * VarZ + EZ^3

    return (x0y0 = C[1, 1], x0y1 = C[1, 2],
            x1y0 = C[2, 1], x1y1 = C[2, 2],
            EZ = EZ, EZ2 = EZ2, EZ3 = EZ3)
end

# ── Bit-flip MH sampler ────────────────────────────────────────────────────────

# For bit k, half the particles (those with bit k = 0) are paired with
# their XOR-partners. Pairs are disjoint so can be processed in any order.
function make_pairs(M::Int, logN::Int)
    [begin
         ii = [i for i in 1:M if (((i - 1) >> k) & 1) == 0]
         jj = [(i - 1) ⊻ (1 << k) + 1 for i in ii]
         (ii, jj)
     end for k in 0:logN-1]
end

function shuffle_pairs!(X::Matrix{Float64}, Y::Matrix{Float64})
    d, M = size(X)
    @inbounds for i in M:-1:2
        j = rand(1:i)
        i == j && continue
        for k in 1:d
            X[k, i], X[k, j] = X[k, j], X[k, i]
            Y[k, i], Y[k, j] = Y[k, j], Y[k, i]
        end
    end
end

# Sweep over all bits, swapping the A array (X for x-phase, Y for y-phase).
# ΔE = 2/ε · (X[:,i]−X[:,j])·(Y[:,i]−Y[:,j]).  Same formula for both phases.
function bit_sweep!(A::Matrix{Float64}, X::Matrix{Float64}, Y::Matrix{Float64},
                    ε::Float64, pairs)
    d    = size(X, 1)
    hp   = size(X, 2) ÷ 2
    c2ε  = 2.0 / ε
    @inbounds for (ii, jj) in pairs
        for t in 1:hp
            i, j = ii[t], jj[t]
            dE = 0.0
            @simd for k in 1:d
                dE += (X[k, i] - X[k, j]) * (Y[k, i] - Y[k, j])
            end
            dE *= c2ε
            if dE <= 0.0 || rand() < exp(-dE)
                for k in 1:d
                    A[k, i], A[k, j] = A[k, j], A[k, i]
                end
            end
        end
    end
end

# One full iteration: x-sweep → shuffle → y-sweep → shuffle.
function do_iter!(X::Matrix{Float64}, Y::Matrix{Float64}, ε::Float64, pairs)
    bit_sweep!(X, X, Y, ε, pairs)
    shuffle_pairs!(X, Y)
    bit_sweep!(Y, X, Y, ε, pairs)
    shuffle_pairs!(X, Y)
end

# ── Test functions ─────────────────────────────────────────────────────────────

function test_fns(X::Matrix{Float64}, Y::Matrix{Float64})
    d, M = size(X)
    x0y0 = 0.0; x0y1 = 0.0; x1y0 = 0.0; x1y1 = 0.0
    EZ = 0.0; EZ2 = 0.0; EZ3 = 0.0
    @inbounds for k in 1:M
        x0y0 += X[1, k] * Y[1, k]
        x0y1 += X[1, k] * Y[2, k]
        x1y0 += X[2, k] * Y[1, k]
        x1y1 += X[2, k] * Y[2, k]
        dp = 0.0
        @simd for j in 1:d
            dp += X[j, k] * Y[j, k]
        end
        EZ  += dp
        EZ2 += dp * dp
        EZ3 += dp * dp * dp
    end
    inv_M = 1.0 / M
    return (x0y0 * inv_M, x0y1 * inv_M, x1y0 * inv_M, x1y1 * inv_M,
            EZ * inv_M, EZ2 * inv_M, EZ3 * inv_M)
end

# ── ESS via initial positive sequence ─────────────────────────────────────────

function ess_ips(chain::AbstractVector{Float64})
    n = length(chain)
    n <= 1 && return float(n)
    μ = mean(chain)
    v = 0.0
    @inbounds for x in chain; v += (x - μ)^2; end
    v /= (n - 1)
    v < 1e-14 && return float(n)
    ac_sum = 0.0
    @inbounds for k in 1:min(n ÷ 2, 500)
        s = 0.0
        for i in 1:n-k
            s += (chain[i] - μ) * (chain[i + k] - μ)
        end
        ρk = s / ((n - 1) * v)
        ρk <= 0.0 && break
        ac_sum += ρk
    end
    return n / max(1.0 + 2ac_sum, 1.0)
end

# ── Single experiment ──────────────────────────────────────────────────────────

function run_experiment(d::Int, M::Int, ε::Float64; seed::Int = 42)
    Random.seed!(seed)
    logN = round(Int, log2(M))
    Σ1   = ar1_cov(d)
    m1   = zeros(d); m1[1] = 1.0
    L    = cholesky(Symmetric(Σ1)).L

    X = randn(d, M)
    Y = L * randn(d, M) .+ m1

    pairs = make_pairs(M, logN)

    for _ in 1:N_BURN
        do_iter!(X, Y, ε, pairs)
    end

    meas = Matrix{Float64}(undef, N_MEAS, 7)
    for t in 1:N_MEAS
        for _ in 1:THIN
            do_iter!(X, Y, ε, pairs)
        end
        tf = test_fns(X, Y)
        meas[t, :] .= collect(tf)
    end
    return meas
end

# ── Main ───────────────────────────────────────────────────────────────────────

function main()
    dims  = [10, 30, 100]
    Ms    = [128, 1024, 8192]   # 2^7, 2^10, 2^13
    εs    = [0.05, 0.1, 1.0]
    fnms  = ["x0y0", "x0y1", "x1y0", "x1y1", "EZ", "EZ2", "EZ3"]
    n_fn  = length(fnms)

    results_dir = joinpath(@__DIR__, "..", "results")
    mkpath(results_dir)

    rows = NamedTuple[]

    for d in dims, M in Ms, ε in εs
        @printf("d=%-3d  M=%-5d  ε=%.2f  ...", d, M, ε); flush(stdout)
        t0   = time()
        gt   = analytic_gt(d, ε)
        gtv  = [gt.x0y0, gt.x0y1, gt.x1y0, gt.x1y1, gt.EZ, gt.EZ2, gt.EZ3]
        meas = run_experiment(d, M, ε)
        elapsed = time() - t0
        @printf("  %.1f s\n", elapsed)

        for (fi, fn) in enumerate(fnms)
            chain  = meas[:, fi]
            μ_hat  = mean(chain)
            σ_hat  = std(chain, corrected = true)
            gt_val = gtv[fi]
            bias   = μ_hat - gt_val
            var_   = var(chain,  corrected = true)
            mse    = bias^2 + var_
            ess_   = ess_ips(chain)
            mcse   = σ_hat / sqrt(max(ess_, 1.0))

            push!(rows, (
                d        = d,
                M        = M,
                ε        = ε,
                fn       = fn,
                gt       = gt_val,
                mean     = μ_hat,
                std      = σ_hat,
                bias     = bias,
                bias2    = bias^2,
                variance = var_,
                mse      = mse,
                ess      = ess_,
                mcse     = mcse,
            ))
        end
    end

    df = DataFrame(rows)
    out_path = joinpath(results_dir, "sb_gaussian.csv")
    CSV.write(out_path, df)

    # ── Print summary table ────────────────────────────────────────────────────
    println("\n", "="^110)
    println("GAUSSIAN SCHRÖDINGER BRIDGE EXPERIMENT RESULTS")
    println("Source: N(0,Iᵈ)   Target: N(e₁, AR1(ρ=0.9))   Burn-in: $N_BURN   Meas: $N_MEAS   Thin: $THIN")
    println("="^110)
    @printf("%-4s %-6s %-5s %-6s  %10s  %10s  %8s  %+9s  %9s  %9s  %7s  %8s\n",
            "d", "M", "ε", "fn",
            "gt", "mean", "std",
            "bias", "bias²", "variance", "MSE", "ESS", )
    println("-"^110)
    for r in rows
        @printf("%-4d %-6d %-5.2f %-6s  %+10.5f  %+10.5f  %8.5f  %+9.5f  %9.6f  %9.6f  %9.6f  %7.1f\n",
                r.d, r.M, r.ε, r.fn,
                r.gt, r.mean, r.std,
                r.bias, r.bias2, r.variance, r.mse, r.ess)
    end
    println("="^110)
    println("\nFull results written to: $out_path")
end

main()
