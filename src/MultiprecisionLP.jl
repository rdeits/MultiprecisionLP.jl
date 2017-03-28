module MultiprecisionLP

using NNLS
import Base: convert
export LinearProgram,
       PNNLSProblem,
       NNLSProblem,
       solve,
       solve_multiprecision

immutable PNNLSProblem{T, QR}
    A::Matrix{T}
    B::Matrix{T}
    c::Vector{T}
    B_QR::QR
end

function PNNLSProblem{T}(A::AbstractMatrix{T}, B::AbstractMatrix{T}, c::AbstractVector{T})
    B_QR = qrfact(B)
    PNNLSProblem{T, typeof(B_QR)}(A, B, c, B_QR)
end

immutable LinearProgram{T}
    f::Vector{T}
    G::Matrix{T}
    W::Vector{T}
end

immutable NNLSProblem{T}
    A::Matrix{T}
    b::Vector{T}
end


function convert{T1, T2}(::Type{NNLSProblem{T1}}, p::PNNLSProblem{T2})
    A = convert(Matrix{T1}, p.A)
    B = convert(Matrix{T1}, p.B)
    c = convert(Vector{T1}, p.c)
    Abar = A - B * (p.B_QR \ A)
    bbar = c - B * (p.B_QR \ c)
    NNLSProblem{T1}(Abar, bbar)
end

convert{T}(::Type{NNLSProblem}, p::PNNLSProblem{T}) = convert(NNLSProblem{T}, p)

function convert{T1, T2}(::Type{PNNLSProblem{T1}}, lp::LinearProgram{T2})
    q, n = size(lp.G)
    A = [lp.W'       zeros(1, q)
         zeros(q, q) eye(q)
         lp.G'       zeros(n, q)]
    B = [lp.f'
         lp.G
         zeros(n, n)]
    c = vcat(0, lp.W, -lp.f)
    PNNLSProblem(convert(Matrix{T1}, A),
        convert(Matrix{T1}, B),
        convert(Vector{T1}, c))
end

convert{T}(::Type{PNNLSProblem}, lp::LinearProgram{T}) = convert(PNNLSProblem{T}, lp)

convert{T1, T2}(::Type{LinearProgram{T1}}, lp::LinearProgram{T2}) = LinearProgram{T1}(lp.f, lp.G, lp.W)

function solve(p::NNLSProblem)
    work = NNLSWorkspace(p.A, p.b)
    nnls!(work)
    work.x, work.rnorm
end

function solve(p::PNNLSProblem)
    v, residual = solve(convert(NNLSProblem, p))
    u = -(p.B_QR \ (p.A * v - p.c))
    v, u, residual
end

function solve(p::LinearProgram)
    v, u, residual = solve(convert(PNNLSProblem, p))
    y = v[1:length(p.W)]
    s = v[(length(p.W) + 1):end]
    y, s, u, residual
end

function _solve_multiprecision(p::LinearProgram{BigFloat}, tolerance=1e-12)
    lastresidual = BigFloat(Inf)
    precision = 100
    while true
        y, s, u, residual = setprecision(BigFloat, precision) do
            solve(p)
        end
        @show residual
        is_feasible = residual < tolerance
        insufficient_progress = residual >= lastresidual^(3/2)
        at_iteration_limit = precision > 1000
        if is_feasible || insufficient_progress || at_iteration_limit
            return y, s, u, residual
        else
            precision += 100
            lastresidual = residual
        end
    end
end

function solve_multiprecision{T}(p::LinearProgram{T}, tolerance=1e-12)
    y, s, u, residual = solve(p)
    @show residual
    if residual < tolerance
        return y, s, u, residual, true
    else
        ybig, sbig, ubig, rbig = _solve_multiprecision(LinearProgram{BigFloat}(p), tolerance)
        return T.(ybig), T.(sbig), T.(ubig), T(rbig), rbig < tolerance
    end
end

end