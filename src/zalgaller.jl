using LinearAlgebra
using StaticArrays

struct Cone
    α::Float64               # angle of cone opening from center
    R::SMatrix{3,3,Float64}  # cone rotation matrix from opening directly up
    S::SVector{3,Float64}    # cone shift from apex at origin
    Ri::SMatrix{3,3,Float64} # inverse rotation
end

# different cone constructors
Cone(α::Number, R::AbstractMatrix, S::AbstractVector) = Cone(α, R, S, inv(R))
Cone(α::Number, R::Matrix) = Cone(α, R, SVector(0.,0,0), inv(R))
Cone(α::Number, S::Vector) = Cone(α, SMatrix{3,3}([1. 0 0; 0 1 0; 0 0 1]),
                                  S, SMatrix{3,3}([1. 0 0; 0 1 0; 0 0 1]))
Cone(α::Number) = Cone(α, SVector(0.,0,0))
Cone() = Cone(pi/4)

# shortest path on C from point P to Q
function path(C::Cone, P::Vector, Q::Vector, num_samples)
    # untransform
    TiP = C.Ri * (P .- C.S)
    TiQ = C.Ri * (Q .- C.S)

    # points in unrolled space
    rp = norm(TiP[1:2]) / sin(C.α)
    rq = norm(TiQ[1:2]) / sin(C.α)
    θp = atan(TiP[2], TiP[1]) * sin(C.α)
    θq = atan(TiQ[2], TiQ[1]) * sin(C.α)
    xp = rp * cos(θp)
    xq = rq * cos(θq)
    yp = rp * sin(θp)
    yq = rq * sin(θq)

    # path in unrolled space
    ts = range(0, 1, length = num_samples)
    xs = (1 .- ts) .* xp .+ ts .* xq
    ys = (1 .- ts) .* yp .+ ts .* yq

    # path in untransformed cylindrical coordinates
    θs = atan.(ys, xs) ./ sin(C.α)
    rs = sqrt.(xs.^2 + ys.^2) .* sin(C.α)
    zs = rs .* cot(C.α)

    points = hcat([[rs[i]*cos(θs[i]), rs[i]*sin(θs[i]), zs[i]] for i in eachindex(rs)]...)
    return C.R * points .+ C.S
end

# point over the belt and on the cone (see zalgaller paper) at angle θ in xy-plane
belt(θ; α = pi/2 - 1.1421) = [(cos(θ) - 1) / (cos(θ)*tan(α)^2 + 1) + 1
                              sec(α)*sin(θ) / (cos(θ)*tan(α)^2 + 1)
                              0]

# construction of the Zalgaller path with 4n-3 sample points
function zalgaller(n = 51; α = pi/2 - 1.1421, v = 2.65558)
    c = Cone(α, SMatrix{3,3}([cos(α) 0 -sin(α);
                              0      1 0;
                              sin(α) 0 cos(α)]),
             SVector(1, 0, -cot(α)))
    p1 = [1, 0, cot(α)]
    p2 = belt(atan(sec(α)*tan(v)) + pi)
    p1p2 = path(c, p1, p2, n)
    ts = range(atan(sec(α)*tan(v)) + pi, pi, length = n)
    p2p3 = hcat(belt.(ts)[2:end]...)
    p1p3 = hcat(p1p2,p2p3)
    p3p5 = p1p3[:, end:-1:1] .* [1, -1, -1]
    p1p5 = hcat(p1p3, p3p5[:, 2:end])

    return p1p5
end
