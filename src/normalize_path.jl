# Normalizing means moving the path so that the largest sphere in its
# convex hull is centered at the origin, scaling the path to inradius
# 1, rotating so that the first point on the path is at [1,0,h] for
# some value h, and reflecting so that the first turn goes around the
# z-axis counter-clockwise when viewed from above.

# For open paths, the mathematical description leads fairly directly
# to an algorithm. For closed, there is no "starting point"
# mathematically speaking, and reflections are not necessary for
# modding out. Instead, we minimize the distance of the points
# defining the curve from what we know the true solution to be. This
# has the same effect.

# Generally, everything after the definition of `normalize_path!`
# becomes a bit of a mess, but the code works well.



# Symbolics.jl results in faster calculations for this than doing it by hand
using Symbolics
@variables a,b,c

A = [cos(a) -sin(a) 0
     sin(a)  cos(a) 0
     0       0      1]
B = [ cos(b)  0 sin(b)
      0       1 0
     -sin(b)  0 cos(b)]
C = [1 0       0
     0 cos(c) -sin(c)
     0 sin(c)  cos(c)]

rot_mat = eval(build_function(A*B*C,[a,b,c])[1])
rot_reflect = eval(build_function(A*B*C*[-1 0 0; 0 1 0; 0 0 1],[a,b,c])[1])

function normalize_path!(x)
    # center = chebyshev(x)[2]
    # x .-= center
    proj!(x)
    γ = atan((x[2,1]-x[2,end]) / (x[3,1]-x[3,end]))
    β = atan(cos(γ) * (x[1,end]-x[1,1]) / (x[3,1]-x[3,end]))
    α = atan((x[3,1] * tan(γ) - x[2,1]) / (x[1,1] * cos(β) / cos(γ) + (x[2,1] * tan(γ) + x[3,1]) * sin(β)))
    x .= rot_mat([α,β,γ]) * x
    reflection_mat = [(x[1,1] < 0 ? -1 : 1) 0 0
                      0 (x[2,2] < 0 ? -1 : 1) 0
                      0 0 (x[3,1] < 0 ? -1 : 1)]
    x .= reflection_mat * x
    nothing
end

function normalize_path_closed!(x)
    # center = chebyshev(x)[2]
    # x .-= center
    proj!(x)
    l = Lc(x)
    y = pts_along_path_c(x, range(0, 1, length = 25)[1:24])
    a = zeros(3)
    da = zeros(3)
    d(a) = dist_closed_path_normalization(rot_mat(a) * y)
    dd!(da, a) = grad!(da, d, a, pi * 1e-5)
    local_minimum!(da, a, d, dd!, identity, iterations = 500)
    # local_minimum!(da, a,
    #                a -> dist_closed_path_normalization(rot_mat(a) * x),
    #                (da, a) -> grad!(da,
    #                                 a -> dist_closed_path_normalization(rot_mat(a) * x),
    #                                 a, pi * 1e-5),
    #                identity, iterations = 500)
    x .= rot_mat(a) * x
    if minimum(norm(c - [1,0,1]) for c in eachcol(rot_mat(a)*y)) < 0.5
        x .= rot_mat([pi/2, 0, 0]) * x
    end
    nothing
end

d_closed_path_normalization(x,t) = minimum(sum((x .- y) .^ 2)
                                           for y in ([1, -cos(t), sin(t)],
                                                     [-1, -cos(t), sin(t)],
                                                     [cos(t), 1, sin(t)],
                                                     [cos(t), -1, sin(t)]))
_test_ts(x) = x[3] == 0 ? [0, pi] : [0, pi, acot(-x[2]/x[3]), acot(x[1]/x[3])]
test_ts(x) = [t < 0 ? t + pi : t
              for t in _test_ts(x)]
dist_closed_path_normalization(x::AbstractVector{<:Real}) = minimum(d_closed_path_normalization(x,t) for t in test_ts(x))
dist_closed_path_normalization(x::AbstractMatrix{<:Real}) = sum(dist_closed_path_normalization(c) for c in eachcol(x))

function pt_along_path_c(x,_l)
    @assert 0 ≤ _l ≤ 1
    partials = vcat(0., [L(x[:,1:i]) for i=2:size(x,2)], Lc(x))
    l = _l * partials[end]
    i = findfirst(l .< partials)
    if isnothing(i)
        x[:,1]
    elseif i <= size(x,2)
        t = (l - partials[i-1]) / (partials[i] - partials[i-1])
        x[:,i-1] * (1-t) + x[:,i] * t
    else
        t = (l - partials[i-1]) / (partials[i] - partials[i-1])
        x[:,end] * (1-t) + x[:,1] * t
    end
end

function pts_along_path_c(x, _ls)
    hcat([pt_along_path_c(x,l) for l in _ls]...)
end
