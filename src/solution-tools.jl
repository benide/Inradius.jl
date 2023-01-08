"inradius of the path x"
R(x) = chebyshev(x)[1]

"length of the path x"
L(x) = @views sum(norm(x[:,i] - x[:,i+1]) for i=1:size(x,2)-1)

"length divided by radius of path x, objective function of our optimization"
LR(x) = L(x)/R(x)

"length of the closed path x where the final point is attached to the first"
Lc(x) = L(x) + norm(view(x,:,1) - view(x,:,size(x,2)))

"length/inradius for closed paths"
LcR(x) = Lc(x)/R(x)

# really basic forward difference method, modifies dx in place
function grad!(dx,f,x,epsilon=6e-6)
    fx = f(x)
    x1 = copy(x)
    for i in eachindex(dx)
        x1[i] += epsilon
        dx[i] = (f(x1) - fx) / epsilon
        x1[i] = x[i]
    end
    nothing
end
dLR(x) = (dx = similar(x); grad!(dx, LR, x); dx)
dLcR(x) = (dx = similar(x); grad!(dx, LcR, x); dx)
dLR!(dx,x) = grad!(dx, LR, x)
dLcR!(dx,x) = grad!(dx, LcR, x)

"""
    proj!(x)

Scale the path to inradius 1 and shift so that the center of the
contained unit sphere is at the origin.

After this projection, rotation and reflection will still produce
equivalent paths. See `normalize_path!(x)` and
`normalize_path_closed!(x)` to completely mod out equivalent paths.
"""
proj!(x) = ((r,c) = chebyshev(x); x .-= c; x ./= r;)

function local_minimum!(dx, x, f, df!, proj!; iterations = 500)
    proj!(x)
    for i=1:iterations
        df!(dx,x)
        alpha = linesearch!(x, f, -dx)
        proj!(x)
        iszero(alpha) && break
    end
    nothing
end

function linesearch!(x, f::Function, dir, m = -norm(dir); alpha_max = 1.0, c=0.1, tau=0.8, max_iter=75)
    t = -c*m
    alpha = alpha_max
    start_val = f(x)
    i = 0
    while start_val - f(x + alpha * dir) < alpha*t
        (i += 1) >= max_iter && return 0.0
        alpha *= tau
    end
    x .+= alpha * dir
    return alpha
end

function perturb_path!(dx, x::AbstractArray{<:Real,2}, df!, proj!,
                       diff_str = 5, diff_time = 1, diff_dt = 1e-3)
    σ = diff_str * rand()
    Time = diff_time * rand()
    for j = 1:max(fld(Time, diff_dt),1)
        df!(dx, x) # write gradient to preallocated array dx
        # note that dW = randn(size(x)) * diff_dt
        x .-= dx * diff_dt - σ * randn(size(x)) * diff_dt
        proj!(x)
    end
    nothing
end
