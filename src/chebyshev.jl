"""
    chebyshev(x)

Find the Chebyshev radius (i.e., the inradius) and center of a point
cloud given by the columns of x.
"""
function chebyshev(x)
# function chebyshev(x::AbstractMatrix{<:Real})

    # Shift the point cloud so that all components are positive.
    # This is to make standard form a little easier (no need to separate objective variables into positive and negative, doubling the size of the matrix)
    Ab = hull(x .- minimum(x))

    # Create the simplex tableau T to solve the linear program
    T = hcat(vcat(1.0, zeros(size(Ab,2))), # objective function variable
             vcat([0. 0 0], Ab[1:3, :]'),  # x,y,z center variables
             vcat(-1.0, ones(size(Ab,2))), # radius variable
             vcat(zeros(1,size(Ab,2)), I), # slack variables
             vcat(0., -1 .* Ab[4, :]))

    solve_tableau!(T)

    center = [T[findfirst(T[:,i] .== 1),end] + minimum(x) for i=2:4]
    radius = T[1,end]
    
    return radius, center
end


# TODO: check if Kyle Novak's "Numerical Methods for Scientific Computing" has better code than this.
function solve_tableau!(T)
    # T should be in the form:
    #    (1 c' 0)
    #    (0 A  b)
    # where we want to solve:
    #    min c·x subject to Ax = b, x ≥ 0
    
    # preallocate this array and reuse it
    ratios = fill(Inf, size(T,1))
    
    # pivot so that RHS is nonnegative
    for i in axes(T,1)
        @inbounds while T[i,end] < 0
            j0 = findfirst(view(T,i,:) .< 0)
            isnothing(j0) && error("infeasible linear program")
            fill!(ratios, Inf)
            for k = 2:size(T,1)
                (T[k,j0] > 0) && (T[k,end] > 0) && (ratios[k] = T[k,end] / T[k,j0])
            end
            ratios[i] = T[i,end] / T[i,j0]
            i0 = argmin(ratios)
            pivot!(T,i0,j0)
        end
    end

    # pivot so that objective coefficients are positive
    while minimum(T[1,:]) < 0
        j0 = findfirst(T[1,:] .< 0)
        fill!(ratios, Inf)
        @inbounds for k = 2:size(T,1)
            T[k,j0] > 0 && (ratios[k] = T[k,end] / T[k,j0])
        end
        i0 = argmin(ratios)
        i0 == 1 && error("unbounded linear prorgam")
        pivot!(T,i0,j0)
    end
    nothing
end


"""
    pivot!(T,i0,j0)

Used for linear programming. Pivots the tableau T at T[i0,j0].

This operation scales row i0 so that T[i0,j0] = 1, then subtracts
the proper multiple of that row from all other rows so that
T[i≠i0,j0] = 0.
"""
function pivot!(T::AbstractMatrix, i0, j0)
    T[i0,j0] == 0 && error("Can't pivot at a zero element")
    @views T[i0,:] ./= T[i0,j0]
    @inbounds for i ∈ axes(T, 1)
        i != i0 && (@views T[i,:] .-= T[i0,:] .* T[i,j0])
    end
    nothing
end
