# This is a drop-in replacement for the qhull.jl file. Up to the
# initial tetrahedron, this code is great. After that, it needs work.
# I'd like it to be a true implementation of Quick Hull eventually, at
# which point this will be moved to its own package. For the moment,
# qhull.jl is the better choice.


using LinearAlgebra
using StaticArrays

function extrema(v::AbstractMatrix{<:Real}, dim::Integer)
    @views (argmin(v[dim,:]), argmax(v[dim,:]))
end

function extrema(v::AbstractMatrix{<:Real}, n::AbstractVector{<:Real})
    w = [c ⋅ n for c ∈ eachcol(v)]
    (argmin(w), argmax(w))
end

function argmax_in_direction(v::AbstractMatrix{<:Real}, n::AbstractVector{<:Real})
    argmax(c ⋅ n for c ∈ eachcol(v))
end

function findmax_in_direction(v::AbstractMatrix{<:Real}, n::AbstractVector{<:Real})
    findmax(c ⋅ n for c ∈ eachcol(v))
end

normal(a,b,c) = (b - a) × (c - a) |> normalize
normal(x,f) = normal(x[:,f[1]], x[:,f[2]], x[:,f[3]])
dist_line_to_point(a,b,p) = norm((p - a) × (p - b)) / norm(b - a)
dist_plane_to_point(a,b,c,p) = (p - a) ⋅ normal(a,b,c)
dist_plane_to_point(x,i,j,k,l) = @views dist_plane_to_point(x[:,i], x[:,j], x[:,k], x[:,l])

function furthest_from_line(points, i, j)
   @views argmax(dist_line_to_point(points[:,i], points[:,j], points[:,k]) for k=1:size(points,2))
end

function furthest_from_plane(points, i, j, k)
    @views argmax(abs(dist_plane_to_point(points[:,i], points[:,j], points[:,k], points[:,h])) for h=1:size(points,2))
end

function hull_initial_line(x::AbstractMatrix{<:Real})
    choices = (extrema(x, 1), extrema(x, 2), extrema(x, 3))
    @views choices[argmax(norm(x[:,choices[i][1]] - x[:,choices[i][2]]) for i=1:3)]
end

function hull_initial_tetrahedron(x::AbstractMatrix{<:Real})
    # get the 4 points
    i,j = hull_initial_line(x)
    k = furthest_from_line(x,i,j)
    l = furthest_from_plane(x,i,j,k)

    # ensure we actually have a 3d hull
    @views dist_plane_to_point(x[:,i], x[:,j], x[:,k], x[:,l]) == 0 && error("Hull is not 3-dimensional")

    # get the orientations correct
    @views if dist_plane_to_point(x[:,i], x[:,j], x[:,k], x[:,l]) > 0
        rotate_face_vertices(i,k,j),
        rotate_face_vertices(i,j,l),
        rotate_face_vertices(i,l,k),
        rotate_face_vertices(j,k,l)
    else
        rotate_face_vertices(i,j,k),
        rotate_face_vertices(i,k,l),
        rotate_face_vertices(i,l,j),
        rotate_face_vertices(j,l,k)
    end
end

rotate_face_vertices(a::Int, b::Int, c::Int) = rotate_face_vertices((a,b,c))
rotate_face_vertices(f::NTuple{3,Int}) = rotate_face_vertices(f,Val(argmin(f)))
rotate_face_vertices(f::NTuple{3,Int}, ::Val{1}) = f
rotate_face_vertices(f::NTuple{3,Int}, ::Val{2}) = (f[2], f[3], f[1])
rotate_face_vertices(f::NTuple{3,Int}, ::Val{3}) = (f[3], f[1], f[2])

const Face = NamedTuple{(:v,:n,:b), Tuple{NTuple{3,Int}, SVector{3,Float64}, Float64}}

# note that forcing this to use an SMatrix is actually faster when
# starting from a Julia matrix. It would be worth testing out
# StrideArrays.jl as well
hull(x::AbstractMatrix{<:Real}) = size(x,1) == 3 ? hull(SMatrix{size(x)...}(x)) : error("this is only implemented for 3d")

function hull(x::SMatrix{3,N,<:Real}) where N
    faces = Face[]
    for f in hull_initial_tetrahedron(x)
        n = SVector{3}(normal(x,f))
        b = n ⋅ x[:,f[1]]
        push!(faces, (v=f,n=n,b=b))
    end
    idxs = setdiff(1:size(x,2), union([f.v for f in faces]...))
    while !isempty(idxs)
        i = pop!(idxs)
        vis = [f.n ⋅ x[:,i] > f.b for f in faces]
        newfaces = Face[]
        for f in faces[vis], g in faces[.!vis]
            if length(f.v ∩ g.v) == 2
                v = replace(f.v, first(setdiff(f.v, g.v)) => i) |> rotate_face_vertices
                n = SVector{3}(normal(x,v))
                b = n ⋅ x[:,v[1]]
                push!(newfaces, (v=v,n=n,b=b))
            end
        end
        deleteat!(faces, vis)
        union!(faces, newfaces)
    end
    hcat([vcat(f.n, -f.b) for f in faces]...)
end


# UNUSED TEST STUFF BELOW

# function hspace_from_vertices(a,b,c)
#     n = normal(a,b,c)
#     return n, (a ⋅ n)
# end
# hspace_from_vertices(x,i,j,k) = hspace_from_vertices(view(x,:,i), view(x,:,j), view(x,:,k))
# hspace_from_vertices(x,f) = hspace_from_vertices(x,f...)

# function hspaces_from_faces(x, faces)
#     A = @MMatrix zeros(3,length(faces))
#     b = @MVector zeros(length(faces))
#     @inbounds for (i,f) in enumerate(faces)
#         @views A[:,i] .= normal(x[:,f[1]], x[:,f[2]], x[:,f[3]])
#         b[i] = view(x,:,f[1]) ⋅ view(A,:,i)
#     end
#     return A, b
# end

# function hspaces_from_faces2(x::SMatrix{3,N,<:Real}, faces::NTuple) where N
#     A = @MMatrix zeros(3,length(faces))
#     @inbounds for (i,f) in enumerate(faces)
#         @views A[:,i] .= normal(x[:,f[1]], x[:,f[2]], x[:,f[3]])
#     end
#     b = SVector{length(faces),eltype(A)}(x[:,f[1]] ⋅ A[:,i] for (i,f) in enumerate(faces))
#     return A, b
# end

# function hspaces_from_faces3(x, faces)
#     A = @MMatrix zeros(length(faces),3)
#     b = @MVector zeros(length(faces))
#     @inbounds for (i,f) in enumerate(faces)
#         @views A[i,:] .= normal(x[:,f[1]], x[:,f[2]], x[:,f[3]])
#         b[i] = view(x,:,f[1]) ⋅ view(A,i,:)
#     end
#     return A, b
# end
