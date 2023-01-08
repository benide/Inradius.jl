# requires DirectQhull master branch as of Jan 07, 2023 to get rid of memory leak
# requires julia >= 1.3 so that Qhull_jll can be loaded

# NOTE: we could also do
#   using DirectQhull
#   hull(p) = ConvexHull(p).equations
# but the code below factors out only the necessary part

import DirectQhull: qh_new_qhull, qh_get_simplex_facet_arrays, qhullptr

"""
    A = hull(pts::Matrix)

pts has size (dim, num_pts) with dim < 5.
A will have size (dim+1, number of facets). Each column is a facet, with the first `dim` entries representing the normal vector and the last entry representing the offset.
"""
function hull(points::Matrix{Cdouble})
    @assert 1 < size(points,1) < 5 "points must live in 2d, 3d, or 4d"
    qhullptr() do qh_ptr
        qh_new_qhull(qh_ptr, points, "Qt i")
        qh_get_simplex_facet_arrays(qh_ptr, Val(size(points,1)))[3]
    end
end
hull(points) = hull(Matrix(points))
