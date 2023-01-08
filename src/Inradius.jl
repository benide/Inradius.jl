module Inradius

using LinearAlgebra

include("qhull.jl") # replace with "hull.jl" for a pure julia convex hull solver
include("chebyshev.jl")
include("solution-tools.jl")
include("normalize_path.jl")
include("experiments.jl")

export L, Lc, R, dLR, dLR!, dLcR, dLcR!, proj!, local_minimum!, perturb_path!, normalize_path!, normalize_path_closed!, experiment, experimentc

end
