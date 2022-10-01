module MLOptm

include("LinearMethods.jl")
include("GradientMethods.jl")

export Golden
export BiSection
export Newton
export Secant

export SteepestDescent
export NewtonND

export Minimize!

end
