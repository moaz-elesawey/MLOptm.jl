module MLOptm

include("LinearMethods.jl")
include("GradientMethods.jl")

export Golden
export BiSection
export Newton
export Secant

export GradientDescent
export ConjugateDescent
export SteepestDescent
export NewtonND
export Momentum
export NestrovMomentum
export AdaGrad
export RMSProp

export Minimize!

end
