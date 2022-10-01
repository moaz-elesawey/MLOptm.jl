using LinearAlgebra;


abstract type GradientMethod end


mutable struct NewtonND <: GradientMethod
    ∇f
    H
end
function Minimize!(G::NewtonND, x0, ϵ=1e-5; maxiters=10)
    ∇f, H = G.∇f, G.H;

    xk = x0 - inv(H(x0...)) * ∇f(x0...);

    for i in 1:maxiters
        xk = x0 - inv(H(x0...)) * ∇f(x0...);
        err = norm(xk-x0) / max(1, norm(x0));
        if err < ϵ break end
        x0 = xk;
    end

    return xk;
end


mutable struct SteepestDescent <: GradientMethod
    f
    ∇f
end
function Minimize!(G::SteepestDescent, x0, ϵ=1e-7; maxiters=10)
    f, ∇f = G.f, G.∇f;

    Φ(α) = f((x0 - α*∇f(x0...))...);
    α = Minimize!(Golden(Φ, 0, 1.0));
    xk = x0 - α*∇f(x0...);

    for i in 1:maxiters
        Φ(α) = f((x0 - α*∇f(x0...))...);
        α = Minimize!(Golden(Φ, 0.0, 1.0));
        xk = x0 - α*∇f(x0...);
        err = norm(xk-x0) / max(1, norm(x0));
        if err < ϵ break end
        x0 = xk;
    end

    return xk;
end


