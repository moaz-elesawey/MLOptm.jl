using LinearAlgebra;


abstract type GradientMethod end


mutable struct GradientDescent <: GradientMethod
    ∇f
    α
end
function Minimize!(G::GradientDescent, x0, ϵ=1e-7; maxiters=100)
    ∇f, α = G.∇f, G.α;

    xk = x0 - α*∇f(x0...);
    for k in 1:maxiters
        xk = x0 - α*∇f(x0...);
        err = norm(xk - x0) / max(1, norm(x0));
        if err < ϵ break end
        x0 = xk;
    end

    return xk;
end


mutable struct ConjugateDescent <: GradientMethod
    f
    ∇f
end
function Minimize!(G::ConjugateDescent, x0, ϵ=1e-7; maxiters=100)
    f, ∇f = G.f, G.∇f;

    g0 = ∇f(x0...); gk = ∇f(x0...);
    β = max(0, g0⋅(gk - g0) / (g0⋅g0));
    Φ(α) =f((x0 - α*gk)...);
    α = Minimize!(Golden(Φ, 0, 1));
    xk = x0 - α*gk;

    for k in 1:maxiters
        g0 = ∇f(x0...); gk = ∇f(x0...);
        β = max(0, g0⋅(gk - g0) / (g0⋅g0));
        Φ(α) =f((x0 - α*gk)...);
        α = Minimize!(Golden(Φ, 0, 1));
        xk = x0 - α*gk;
        err = norm(xk - x0) / max(1, norm(x0));
        if err < ϵ break end
        x0 = xk;
    end

    return xk;
end


mutable struct SteepestDescent <: GradientMethod
    f
    ∇f
end
function Minimize!(G::SteepestDescent, x0, ϵ=1e-7; maxiters=100)
    f, ∇f = G.f, G.∇f;

    Φ(α) = f((x0 - α*∇f(x0...))...);
    α = Minimize!(Golden(Φ, 0, 1.0));
    xk = x0 - α*∇f(x0...);

    for k in 1:maxiters
        Φ(α) = f((x0 - α*∇f(x0...))...);
        α = Minimize!(Golden(Φ, 0.0, 1.0));
        xk = x0 - α*∇f(x0...);
        err = norm(xk-x0) / max(1, norm(x0));
        if err < ϵ break end
        x0 = xk;
    end

    return xk;
end


mutable struct NewtonND <: GradientMethod
    ∇f
    H
end
function Minimize!(G::NewtonND, x0, ϵ=1e-5; maxiters=100)
    ∇f, H = G.∇f, G.H;

    xk = x0 - inv(H(x0...)) * ∇f(x0...);

    for k in 1:maxiters
        xk = x0 - inv(H(x0...)) * ∇f(x0...);
        err = norm(xk-x0) / max(1, norm(x0));
        if err < ϵ break end
        x0 = xk;
    end

    return xk;
end


########## Momentum Methods ##########

mutable struct Momentum <: GradientMethod
    ∇f
    α
    β
end
function Minimize!(G::Momentum, x0, ϵ=1e-7; maxiters=100)
    ∇f, α, β = G.∇f, G.α, G.β;

    v0 = zeros(length(x0));
    vk = β*v0 - α*∇f(x0...);
    xk = x0 + vk

    for k in 1:maxiters
        vk = β*v0 - α*∇f(x0...);
        xk = x0 + vk;
        err = norm(xk - x0) / max(1, norm(x0));
        v0 = vk; x0 = xk;
    end

    return xk;
end


mutable struct NestrovMomentum <: GradientMethod
    ∇f
    α
    β
end
function Minimize!(G::NestrovMomentum, x0, ϵ=1e-7; maxiters=100)
    ∇f, α, β = G.∇f, G.α, G.β;

    v0 = zeros(length(x0));
    vk = β*v0 - α*∇f(x0...);
    xk = x0 + vk

    for k in 1:maxiters
        vk = β*v0 - α*∇f((x0 + β*v0)...);
        xk = x0 + vk;
        err = norm(xk - x0) / max(1, norm(x0));
        v0 = vk; x0 = xk;
    end

    return xk;
end


mutable struct AdaGrad <: GradientMethod
    ∇f
    α
    ϵ
end
function Minimize!(G::AdaGrad, x0, ϵ=1e-7; maxiters=100)
    ∇f, α = G.∇f, G.α;

    sk = zeros(length(x0));
    xk = x0 - α*∇f(x0...) ./ (G.ϵ .+ sqrt.(sk));

    for k in 1:maxiters
        sk += ∇f(x0...).^2
        xk = x0 - α*∇f(x0...) ./ (G.ϵ .+ sqrt.(sk));
        err = norm(xk - x0) / max(1, norm(x0));
        if err < ϵ break end
        x0 = xk;
    end

    return xk;
end


mutable struct RMSProp <: GradientMethod
    ∇f
    α
    ϵ
    γ
end
function Minimize!(G::RMSProp, x0, ϵ=1e-7; maxiters=100)
    ∇f, α, γ = G.∇f, G.α, G.γ;

    s = zeros(length(x0));
    xk = x0 - (α*∇f(x0...) ./ (G.ϵ .+ sqrt.(s)));

    for k in 1:maxiters
        s[:] = γ*s + (1-γ)*(∇f(x0...).*∇f(x0...));
        xk = x0 - (α*∇f(x0...) ./ (G.ϵ .+ sqrt.(s)));
        err = norm(xk - x0) / max(1, norm(x0));
        if err < ϵ break end
        x0 = xk;
    end

    return xk;
end

