using Printf;

abstract type LinearMethod end

mutable struct Golden <: LinearMethod
    f  :: Function
    a0 :: Float64
    b0 :: Float64
end

function Minimize!(M::Golden, ϵ::Float64=1e-5)
    ρ = (3-√5)/2;

    f, a0, b0 = M.f, M.a0, M.b0;

    N = Int(ceil(log(ϵ/2)/log(1-ρ)));

    for i in 1:N
        a1 = a0 + ρ*(b0 - a0);
        b1 = b0 - ρ*(b0 - a0);

        f(b1) > f(a1) ? b0 = b1 : a0 = a1;
    end

    (a0+b0)/2;
end

mutable struct BiSection <: LinearMethod
    ∇f :: Function
    a0 :: Float64
    b0 :: Float64
end

function Minimize!(M::BiSection, ϵ::Float64=1e-5)
    ∇f, a0, b0 = M.∇f, M.a0, M.b0;

    c = (a0+b0)/2;
    while abs(b0-a0) > ϵ
        c = (a0+b0)/2;
        ∇f(c) > 0 ? b0 = c : a0 = c;
    end

    return c;
end

mutable struct Newton <: LinearMethod
    ∇f  :: Function
    ∇²f :: Function
    x0  :: Float64
end

function Minimize!(M::Newton, ϵ::Float64=1e-5)
    ∇f, ∇²f, x0 = M.∇f, M.∇²f, M.x0;

    xk = x0 - ∇f(x0)/∇²f(x0);

    while abs(xk-x0) > ϵ
        xk = x0 - ∇f(x0)/∇²f(x0);
        x0 = xk;
    end

    return xk;
end

mutable struct Secant <: LinearMethod
    ∇f :: Function
    x0 :: Float64
    x1 :: Float64
end

function Minimize!(M::Secant, ϵ::Float64=1e-5)
    ∇f, x0, x1 = M.∇f, M.x0, M.x1;

    xk = (∇f(x1)*x0 - ∇f(x0)*x1) / (∇f(x1) - ∇f(x0));

    while abs(xk-x0) > ϵ
        xk = (∇f(x1)*x0 - ∇f(x0)*x1) / (∇f(x1) - ∇f(x0));
        x0 = x1; x1 = xk;
    end

    return xk;
end


f   = (x) -> x^4 - 14x^3 + 60x^2 - 70x;
∇f  = (x) -> 4x^3 - 14*3*x^2 + 120x - 70;
∇²f = (x) -> 12x^2 - 14*6*x + 120;

m = Minimize!(Golden(f, -10, 10));
@printf("Golden Minima    = %10.8f\n", m);

m = Minimize!(BiSection(∇f, -10, 10));
@printf("BiSection Minima = %10.8f\n", m);

m = Minimize!(Newton(∇f, ∇²f, -0.8));
@printf("Newton Minima    = %10.8f\n", m);

m = Minimize!(Secant(∇f, -1.0, 0.9));
@printf("Secant Minima    = %10.8f\n", m);

