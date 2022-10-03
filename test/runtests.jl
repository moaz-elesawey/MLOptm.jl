using MLOptm
using Test

@testset "MLOptm.jl" begin

    f   = (x) -> x^2;
    ∇f  = (x) -> 2x;
    ∇²f = (x) -> 2;

    @test isapprox(Minimize!(Golden(f, -10, 10))    , 0.0; atol=1e-4, rtol=0);
    @test isapprox(Minimize!(BiSection(∇f, -10, 10)), 0.0; atol=1e-4, rtol=0);
    @test isapprox(Minimize!(Newton(∇f, ∇²f, -0.1)) , 0.0; atol=1e-4, rtol=0);
    @test isapprox(Minimize!(Secant(∇f, -1.0, -0.9)), 0.0; atol=1e-4, rtol=0);

    f(x, y)  = x^2 + y^2;
    ∇f(x, y) = [2x ; 2y];
    H(x, y)  = [2 0; 0 2];
    x0 = [10; 10]

    @test isapprox(Minimize!(GradientDescent(∇f, 0.1), x0)      , [1.0, 1.0]; atol=1e-1, rtol=1);
    @test isapprox(Minimize!(ConjugateDescent(f, ∇f), x0)       , [0.0, 0.0]; atol=1e-4, rtol=0);
    @test isapprox(Minimize!(SteepestDescent(f, ∇f), x0)        , [0.0, 0.0]; atol=1e-4, rtol=0);
    @test isapprox(Minimize!(NewtonND(∇f, H), x0)               , [0.0, 0.0]; atol=1e-4, rtol=0);
    @test isapprox(Minimize!(Momentum(∇f, 0.1, 0.1), x0)        , [1.0, 1.0]; atol=1e-1, rtol=1);
    @test isapprox(Minimize!(NestrovMomentum(∇f, 0.1, 0.1), x0) , [1.0, 1.0]; atol=1e-1, rtol=1);

end

