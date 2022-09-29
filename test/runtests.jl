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

end

