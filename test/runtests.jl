using ColumnPartitions, Test, Statistics

# test internals
using ColumnPartitions: AtomSchema, ArraySchema, TupleSchema, schema_and_eltype, flat_get,
    flat_set!, promote_schema, eltype_schema

@testset "schema building blocks" begin
    @testset "AtomSchema" begin
        @test schema_and_eltype(1) == (AtomSchema(), Int)
        @test length(AtomSchema()) == 1
        @test promote_schema(AtomSchema(), AtomSchema()) == AtomSchema()
    end
    @testset "ArraySchema" begin
        sch = ArraySchema{(3,4)}()
        @test schema_and_eltype(ones(Float64, 3, 4)) == (sch, Float64)
        @test length(sch) == 12
        @test promote_schema(sch, sch) == sch
        @test_throws ArgumentError promote_schema(sch, ArraySchema{(5, 6)}())
    end
    @testset "TupleSchema" begin
        sch = TupleSchema((AtomSchema(), ArraySchema{(7,)}()))
        schN = TupleSchema((a = AtomSchema(), b = ArraySchema{(7,)}()))
        @test schema_and_eltype((1, ones(Float64, 7))) == (sch, Float64)
        @test schema_and_eltype((a = 1, b = ones(Float64, 7))) == (schN, Float64)
        @test length(sch) == length(schN) == 8
        @test promote_schema(sch, sch) == sch
        @test promote_schema(schN, schN) == schN
        AE = ArgumentError("incompatible schemas")
        @test_throws AE promote_schema(sch, TupleSchema((AtomSchema(), )))
        @test_throws AE promote_schema(sch, schN)
        @test_throws AE promote_schema(schN, TupleSchema((a = AtomSchema(), )))
    end
end

@testset "partitions" begin
    v = [(a = Float64(i), b = [2*i, 3*i]) for i in 1:10]
    p = column_partition(v)
    @test collect(p) == v
    @test map_elementwise(mean, p)[] == (m = mean(1:10); (a = m, b = [2*m, 3*m]))
    qs = map_elementwise(c -> NamedTuple{(:q25, :q50, :q75)}(quantile(c, [0.25, 0.5, 0.75])), p)
    @test map_elementwise(first, qs)[] == (q = quantile(1:10, 0.25); (a = q, b = [2*q, 3*q]))
end
