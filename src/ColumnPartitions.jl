module ColumnPartitions

export column_partition, vector_partition, map_elementwise

using ArgCheck: @argcheck

####
#### schemas
####

abstract type AbstractSchema end

function promote_schema_and_eltype(a, b)
    sch_a, T_a = a
    sch_b, T_b = b
    (promote_schema(sch_a, sch_b), promote_type(T_a, T_b))
end

###
### AtomSchema
###

struct AtomSchema <: AbstractSchema end

function schema_and_eltype(x::T) where T
    AtomSchema(), T
end

Base.length(sch::AtomSchema) = 1

flat_get(column, offset, sch::AtomSchema) = column[offset + 1]

function flat_set!(column, offset, sch::AtomSchema, x)
    column[offset + 1] = x
end

promote_schema(::AtomSchema, ::AtomSchema) = AtomSchema()

eltype_schema(::AtomSchema, T) = T

###
### ArraySchema
###

struct ArraySchema{S} <: AbstractSchema end       # FIXME check S

Base.size(sch::ArraySchema{S}) where S = S

Base.length(sch::ArraySchema) = prod(size(sch))

function schema_and_eltype(A::AbstractArray)
    @argcheck !Base.has_offset_axes(A)
    ArraySchema{size(A)}(), eltype(A)
end

function flat_get(column, offset, sch::ArraySchema)
    reshape(view(column, offset .+ (1:length(sch))), size(sch))
end

function flat_set!(column, offset, sch::ArraySchema, A::AbstractArray)
    @argcheck size(A) == size(sch)
    column[offset .+ (1:length(sch))] = vec(A)
end

function promote_schema(a::ArraySchema, b::ArraySchema)
    @argcheck size(a) == size(b)
    a
end

eltype_schema(sch::ArraySchema, T) = Array{T, length(size(sch))}

###
### TupleSchema
###

struct TupleSchema{T <: Union{Tuple,NamedTuple}} <: AbstractSchema
    sub_schemas::T
end

function schema_and_eltype(tup::Union{Tuple,NamedTuple})
    sch_and_Ts = map(schema_and_eltype, tup)
    TupleSchema(map(first, sch_and_Ts)), mapreduce(last, promote_type, sch_and_Ts)
end

Base.length(sch::TupleSchema) = sum(length, sch.sub_schemas)

function flat_get(column, offset, sch::TupleSchema)
    map(sch.sub_schemas) do sub_schema
        ret = flat_get(column, offset, sub_schema)
        offset += length(sub_schema)
        ret
    end
end

function flat_set!(column, offset, sch::TupleSchema{S}, tup::T
                   ) where {S, T <: Union{Tuple, NamedTuple}}
    @argcheck fieldnames(S) == fieldnames(T)
    map(sch.sub_schemas, tup) do sub_schema, elt
        ret = flat_set!(column, offset, sub_schema, elt)
        offset += length(sub_schema)
        ret
    end
end

function promote_schema(a::TupleSchema{S}, b::TupleSchema{T}) where {S,T}
    @argcheck fieldnames(S) == fieldnames(T) "incompatible schemas"
    TupleSchema(map(promote_schema, a.sub_schemas, b.sub_schemas))
end

function eltype_schema(sch::TupleSchema{<:Tuple}, T)
    Tuple{map(s -> eltype_schema(s, T), sch.sub_schemas)...}
end

function eltype_schema(sch::TupleSchema{S}, T) where {N, S <: NamedTuple{N}}
    NamedTuple{N, Tuple{map(s -> eltype_schema(s, T), values(sch.sub_schemas))...}}
end

_unflatten(sch, v) = flat_get(v, 0, sch)

####
#### API
####

###
### Column partitioned matrix
###

struct ColumnPartition{T, S <: AbstractSchema, M <: AbstractMatrix} <: AbstractVector{T}
    sch::S
    matrix::M
end

function ColumnPartition(sch::S, matrix::M) where {M,S}
    ColumnPartition{eltype_schema(sch, eltype(M)),S,M}(sch, matrix)
end

schema(p::ColumnPartition) = p.sch

function column_partition(v::AbstractVector)
    @argcheck !Base.has_offset_axes(v)
    @argcheck !isempty(v)
    sch, T = mapreduce(schema_and_eltype, promote_schema_and_eltype, v)
    matrix = Matrix{T}(undef, length(sch), length(v))
    for (column, elt) in zip(eachcol(matrix), v)
        flat_set!(column, 0, sch, elt)
    end
    ColumnPartition(sch, matrix)
end

Base.size(p::ColumnPartition) = (size(p.matrix, 2), )

Base.getindex(p::ColumnPartition, i) = _unflatten(p.sch, @view p.matrix[:, i])

###
### Partitioned vector
###

struct VectorPartition{S <: AbstractSchema, V <: AbstractVector}
    sch::S
    vector::V
end

Base.getindex(p::VectorPartition) = _unflatten(p.sch, p.vector)

####
#### mapping
####

function schema_and_flatten(x)
    sch, T = schema_and_eltype(x)
    @show sch
    v = Vector{T}(undef, length(sch))
    flat_set!(v, 0, sch, x)
    sch, @show v
end

# FIXME replace by broadcast?
function map_elementwise(f, p_first::ColumnPartition, p_rest::ColumnPartition...)
    sch = p_first.sch
    @assert all(p -> schema(p) == sch, p_rest)
    VectorPartition(sch, map(f, eachrow(p_first.matrix), map(p -> eachrow(p.matrix), p_rest)...))
end

function map_elementwise(f, p_first::VectorPartition, p_rest::VectorPartition...)
    sch = p_first.sch
    @assert all(p -> schema(p) == sch, p_rest)
    VectorPartition(sch, map(f, p_first.vector, map(p -> p.vector, p_rest)...))
end

end # module
