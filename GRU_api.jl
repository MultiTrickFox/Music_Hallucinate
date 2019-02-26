using Distributed: @everywhere, @distributed, addprocs, procs
if length(procs()) <= 2
    addprocs(Sys.CPU_THREADS-2)
end ; @everywhere include("GRU_dynamic_struct.jl")



create_model_definitions(layer) =
begin
    eval(Meta.parse("@everywhere " * "struct Model\n" * *(["l$i::Layer\n" for i in 1:length(layers) +1]...) * "end"))
    eval(Meta.parse("@everywhere " * "(model::Model)(io) =\n" * "begin\n" * *(["io = model.l$i(io)\n" for i in 1:length(layers) +1]...) * "end"))
end


@everywhere prop(model, x) =
begin
    for mfield in fieldnames(Model)
        layer = getfield(model, mfield)
        # setfield!(layer, :state, zeros(1, length(getfield(layer, :bs))))
        layer.state = zeros(1, length(layer.bs))
    end
softmax([model(t) for t in x][end])
end


@everywhere cross_entropy(out, label) =
begin
    - sum(label .* log.(out))
end


train!(model, datas, lr) =
begin

    result = @distributed vcat for (x,y) in datas
        d = @diff cross_entropy(prop(model, x), y)
        grads = []
        for mfield in fieldnames(Model)
            layer = getfield(model, mfield)
            for lfield in fieldnames(Layer)
                push!(grads, grad(d, getfield(layer, lfield)))
            end
        end
        grads, value(d)
    end

    loss = 0.0
    for (g,l) in result
        loss += l
        i = 0
        for mfield in fieldnames(Model)
            layer = getfield(model, mfield)
            for lfield in fieldnames(Layer)
                i +=1
                setfield!(layer, lfield, Param(getfield(layer, lfield) - g[i] .* lr))
            end
        end

    end

    @show loss
end
