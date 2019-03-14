using Random: shuffle
using Distributed: @everywhere, @spawnat, @distributed
@everywhere include("gru_dynamic_struct.jl")



create_model_definitions(layer) =
begin
    eval(Meta.parse("@everywhere " * "struct Model\n" * *(["l$i::Layer\n" for i in 1:length(layers) +1]...) * "end"))
    eval(Meta.parse("@everywhere " * "(model::Model)(io) =\n" * "begin\n" * *(["io = model.l$i(io)\n" for i in 1:length(layers) +1]...) * "end"))
end


soften = arr -> (begin
    new_arr = []
    soft = softmax(arr[1:Int(length(arr)/4)])
    for i in 1:Int(length(arr)/4)
        push!(new_arr, soft[i])
    end
    for i in Int(length(arr)/4):Int(length(arr)*3/4)
        push!(new_arr, sigm(arr[i]))
    end
new_arr
end)

prop(model, x) =
begin
    for mfield in fieldnames(Model)
        layer = getfield(model, mfield)
        layer.state = zeros(1, length(layer.bs))
    end
    outs = []
    for t in x
        out_t = model(t)
        out_t = soften(out_t)
        push!(outs, out_t)
    end
outs
# [soften(model(t)) for t in x]
end


cross_entropy(out, label) =
begin
    - sum(label .* log.(out))
end



train!(model, datas, lr) =
begin
    results = @distributed (vcat) for sequence in shuffle(datas)
        d = @diff (begin
                x = sequence[1:end-1]
                y = sequence[2:end]
                o = prop(model, x)
                l = [cross_entropy(o_t, y_t) for (o_t, y_t) in zip(o, y)]
            sum(l) end)
        grads = []
        for mfield in fieldnames(Model)
            layer = getfield(model, mfield)
            for lfield in fieldnames(Layer)
                push!(grads, grad(d, getfield(layer, lfield)))
            end
        end
        @spawnat 1 print("/")
        grads, value(d)
    end
    print("\n")

    loss = 0.0
    for (g,l) in results
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



### Helper Utils ###

import_data(file) =
    open(file) do f
        data = []
        for line in eachline(f)
            nrs = split(line, " ")
            arr = [parse(Float32, nr) for nr in nrs[1:end-1]]
            push!(data, reshape(arr, 1, length(arr)))
        end
    data
    end
