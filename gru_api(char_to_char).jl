using Distributed: @everywhere, @spawnat, @distributed
@everywhere include("gru_dynamic_struct.jl")


const window_size = 50



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
    # for mfield in fieldnames(Model)
    #     layer = getfield(model, mfield)
    #     layer.state = zeros(1, length(layer.bs))
    # end
[soften(model(t)) for t in x]
end


cross_entropy(out, label) =
begin
    - sum(label .* log.(out))
end



train!(model, data, lr, batch_size) =
begin
    results = []
    for i in 1:trunc(Int, length(data)/batch_size)
        res = @distributed (vcat) for sequence in data[(i-1)*batch_size+1:i*batch_size]
            d = @diff (begin
                    sequence_parts = batchify(sequence, window_size)
                    part_losses = []
                    for part in sequence_parts
                        x = sequence[1:end-1]
                        y = sequence[2:end]
                        o = prop(model, x)
                        l = [cross_entropy(o_t, y_t) for (o_t, y_t) in zip(o, y)]
                        push!(part_losses, sum(l))
                    end
                sum(part_losses) end)
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
        push!(results, res)
    end

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
