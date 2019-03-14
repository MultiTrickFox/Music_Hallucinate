using Random: shuffle
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



train!(model, datas, lr) =
begin
    results = @distributed (vcat) for sequence in shuffle(datas)
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

batchify(resource, batch_size) =
begin
    hm_batches = trunc(Int, length(resource)/batch_size)
    hm_leftover = length(resource)%batch_size
    batches = []
    if batch_size > length(resource)
        push!(batches, resource)
    else
        for i in 1:hm_batches
            push!(batches, resource[(i-1)*batch_size+1:i*batch_size])
        end
        if hm_leftover != 0
            push!(batches, resource[(end-hm_leftover)-1:end])
        end
    end
batches
end

import_data(file) =
    open(file) do f
        data = []
        sample = []
        for line in eachline(f)
            if line == ";"
                if length(sample) != 0
                    push!(data, sample)
                    sample = []
                end
            else
                nrs = split(line, " ")
                arr = []
                for nr in nrs
                    try
                        push!(arr, parse(Float32, nr))
                    catch end
                end
                # arr = [parse(Float32, nr) for nr in nrs]
                push!(sample, reshape(arr, 1, length(arr)))
            end
        end
        if length(sample) != 0
            push!(data, sample)
        end
        println("from $file imported $(length(data)) samples.")
    data
    end

save_model(model) =
begin
    open("model.txt", "w+") do file
        for mfield in fieldnames(Model)
            layer = getfield(model, mfield)
            for lfield in fieldnames(Layer)
                if lfield != :state
                    items = value(getfield(layer, lfield))
                    for i in items
                        write(file, string(i) * " ")
                    end
                    write(file, "\n")
                end
            end
        end
    end
end

load_model() =
begin

end
