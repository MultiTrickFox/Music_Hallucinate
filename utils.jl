
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
        println("importing $file")
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
        println("obtained $(length(data)) samples.")
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
