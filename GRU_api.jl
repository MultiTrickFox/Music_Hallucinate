using Distributed: @everywhere, @distributed, addprocs, procs
if length(procs()) <= 2
    addprocs(Sys.CPU_THREADS-2)
end ; @everywhere include("GRU_dynamic_struct.jl")



train!(model, datas, lr) =
begin

    result = @distributed vcat for (x,y) in datas
        d = @diff - sum(y .* log.(softmax([model(t) for t in x][end])))
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



prop(model, x) = softmax([model(t) for t in x][end])
