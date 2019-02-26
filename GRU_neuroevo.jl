include("GRU_api.jl")



noise(track_length, size_per_time) =
begin
    [randn(1,size_per_time) for i in 1:track_length]
end


scores(model, noises, wrt) =
begin
    label = [i == wrt ? 1 : 0 for i in 1:hm_classes]
    @distributed (vcat) for noise in noises
        out = prop(model, noise)
    [cross_entropy(out, label)]
    end
end


update(noises, model, lr, class) =
begin
    @distributed (vcat) for noise in noises
            sequence = [Param(t) for t in noise]
            label = [i == class ? 1 : 0 for i in 1:hm_classes]
            result =
                @diff begin
                    out = prop(model, sequence) # label = argmax(reshape(out, length(out)))
                    cross_entropy(out, label)
                end
            sequence = [value(t - grad(result,t)*lr) for t in sequence]
        [sequence]
    end
end


mutate(noises, rate, prob) =
begin
    new_noises = @distributed (vcat) for noise in noises
        new_noise = []
        for (it,t) in enumerate(noise)
            timestep = []
            for (iv,v) in enumerate(t)
                if rand() <= prob
                    v += randn() * rate
                end
                push!(timestep, v)
            end
            timestep = reshape(timestep, 1, length(timestep))
            push!(new_noise, timestep)
        end
        [new_noise]
    end
new_noises
end


mostfit(noises, hm, model, class) =
begin
    noises = deepcopy(noises)
    fits = []
    sc = scores(model, noises, class)
    for _ in 1:hm
        am = argmin(sc)
        push!(fits, noises[am])
        deleteat!(sc, am)
        deleteat!(noises, am)
    end
fits
end


crossover(fits, prob) =
begin
    len = Int8(length(fits[1][1]) * 1/4)
    offsprings = []
    for fit1 in fits
        @distributed (vcat) for fit2 in fits
            if fit1 != fit2
                offspring = []
                for (t1,t2) in zip(fit1, fit2)
                    timestep = []
                    for i in 1:len
                        if rand() <= prob
                            push!(timestep, t2[i])
                        else
                            push!(timestep, t1[i])
                        end
                    push!(offspring, timestep)
                    end
                end
            [offspring]
            end
        end
    end
offsprings
end
