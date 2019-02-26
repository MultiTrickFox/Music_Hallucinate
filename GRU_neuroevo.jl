include("GRU_api.jl")
# TODO : use pmap



noise(track_length, size_per_time) =
begin
    [randn(1,size_per_time) for i in 1:track_length]
end


scores(model, noises, wrt) =
begin
    hm_classes = length(noises)
    scores = [0.0 for _ in 1:hm_classes]
    label = [i == wrt ? 0.0 : 1.0 for i in 1:hm_classes]
    @distributed (vcat) for noise in noises
        out = prop(model, noise)
    [cross_entropy(out, label)]
    end
# reshape(scores, length())
end


update(noise, model, lr, class) =
begin
    sequence = [Param(t) for t in noise]
    result =
        @diff begin
            out = prop(model, sequence) # label = argmax(reshape(out, length(out)))
            label = [i == class ? 1.0 : 0.0 for i in 1:length(out)]
            cross_entropy(out, label)
    end
    sequence = [value(t - grad(result,t)*lr) for t in sequence]
sequence
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
    sc = scores(model, noises, class)
    fits = []
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
    offsprings = []
    for fit1 in fits
        @distributed (vcat) for fit2 in fits
            if fit1 != fit2
                offspring = []
                for (t1,t2) in zip(fit1, fit2)
                    timestep = []
                    for (e1,e2) in zip(t1,t2)
                        if rand() <= prob
                            push!(timestep, e2)
                        else
                            push!(timestep, e1)
                        end
                    push!(offspring, timestep)
                    # [timestep]
                    end
                end
                # push!(offsprings, offspring)
            [offspring]
            end
        end
    end
offsprings
end
