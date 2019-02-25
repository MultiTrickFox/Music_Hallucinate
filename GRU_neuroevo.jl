include("GRU_api.jl")



noise(track_length, size_per_time) =
begin
    [randn(1,size_per_time) for i in 1:track_length]
end


scores(model, noises, wrt) =
begin
    hm_classes = length(noises)
    scores = [0.0 for _ in 1:hm_classes]
    label = [i == wrt ? 0.0 : 1.0 for i in 1:hm_classes]
    for (i,noise) in enumerate(noises)
        out = prop(model, noise)
        # scores[i] = argmax(reshape(result, hm_classes))
        scores[i] = - sum(label .* log.(out))
    end
scores
end


update(noise, model, lr, class) =
begin
    sequence = [Param(t) for t in noise]
    result =
        @diff begin
            out = prop(model, sequence) # label = argmax(reshape(out, length(out)))
            label = [i == class ? 1.0 : 0.0 for i in 1:length(out)]
            loss  = cross_entropy(out, label)
    end
    sequence = [value(t - grad(result,t)*lr) for t in sequence]
sequence
end


mutate(noise; prob=.1) =
begin
    new_noise = []
    for (it,t) in enumerate(noise)
        timestep = []
        for (iv,v) in enumerate(t)
            if rand() <= prob
                v += randn()
            end
            push!(timestep, v)
        end
        timestep = reshape(timestep, 1, length(timestep))
        push!(new_noise, timestep)
    end
new_noise
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


crossover(fits, rate) =
begin
    offsprings = []
    for fit1 in fits
        for fit2 in fits
            if fit1 != fit2
                offspring = []
                for (t1,t2) in zip(fit1, fit2) # TODO : improve.
                    if rand() <= rate
                        push!(offspring, t1)
                    else
                        push!(offspring, t2)
                    end
                end
                push!(offsprings, offspring)
            end
        end
    end
vcat(fits, offsprings)
end
