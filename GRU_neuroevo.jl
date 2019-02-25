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
            out = prop(model, sequence)
            # label = argmax(reshape(out, length(out)))
            label = [i == class ? 1.0 : 0.0 for i in 1:length(out)]
            loss  = - sum(label .* log.(out))
    end
    for t in sequence
        g = grad(result, t)
        t += g*lr
    end
[value(t) for t in sequence]
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
                for (f1,f2) in zip(fit1, fit2)
                    if rand() <= rate
                        push!(offspring, f2)
                    else
                        push!(offspring, f1)
                    end
                end
                push!(offsprings, offspring)
            end
        end
    end
vcat(fits, offsprings)
end
