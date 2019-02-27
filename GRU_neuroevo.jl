using Distributed: @sync, @async, remotecall_fetch, @everywhere
include("GRU_api.jl")


@everywhere noise(track_length, size_per_time) =
begin
    [randn(1,size_per_time) for i in 1:track_length]
end


@everywhere scores(model, noises, wrt) =
begin
    label = [i == wrt ? 1 : 0 for i in 1:hm_classes]
    @distributed (vcat) for noise in noises
        out = prop(model, noise)
    [cross_entropy(out, label)]
    end
end


@everywhere mostfit(noises, hm, model, class) =
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


@everywhere update((noises, model, lr)) =
begin
    @distributed (vcat) for noise in noises
            sequence = [Param(t) for t in noise]
            label = [i == class ? 1 : 0 for i in 1:hm_classes]
            result =
                @diff begin
                    out = prop(model, sequence)
                    cross_entropy(out, label)
                end
            sequence = [value(t - grad(result,t)*lr) for t in sequence]
        [sequence]
    end
end


@everywhere evolution((population, fits)) =
begin
    offsprings = crossover(fits, crossover_prob)
    population = vcat(population, offsprings)
    population = mutate(population, mutate_rate, mutate_prob)
population
end


@everywhere mutate(noises, rate, prob) =
begin
    new_noises = @distributed (vcat) for noise in noises
        new_noise = []
        for t in noise
            timestep = [rand() <= prob ? check_bounds(v+(randn()*rate)) : v for v in t]
            push!(new_noise, reshape(timestep, 1, length(timestep)))
        end
        [new_noise]
    end
new_noises
end


@everywhere crossover(fits, prob) =
begin
    len = Int8(length(fits[1][1]) * 1/4)
    hm  = Int8(hm_offspring/2)
    all_offsprings = []
    for fit1 in fits
        results = @distributed (vcat) for fit2 in fits
            if fit1 != fit2
                offsprings = []
                for _ in 1:hm
                    offspring = []
                    for (t1,t2) in zip(fit1, fit2)
                        timestep = []
                        for i in 1:len
                            if rand() <= prob
                                push!(timestep, t2[i])
                            else
                                push!(timestep, t1[i])
                            end
                        end
                        for i in len+1:length(fits[1][1])
                            push!(timestep, t1[i])
                        end
                        push!(offspring, timestep)
                    end
                    push!(offsprings, offspring)
                end
            offsprings
            end
        end
        all_offsprings = vcat(all_offsprings, results)
    end
    all_offsprings = [e for e in all_offsprings if e != nothing]
all_offsprings
end


@everywhere check_bounds(val; min_val=-1, max_val=1) =
    if     val > max_val max_val
    elseif val < min_val min_val
    else   val
    end



evolve(population, hm_iterations) =
begin
    for i in 1:hm_iterations

        fits = mostfit(population, hm_mostfit, model, class)


        arr = ["str", 1]
        @sync begin
            @async arr[1] = remotecall_fetch(evolution, 1, [population, fits])
            @async arr[2] = remotecall_fetch(update, 2, [fits, model, update_rate])
        end

        # population = vcat(evolution([population, fits]), fits)

        population = mostfit(vcat(arr[1], arr[2], fits), hm_population, model, class)

        print("/")
    end
    print("\n")

    loss = sum(scores(model, mostfit(population, hm_mostfit, model, class), class))
    @show loss

[population, loss]
end
