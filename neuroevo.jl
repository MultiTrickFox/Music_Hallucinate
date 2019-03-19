using Distributed: @sync, @async, remotecall_fetch, @everywhere
include("gru_api(seq_to_vec).jl")


@everywhere noise(track_length, size_per_time) =
begin
    noise = []
    total_duration = 0.0
    while total_duration < track_length
        vec = randn(1,size_per_time)
        durations = vec[Int8(length(vec)*2/4):Int8(length(vec)*3/4)-1]
        durations = [total_duration+e<track_length ? e : track_length-total_duration for e in durations]
        total_duration += maximum(durations)
        for i in 1:length(vec)*1/4
            vec[Int8(i)] = durations[Int8(i)]
        end
        push!(noise, vec)
    end
noise
end


@everywhere scores(model, noises, wrt) =
begin
    label = [i == wrt ? 1 : 0 for i in 1:hm_classes]
    sc = @distributed (vcat) for noise in noises
        cross_entropy(prop(model, noise), label)
    end
sc
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
            noise = [Param(t) for t in noise]
            label = [i == class ? 1 : 0 for i in 1:hm_classes]
            result =
                @diff begin
                    out = prop(model, noise)
                    cross_entropy(out, label)
                end
            noise = [value(t - grad(result,t)*lr) for t in noise]
            noise = reshape(noise, 1, length(noise))
        [noise]
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
            timestep = []
            for v in t[1:Int8(length(t)*3/4)]
                value = rand() <= prob ? bound(v+(randn()*rate)) : v
                push!(timestep, value)
            end
            for v in t[end-(Int8(length(t)*1/4)-1):end]
                push!(timestep, v)
            end
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
                        timestep = reshape(timestep, 1, length(timestep))
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


@everywhere bound(val; min_val=-1, max_val=1) =
    if     val > max_val max_val
    elseif val < min_val min_val
    else   val
    end



evolve(population, hm_iterations) =
begin
    for i in 1:hm_iterations

        fits = mostfit(population, hm_mostfit, model, class)

        # results = ["str", 1]
        # @sync begin
        #     @async results[1] = remotecall_fetch(evolution, 1, [population, fits])
        #     @async results[2] = remotecall_fetch(update, 2, [fits, model, update_rate])
        # end

        # population = vcat(results[1], results[2], fits)

        population = vcat(evolution([population, fits]), fits)

        population = mostfit(population, hm_population, model, class)

        print("/")
    end
    print("\n")

    loss = sum(scores(model, mostfit(population, hm_mostfit, model, class), class))
    # @show loss

[population, loss]
end
