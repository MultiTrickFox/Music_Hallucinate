include("GRU_api.jl")
include("GRU_neuroevo.jl")


@everywhere const in_size  = 52
@everywhere const layers   = [40, 30, 25]
@everywhere const out_size = 20


create_model_definitions(layers)

const model = Model(mk_layers(in_size, layers, out_size)...)


const learning_rate = .01
const hm_epochs     = 10


const data = [[[randn(1, in_size) for i in 1:rand(1:16)], softmax(randn(1, out_size))] for _ in 1:100]



for i in 1:hm_epochs
    print("epoch: $i ")
    train!(model, data, learning_rate)
end



        #####     #####



const class = 4

const hm_initial     = 5_000#10_000
const hm_population  = 500#5_000
const hm_mostfit     = 20#50

const track_length   = 8
const size_per_time  = in_size
const hm_classes     = out_size

const hm_generations = 100
const hm_total_loop  = 1_000

const crossover_prob = .2
const mutate_prob    = .2
const mutate_rate    = .2
const update_rate    = .01



population = [noise(track_length, size_per_time) for _ in 1:hm_initial]



evolve(population, iterations) =
begin
    for i in 1:iterations

        fits = mostfit(population, hm_mostfit, model, class)
        trained_fits = update(fits, model, update_rate, class)
        offsprings = crossover(fits, crossover_prob)
        population = vcat(population, offsprings)
        population = mutate(population, mutate_rate, mutate_prob)
        population = vcat(population, fits)
        population = vcat(population, trained_fits)
        population = vcat(fits, population)
        population = mostfit(population, hm_population, model, class)

        print("/")
    end
    print("\n")

    loss = sum(scores(model, mostfit(population, hm_mostfit, model, class), class))
    @show loss

[population, loss]
end


loop(population, hm_loop, hm_generations) =
begin
    loss_init = sum(scores(model, mostfit(population, hm_mostfit, model, class), class))
    @show loss_init

    for i in 1:hm_loop
        print("loop: ",i," ")
        population, loss = evolve(population, hm_generations)
        println("Progress: ", (1-loss/loss_init)*100)
    end
population
end



population = loop(population, hm_total_loop, hm_generations)
