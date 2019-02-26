include("GRU_api.jl")
include("GRU_neuroevo.jl")


@everywhere const in_size  = 42
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

const hm_population  = 50
const hm_mostfit     = 5

const track_length   = 8
const size_per_time  = in_size
const hm_classes     = out_size

const hm_generations = 10
const hm_total_loop  = 100

const crossover_prob = .2
const mutate_prob    = .5
const mutate_rate    = .2
const update_rate    = .1



noises = [noise(track_length, size_per_time) for _ in 1:hm_population]



evolve(noises, iterations) =
begin
    for i in 1:iterations

        fits = mostfit(noises, hm_mostfit, model, class)
        # trained_fits = [update(noise, model, update_rate, class) for noise in fits]
        produced = crossover(fits, crossover_prob)
        noises = vcat(noises, produced)
        noises = mutate(noises, mutate_rate, mutate_prob)
        noises = vcat(noises, fits)
        # noises = vcat(noises, other_noises) # for noise in other_noises push!(noises, noise) end
        noises = mostfit(noises, hm_population, model, class)

        print("/")
    end
    print("\n")

    loss = sum(scores(model, mostfit(noises, hm_mostfit, model, class), class)/hm_mostfit)
    # loss = scores(model, mostfit(noises, 1, model, class), class)
    @show loss
[noises, loss]
end


loop(noises, hm_loop, hm_generations) =
begin
    loss_init = sum(scores(model, mostfit(noises, hm_mostfit, model, class), class))

    for i in 1:hm_loop
        print("loop: ",i," ")

        noises, loss = evolve(noises, hm_generations)

        println("Progress: ", (1-loss/loss_init)*100)
    end
noises
end



noises = loop(noises, hm_total_loop, hm_generations)
