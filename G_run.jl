include("GRU_api.jl")
include("GRU_neuroevo.jl")


@everywhere const in_size  = 42
@everywhere const layers   = [40, 30, 25]
@everywhere const out_size = 20


create_model_definitions(layers)


const learning_rate = .01
const hm_epochs     = 10


const model = Model(mk_layers(in_size, layers, out_size)...)

const data = [[[randn(1, in_size) for i in 1:rand(1:16)], softmax(randn(1, out_size))] for _ in 1:100]



for _ in 1:hm_epochs

    train!(model, data, learning_rate)

end



        #####     #####



const hm_population = 100
const hm_mostfit    = 10

const track_length  = 16
const size_per_time = 42

const hm_iterations = 20

const crossover_rate = 0.2
const update_rate    = 0.1

const class = 4



noises = [noise(track_length, size_per_time) for _ in 1:hm_population]



evolve(noises, iterations) =

begin

    for _ in 1:iterations

        noises = crossover(noises, crossover_rate)
        noises = [update(noise, model, update_rate, class) for noise in noises]
        noises = mostfit(noises, hm_mostfit, model, class)

        sc = scores(model, noises, class)

        println("Overall progress: ", sum(sc))

    end
noises
end


noises = evolve(noises, hm_iterations)
