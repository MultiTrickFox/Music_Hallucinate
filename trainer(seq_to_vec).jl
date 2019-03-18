using Glob: glob
using Random: shuffle
using Distributed: procs, addprocs
using Distributed: @everywhere

if length(procs()) < Sys.CPU_THREADS-2
    addprocs(Sys.CPU_THREADS-2) ;println("$(length(procs())) cores running.") ;end
@everywhere include("gru_api(seq_to_vec).jl") ; @everywhere include("utils.jl")


const learning_rate = .01
const hm_epochs     = 50

const in_size       = 52
const layers        = [52]
const out_size      = 52


create_model_definitions(layers)


const model = Model(mk_layers(in_size, layers, out_size)...)

const data = [[[randn(1, in_size) for i in 1:rand(1:16)], softmax(randn(1, out_size))] for _ in 1:100] # const data = import_data(glob("class*.txt")[1])


for i in 1:hm_epochs
    print("epoch: $i ")
    train!(model, shuffle(data), learning_rate)
    end
end

save_model(model)


print("Hit enter to continue..")
chomp(readline())
