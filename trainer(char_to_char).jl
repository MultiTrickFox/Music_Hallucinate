using Distributed: procs, addprocs
using Distributed: @everywhere
using Glob: glob
if length(procs()) <= 2
    addprocs(Sys.CPU_THREADS-1)
    println("$(length(procs())) cores running.")
end ; @everywhere include("gru_api(char_to_char).jl")


const learning_rate = .01
const hm_epochs     = 50

const in_size       = 52
const layers        = [94]
const out_size      = 52


create_model_definitions(layers)


const model = Model(mk_layers(in_size, layers, out_size)...)

const data = import_data(glob("class*.txt")[1])


for i in 1:hm_epochs
    print("epoch: $i ")
    train!(model, data, learning_rate)
end

save_model(model)


print("Hit enter to continue..")
chomp(readline())
