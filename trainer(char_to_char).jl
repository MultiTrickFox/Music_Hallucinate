using Glob: glob
using Random: shuffle
using Distributed: procs, addprocs
using Distributed: @everywhere


const learning_rate = .01
const hm_epochs     = 50
const batch_size    = 2

const in_size       = 52
const layers        = [52]
const out_size      = 52


if length(procs()) <= 2
    hm_procs = batch_size > Sys.CPU_THREADS ?
        Sys.CPU_THREADS-1 : batch_size-1
    addprocs(hm_procs) ;println("$(length(procs())) cores running.")
end
@everywhere include("gru_api(char_to_char).jl")
@everywhere include("utils.jl")


create_model_definitions(layers)


const model = Model(mk_layers(in_size, layers, out_size)...)

const data = import_data(glob("class*.txt")[1])


for i in 1:hm_epochs
    print("epoch: $i ")
    train!(model, shuffle(data), learning_rate, batch_size)
    end
end

save_model(model)


print("Hit enter to continue..")
chomp(readline())
