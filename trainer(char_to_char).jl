using Distributed: procs, addprocs, @everywhere
if length(procs()) <= 2
    addprocs(Sys.CPU_THREADS-2)
    println("$(length(procs())) cores running.")
end ; @everywhere include("gru_api(char_to_char).jl")
using Glob: glob


const learning_rate = .01
const hm_epochs     = 10

const in_size       = 52
const layers        = [94]
const out_size      = 52


create_model_definitions(layers)


const model = Model(mk_layers(in_size, layers, out_size)...)

# const data  = [import_data(file) for file in glob("*.txt")]
const data = import_data(glob("*.txt")[1])

for d in data
    println(length(d))
end


# const data = [[randn(1,in_size) for _ in 1:3] for __ in 1:2]

for i in 1:hm_epochs
    print("epoch: $i ")
    train!(model, data, learning_rate)
end
