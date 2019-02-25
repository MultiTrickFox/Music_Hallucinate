using Knet: @diff, Param, value, grad


sigm(x) = 1.0 / (1.0 + exp(-x))


in_size  = 10
layers   = [8, 6, 5]
out_size = 3



mutable struct Layer
    wai::Param
    was::Param
    ba::Param
    wri::Param
    wrs::Param
    br::Param
    wsi::Param
    bs::Param
    state
end

Layer(in_size, out_size) =
begin
    sq = sqrt(in_size)
    wai = Param(randn(in_size, out_size)/sq)
    was = Param(randn(out_size, out_size)/sq)
    ba  = Param(zeros(1, out_size))
    wri = Param(randn(in_size, out_size)/sq)
    wrs = Param(randn(out_size, out_size)/sq)
    br  = Param(zeros(1, out_size))
    wsi = Param(randn(in_size, out_size)/sq)
    bs  = Param(zeros(1, out_size))
    state = zeros(1, out_size)
    layer = Layer(wai, was, ba, wri, wrs, br, wsi, bs, state)
layer
end

(layer::Layer)(in) =
begin
    attention = sigm.(in * layer.wai + layer.state * layer.was + layer.ba)
    remember  = sigm.(in * layer.wri + layer.state * layer.wrs + layer.br)
    short_mem = tanh.(in * layer.wsi + attention .* layer.state + layer.bs)
    layer.state = remember .* short_mem + (1 .- remember) .* layer.state
end



make(in_size, layers, out_size) =
begin
    model = []
    hm_layers = length(layers) +1
    for i in 1:hm_layers
        if     i == 1         layer = Layer(in_size, layers[i])
        elseif i == hm_layers layer = Layer(layers[end], out_size)
        else                  layer = Layer(layers[i-1], layers[i])
        end
        push!(model, layer)
    end
model
end

prop(model, in) =
begin
    for layer in model
        in = layer(in)
    end
in
end
