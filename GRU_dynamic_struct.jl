using Knet: @diff, Param, value, grad, params
using Knet: sigm, tanh, softmax


# sigm(x) = 1.0 / (1.0 + exp(-x))



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
    sq = sqrt(2/(in_size+out_size))
    wai = Param(2*sq .* randn(in_size, out_size) .- sq)
    was = Param(2*sq .* randn(out_size, out_size) .- sq)
    ba  = Param(zeros(1, out_size))
    wri = Param(2*sq .* randn(in_size, out_size) .- sq)
    wrs = Param(2*sq .* randn(out_size, out_size) .- sq)
    br  = Param(zeros(1, out_size))
    wsi = Param(2*sq .* randn(in_size, out_size) .- sq)
    bs  = Param(zeros(1, out_size))
    state = zeros(1, out_size)
    layer = Layer(wai, was, ba, wri, wrs, br, wsi, bs, state)
layer
end

(layer::Layer)(in) =
begin
    focus  = sigm.(in * layer.wai + layer.state * layer.was + layer.ba)
    keep   = sigm.(in * layer.wri + layer.state * layer.wrs + layer.br)
    interm = tanh.(in * layer.wsi + layer.state .* focus + layer.bs)
    layer.state = keep .* interm + (1 .- keep) .* layer.state
end


mk_def(in_size, layers, out_size) =
begin
    hm_layers = length(layers)+1

    model_def = "struct Model\n"
    for i in 1:hm_layers
        model_def *= "l$i::Layer\n"
    end
    model_def *= "end"

    eval(Meta.parse(model_def))

    model_call = "(model::Model)(io) =\n"
    model_call *= "begin\n"
    for i in 1:hm_layers
        model_call *= "io = model.l$i(io)\n"
    end
    model_call *= "end"

    eval(Meta.parse(model_call))

end


mk_layers(in_size, layers, out_size) =
begin
    hm_layers = length(layers)+1

    hm_layers = length(layers) +1
    internal = Layer[]
    for i in 1:hm_layers
        if     i == 1         layer = Layer(in_size, layers[i])
        elseif i == hm_layers layer = Layer(layers[end], out_size)
        else                  layer = Layer(layers[i-1], layers[i])
        end
        push!(internal, layer)
    end
internal
end
