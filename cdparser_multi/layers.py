#!/usr/bin/env python
# encoding: utf-8


from dynet import Saveable, parameter, transpose, dropout, rectify, GlorotInitializer
from dynet import ConstInitializer, layer_norm, affine_transform
from dynet import SaxeInitializer, concatenate, zeroes
import math


class Dense(Saveable):
    def __init__(self, indim, outdim, activation, model, ln=False):
        self.activation = activation
        self.ln = ln
        if activation == rectify:
            self.W = model.add_parameters((outdim, indim), init=GlorotInitializer(gain=math.sqrt(2.)))
        else:
            self.W = model.add_parameters((outdim, indim))
        self.b = model.add_parameters(outdim, init=ConstInitializer(0.))
        if ln:
            self.ln_s = model.add_parameters(outdim, ConstInitializer(1.))

    def __call__(self, x):
        if self.ln:
            return self.activation(layer_norm(parameter(self.W) * x, parameter(self.ln_s), parameter(self.b)))
        else:
            return self.activation(affine_transform([parameter(self.b), parameter(self.W), x]))

    def get_components(self):
        if self.ln:
            return [self.W, self.b, self.ln_s]
        else:
            return [self.W, self.b]

    def restore_components(self, components):
        if self.ln:
            self.W, self.b, self.ln_s = components
        else:
            self.W, self.b = components


class MultiLayerPerceptron(Saveable):
    def __init__(self, dims, activation, model, ln=False):
        self.layers = []
        self.dropout = 0.
        self.outdim = []
        for indim, outdim in zip(dims, dims[1:]):
            self.layers.append(Dense(indim, outdim, activation, model, ln))
            self.outdim.append(outdim)

    def __call__(self, x):
        for layer, dim in zip(self.layers, self.outdim):
            x = layer(x)
            if self.dropout > 0.:
                x = dropout(x, self.dropout)
        return x

    def set_dropout(self, droprate):
        self.dropout = droprate

    def get_components(self):
        return self.layers

    def restore_components(self, components):
        self.layers = components


class Bilinear(Saveable):
    def __init__(self, dim, model):
        self.U = model.add_parameters((dim, dim), init=SaxeInitializer())

    def __call__(self, x, y):
        U = parameter(self.U)
        return transpose(x) * U * y

    def get_components(self):
        return [self.U]

    def restore_components(self, components):
        [self.U] = components


class Biaffine(Saveable):
    def __init__(self, indim, outdim, model):
        self.U = [Bilinear(indim, model) for i in range(outdim)]
        self.x_bias = model.add_parameters((outdim, indim))
        self.y_bias = model.add_parameters((outdim, indim))
        self.bias = model.add_parameters(outdim)

    def __call__(self, x, y):
        x_bias = parameter(self.x_bias)
        y_bias = parameter(self.y_bias)
        bias = parameter(self.bias)
        ret = concatenate([u(x, y) for u in self.U])
        return ret + x_bias * x + y_bias * y + bias

    def get_components(self):
        return self.U + [self.x_bias, self.y_bias, self.bias]

    def restore_components(self, components):
        self.U = components[:-3]
        [self.x_bias, self.y_bias, self.bias] = components[-3:]


class BiaffineBatch(Saveable):
    def __init__(self, indim, outdim, model):
        self.U = [Bilinear(indim + 1, model) for i in range(outdim)]

    def __call__(self, x, y):
        x = concatenate([x, zeroes((1, x.dim()[0][1],)) + 1.])
        y = concatenate([y, zeroes((1, y.dim()[0][1],)) + 1.])
        return concatenate([u(x, y) for u in self.U], 2)

    def get_components(self):
        return self.U

    def restore_components(self, components):
        self.U = components[:-3]


def identity(x):
    return x
