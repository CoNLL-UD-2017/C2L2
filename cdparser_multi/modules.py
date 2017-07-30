#!/usr/bin/env python
# encoding: utf-8

from dynet import *

import pyximport; pyximport.install()
from .calgorithm import parse_proj, parse_ah_dp_mst, parse_ae_dp_mst
from .ahbeamconf import AHBeamConf
from .aebeamconf import AEBeamConf

from .layers import MultiLayerPerceptron, Dense, Bilinear, identity, BiaffineBatch


class UPOSTagger:

    def __init__(self, parser, id="UPOSTagger", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._utagger_mlp_activation = self._activations[kwargs.get('utagger_mlp_activation', 'relu')]
        self._utagger_mlp_dims = kwargs.get("utagger_mlp_dims", 128)
        self._utagger_mlp_layers = kwargs.get("utagger_mlp_layers", 2)
        self._utagger_mlp_dropout = kwargs.get("utagger_mlp_dropout", 0.0)
        self._utagger_discrim = kwargs.get("utagger_discrim", False)

    def init_params(self):
        self._utagger_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._utagger_mlp_dims] * self._utagger_mlp_layers, self._utagger_mlp_activation, self._parser._model)
        self._utagger_final = Dense(self._utagger_mlp_dims, len(self._parser._upos), identity, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._utagger_mlp.set_dropout(self._utagger_mlp_dropout)
        else:
            self._utagger_mlp.set_dropout(0.)

    def sent_loss(self, graph, carriers):
        ret = []
        correct = 0

        for node, c in zip(graph.nodes[1:], carriers[1:]):
            potentials = self._utagger_final(self._utagger_mlp(c.vec))
            pred = np.argmax(potentials.value())
            answer = self._parser._upos[node.upos]

            if (pred == answer):
                correct += 1

            if self._utagger_discrim:
                potential_values = potentials.value()
                best_wrong = max([(i, val) for i, val in enumerate(potential_values) if i != answer], key=lambda x: x[1])

                if best_wrong[1] + 1. > potential_values[answer]:
                    ret.append((potentials[best_wrong[0]] - potentials[answer] + 1.))
            else:
                ret.append(pickneglogsoftmax(potentials, answer))
        return correct, ret

    def predict(self, graph, carriers):
        for node, c in zip(graph.nodes[1:], carriers[1:]):
            potentials = self._utagger_final(self._utagger_mlp(c.vec))
            pred = np.argmax(potentials.value())
            node.upos = self._parser._iupos[pred]

        return self

class XPOSTagger:
    def __init__(self, parser, id="XPOSTagger", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._xtagger_mlp_activation = self._activations[kwargs.get('xtagger_mlp_activation', 'relu')]
        self._xtagger_mlp_dims = kwargs.get("xtagger_mlp_dims", 128)
        self._xtagger_mlp_layers = kwargs.get("xtagger_mlp_layers", 2)
        self._xtagger_mlp_dropout = kwargs.get("xtagger_mlp_dropout", 0.0)
        self._xtagger_discrim = kwargs.get("xtagger_discrim", False)

    def init_params(self):
        self._xtagger_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._xtagger_mlp_dims] * self._xtagger_mlp_layers, self._xtagger_mlp_activation, self._parser._model)
        self._xtagger_final = Dense(self._xtagger_mlp_dims, len(self._parser._xpos), identity, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._xtagger_mlp.set_dropout(self._xtagger_mlp_dropout)
        else:
            self._xtagger_mlp.set_dropout(0.)

    def sent_loss(self, graph, carriers):
        ret = []
        correct = 0

        for node, c in zip(graph.nodes[1:], carriers[1:]):
            potentials = self._xtagger_final(self._xtagger_mlp(c.vec))
            pred = np.argmax(potentials.value())
            answer = self._parser._xpos[node.xupos]

            if (pred == answer):
                correct += 1

            if self._xtagger_discrim:
                potential_values = potentials.value()
                best_wrong = max([(i, val) for i, val in enumerate(potential_values) if i != answer], key=lambda x: x[1])

                if best_wrong[1] + 1. > potential_values[answer]:
                    ret.append(potentials[best_wrong[0]] - potentials[answer] + 1.)
            else:
                ret.append(pickneglogsoftmax(potentials, answer))
        return correct, ret

    def predict(self, graph, carriers):
        for node, c in zip(graph.nodes[1:], carriers[1:]):
            potentials = self._xtagger_final(self._xtagger_mlp(c.vec))
            pred = np.argmax(potentials.value())
            node.xpos = self._parser._ixpos[pred].split("|")[1]

        return self


class MSTParser:
    def __init__(self, parser, id="MSTParser", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._mst_mlp_activation = self._activations[kwargs.get('mst_mlp_activation', 'relu')]
        self._mst_mlp_dims = kwargs.get("mst_mlp_dims", 128)
        self._mst_mlp_layers = kwargs.get("mst_mlp_layers", 2)
        self._mst_mlp_dropout = kwargs.get("mst_mlp_dropout", 0.0)
        self._mst_discrim = kwargs.get("mst_discrim", False)

    def init_params(self):
        self._mst_head_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mst_mlp_dims] * self._mst_mlp_layers, self._mst_mlp_activation, self._parser._model)
        self._mst_mod_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._mst_mlp_dims] * self._mst_mlp_layers, self._mst_mlp_activation, self._parser._model)
        self._mst_bilinear = Bilinear(self._mst_mlp_dims, self._parser._model)
        self._mst_head_bias = Dense(self._mst_mlp_dims, 1, identity, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._mst_head_mlp.set_dropout(self._mst_mlp_dropout)
            self._mst_mod_mlp.set_dropout(self._mst_mlp_dropout)
        else:
            self._mst_head_mlp.set_dropout(0.)
            self._mst_mod_mlp.set_dropout(0.)

    def _mst_arcs_eval(self, carriers):
        head_vecs = [self._mst_head_mlp(c.vec) for c in carriers]
        mod_vecs = [self._mst_mod_mlp(c.vec) for c in carriers]

        head_vecs = concatenate(head_vecs, 1)
        mod_vecs = concatenate(mod_vecs, 1)

        exprs = colwise_add(self._mst_bilinear(head_vecs, mod_vecs), reshape(self._mst_head_bias(head_vecs), (len(carriers),)))

        scores = exprs.value()

        return scores, exprs

    def sent_loss(self, graph, carriers):
        gold_heads = graph.proj_heads
        scores, exprs = self._mst_arcs_eval(carriers)

        if self._mst_discrim:
            # Cost Augmentation
            for m, h in enumerate(gold_heads):
                scores[h, m] -= 1.

            heads = parse_proj(scores)
            correct = sum([1 for (h, g) in zip(heads[1:], gold_heads[1:]) if h == g])
            loss = [exprs[int(h)][int(i)] - exprs[int(g)][int(i)] + 1. for i, (h, g) in enumerate(zip(heads, gold_heads)) if h != g]
        else:
            heads = parse_proj(scores)
            correct = sum([1 for (h, g) in zip(heads[1:], gold_heads[1:]) if h == g])
            loss = [-exprs[gold_heads[d], d] + logsumexp(list(exprs[:, d])) for d in range(1, len(gold_heads))]

        return correct, loss

    def predict(self, graph, carriers):
        scores, exprs = self._mst_arcs_eval(carriers)
        graph.heads = parse_proj(scores)

        return self

class AHDPParser:
    def __init__(self, parser, id="AHDPParser", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._ah_mlp_activation = self._activations[kwargs.get('ah_mlp_activation', 'relu')]
        self._ah_mlp_dims = kwargs.get("ah_mlp_dims", 128)
        self._ah_mlp_layers = kwargs.get("ah_mlp_layers", 2)
        self._ah_mlp_dropout = kwargs.get("ah_mlp_dropout", 0.0)

    def init_params(self):
        self._ah_pad_repr = [self._parser._model.add_parameters(self._bilstm_dims) for i in range(2)]
        self._ah_stack_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._ah_mlp_dims] * self._ah_mlp_layers, self._ah_mlp_activation, self._parser._model)
        self._ah_buffer_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._ah_mlp_dims] * self._ah_mlp_layers, self._ah_mlp_activation, self._parser._model)
        self._ah_scorer = BiaffineBatch(self._ah_mlp_dims, 3, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._ah_stack_mlp.set_dropout(self._ah_mlp_dropout)
            self._ah_buffer_mlp.set_dropout(self._ah_mlp_dropout)
        else:
            self._ah_stack_mlp.set_dropout(0.)
            self._ah_buffer_mlp.set_dropout(0.)

    def _ah_confs_eval(self, carriers):
        rows = list(range(-1, len(carriers))) + [-2]
        vecs = [carriers[f].vec if f >= 0 else parameter(self._ah_pad_repr[f]) for i, f in enumerate(rows)]
        vecs = concatenate(vecs, 1)
        exprs = self._ah_scorer(self._ah_stack_mlp(vecs), self._ah_buffer_mlp(vecs))
        scores = exprs.value()
        return scores, exprs

    def _ah_seq_loss(self, correctseq, wrongseq, beamconf, loss, carriers, exprs, loc=0):
        commonprefix = 0
        for i in range(min(len(correctseq), len(wrongseq))):
            if wrongseq[i] == correctseq[i]:
                commonprefix = i + 1
            else:
                break

        beamconf.init_conf(loc, True)
        for i in range(commonprefix):
            beamconf.make_transition(loc, wrongseq[i])
        for i in range(commonprefix, len(wrongseq)):
            s, b = beamconf.extract_features(loc)
            if b < 0:
                b = len(carriers)
            loss.append(exprs[int(s+1)][int(b+1)][int(wrongseq[i])])
            beamconf.make_transition(loc, wrongseq[i])
        beamconf.init_conf(loc, True)
        for i in range(commonprefix):
            beamconf.make_transition(loc, correctseq[i])
        for i in range(commonprefix, len(correctseq)):
            s, b = beamconf.extract_features(loc)
            if b < 0:
                b = len(carriers)
            loss.append(-exprs[int(s+1)][int(b+1)][int(correctseq[i])])
            beamconf.make_transition(loc, correctseq[i])

    def sent_loss(self, graph, carriers):
        gold_heads = graph.proj_heads

        loss = []
        beamconf = AHBeamConf(len(graph.nodes), 1, np.array(gold_heads), 1, 1)
        beamconf.init_conf(0, True)

        scores, exprs = self._ah_confs_eval(carriers)

        # cost augmentation
        mst_scores = np.ones((len(graph.nodes), len(graph.nodes)))
        for m, h in enumerate(gold_heads):
            mst_scores[h, m] -= 1.

        pred_transitions, pred_heads = parse_ah_dp_mst(scores, mst_scores)
        true_transitions = beamconf.gold_transitions(0, True)

        self._ah_seq_loss(true_transitions, pred_transitions, beamconf, loss, carriers, exprs, loc=0)

        return sum([1 if gold_heads[i] == pred_heads[i] else 0 for i in range(len(graph.nodes))]), loss

    def predict(self, graph, carriers):
        beamconf = AHBeamConf(len(graph.nodes), 1, np.array(graph.heads), 1, 1)
        beamconf.init_conf(0, True)


        scores, exprs = self._ah_confs_eval(carriers)
        mst_scores = np.zeros((len(graph.nodes), len(graph.nodes)))
        transitions, heads = parse_ah_dp_mst(scores, mst_scores)

        graph.heads = heads

        return self

class AEDPParser:
    def __init__(self, parser, id="AEDPParser", **kwargs):
        self._parser = parser
        self.id = id

        self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        self._bilstm_dims = kwargs.get("bilstm_dims", 128)
        self._ae_mlp_activation = self._activations[kwargs.get('ae_mlp_activation', 'relu')]
        self._ae_mlp_dims = kwargs.get("ae_mlp_dims", 128)
        self._ae_mlp_layers = kwargs.get("ae_mlp_layers", 2)
        self._ae_mlp_dropout = kwargs.get("ae_mlp_dropout", 0.0)

    def init_params(self):
        self._ae_pad_repr = [self._parser._model.add_parameters(self._bilstm_dims) for i in range(2)]
        self._ae_stack_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._ae_mlp_dims] * self._ae_mlp_layers, self._ae_mlp_activation, self._parser._model)
        self._ae_buffer_mlp = MultiLayerPerceptron([self._bilstm_dims] + [self._ae_mlp_dims] * self._ae_mlp_layers, self._ae_mlp_activation, self._parser._model)
        self._ae_scorer = BiaffineBatch(self._ae_mlp_dims, 4, self._parser._model)

    def init_cg(self, train=False):
        if train:
            self._ae_stack_mlp.set_dropout(self._ae_mlp_dropout)
            self._ae_buffer_mlp.set_dropout(self._ae_mlp_dropout)
        else:
            self._ae_stack_mlp.set_dropout(0.)
            self._ae_buffer_mlp.set_dropout(0.)

    def _ae_confs_eval(self, carriers):
        rows = list(range(-1, len(carriers))) + [-2]

        vecs = [carriers[f].vec if f >= 0 else parameter(self._ae_pad_repr[f]) for i, f in enumerate(rows)]
        vecs = concatenate(vecs, 1)

        exprs = self._ae_scorer(self._ae_stack_mlp(vecs), self._ae_buffer_mlp(vecs))

        scores = exprs.value()

        return scores, exprs

    def _ae_seq_loss(self, correctseq, wrongseq, beamconf, loss, carriers, exprs, loc=0):
        commonprefix = 0
        for i in range(min(len(correctseq), len(wrongseq))):
            if wrongseq[i] == correctseq[i]:
                commonprefix = i + 1
            else:
                break

        beamconf.init_conf(loc, True)
        for i in range(commonprefix):
            beamconf.make_transition(loc, wrongseq[i])
        for i in range(commonprefix, len(wrongseq)):
            s, b = beamconf.extract_features(loc)
            if b < 0:
                b = len(carriers)
            loss.append(exprs[int(s+1)][int(b+1)][int(wrongseq[i])])
            beamconf.make_transition(loc, wrongseq[i])
        beamconf.init_conf(loc, True)
        for i in range(commonprefix):
            beamconf.make_transition(loc, correctseq[i])
        for i in range(commonprefix, len(correctseq)):
            s, b = beamconf.extract_features(loc)
            if b < 0:
                b = len(carriers)
            loss.append(-exprs[int(s+1)][int(b+1)][int(correctseq[i])])
            beamconf.make_transition(loc, correctseq[i])

    def sent_loss(self, graph, carriers):
        gold_heads = graph.proj_heads

        loss = []
        beamconf = AEBeamConf(len(graph.nodes), 1, np.array(gold_heads), 1, 1)
        beamconf.init_conf(0, True)

        scores, exprs = self._ae_confs_eval(carriers)
        # cost augmentation
        mst_scores = np.ones((len(graph.nodes), len(graph.nodes)))
        for m, h in enumerate(gold_heads):
            mst_scores[h, m] -= 1.
        pred_transitions, pred_heads = parse_ae_dp_mst(scores, mst_scores)
        true_transitions = beamconf.gold_transitions(0, True)

        self._ae_seq_loss(true_transitions, pred_transitions, beamconf, loss, carriers, exprs, loc=0)

        return sum([1 if gold_heads[i] == pred_heads[i] else 0 for i in range(len(graph.nodes))]), loss

    def predict(self, graph, carriers):
        beamconf = AEBeamConf(len(graph.nodes), 1, np.array(graph.heads), 1, 1)
        beamconf.init_conf(0, True)

        scores, exprs = self._ae_confs_eval(carriers)
        mst_scores = np.zeros((len(graph.nodes), len(graph.nodes)))
        transitions, heads = parse_ae_dp_mst(scores, mst_scores)

        graph.heads = heads

        return self

#  class MSTParser:
    #  def __init__(self, parser, id="UPOSTagger", **kwargs):
        #  self._parser = parser
        #  self.id = id

        #  self._activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}

        #  self._bilstm_dims = kwargs.get("bilstm_dims", 128)

    #  def init_params(self):

    #  def init_cg(self, train=False):
        #  if train:
        #  else:

    #  def sent_loss(self, graph, carriers):

    #  def predict(self, graph, carriers):
