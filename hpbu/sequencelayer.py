# Copyright 2019 Sebastian Kahl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" SequenceLayer
Created on 15.08.2017

@author: skahl
"""
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from .layer import *


class SequenceLayer(Layer):
    """ Sequence class for predictive processing hierarchy,
    specialized for sequencing layers that collect lower layer activity in sequence
    of occurrence and predict next steps from known sequences or learn new ones.
    """


    def __init__(self, name):
        super(SequenceLayer, self).__init__(name)
        self.type = 'Sequence'
        self.reset()



    def reset(self):
        super(SequenceLayer, self).reset()
        self.tmp_seq = []  # perceived temporary spatial sequence
        self.tmp_delay = []  # perceived temporary delay sequence
        self.prod_seq = []  # predicted part of intended sequence
        self.last_lower_layer_new_hypo = None
        self.time_since_evidence = 0.
        self.tmp_seq_average_delay = 0.
        self.production_delta_t = 0.

        self.production_look_ahead = 2
        # self.delta_t_production = 0.
        # self.prediction_estimator = None
        self.production_candidate = None
        self.best_hypo = None
        self.intention_hypo = None

        # regenerate discretized sequences and alphabets
        # for seq in self.hypotheses.reps.values():
        #     seq.discretized, seq.alphabet = discretize_coords_sequence(seq.seq, d=self.params["word_length"])

        # self.hypotheses.equalize()
        # self.hypotheses.update_idx_id_mapping()

        print("(Re-)initialized layer", self.name)



    def __str__(self):
        return "Layer '" + self.name + "', type:" + self.type



    def print_out(self):

        _str_ = self.name
        _str_ += "\nhypotheses ("
        _str_ += str(len(self.hypotheses.reps))
        _str_ += "):\n"
        _str_ += str(self.hypotheses)
        return _str_


    def integrate_evidence(self):
        """ Integrate evidence from next lower layer.
        Here, create a temporal sequence for comparison, or extend that
        temporal sequence with the given evidence.
        Compare sequences.
        """
        if self.lower_layer_evidence is not None:
            # process only minor update in order to update current likelihood
            # of current perceived movements given available sequences
            self.log(4, "new evidence", self.lower_layer_evidence)

            temporary_sequence_addition = copy(self.tmp_seq)

            if self.lower_layer_evidence[0] is not None:
                temporary_sequence_addition.append(self.lower_layer_evidence[0])

            # temporal data handling
            if self.lower_layer_evidence[1] is not None:
                # self.delta_t_production += self.lower_layer_evidence[1]
                self.time_since_evidence += self.lower_layer_evidence[1]
                self.log(3, "received only temporal delay over evidence:", self.lower_layer_evidence[1])

            if len(self.hypotheses.reps) > 1 and len(self.tmp_seq) > 0:
                lh, _, best_idx = self.update_sequences(self.hypotheses.reps, temporary_sequence_addition)
                # save as likelihood P(V|S)
                self.likelihood = lh
                self.log(3, "Best fitting sequence [lh id]:", self.likelihood[best_idx])

        if self.lower_layer_new_hypo is not None:
            # process a surprising movement and add it to a temporary sequence for further processing

            # accept and apply pruning from higher level layer, if pruning is enabled
            # self.pruning()

            # receive evidence
            self.log(4, "new surprising movement:", self.lower_layer_new_hypo)

            # temporal data handling
            # if self.lower_layer_new_hypo[1] is not None and self.lower_layer_new_hypo[0] is None:
            #     # self.delta_t_production += self.lower_layer_evidence[1]
            #     self.time_since_evidence += self.lower_layer_new_hypo[1]
            #     print(self.time_since_evidence)

            # movement data handling
            if self.lower_layer_new_hypo[0] is not None:
                self.last_lower_layer_new_hypo = copy(self.lower_layer_new_hypo)  # copy for later use
                self.time_since_evidence = 0.

                # extend temporary sequence
                self.tmp_seq.append(self.lower_layer_new_hypo[0])

                # predict temporal waiting period until next evidence
                if len(self.tmp_delay) > 0:
                    self.tmp_seq_average_delay = np_mean(self.tmp_delay) + np_var(self.tmp_delay)
                else:
                    self.tmp_seq_average_delay = 0
                self.tmp_delay.append(self.lower_layer_new_hypo[1])
                self.log(1, "Temporary sequence length:", len(self.tmp_seq), "/ spanning time:", sum(self.tmp_delay), "avg:", self.tmp_seq_average_delay)
                self.log(4, "tmp sequence:\n", self.tmp_seq)

            if len(self.hypotheses.reps) > 1:
                # and len(self.tmp_seq) > self.params["word_length"]:
                # calculate sequence likelihood
                # calculate fit for new sequence, disregarding timing
                lh, best_hypo, best_idx = self.update_sequences(self.hypotheses.reps, self.tmp_seq)
                # save as likelihood P(V|S)
                self.likelihood = lh
                self.log(2, "Best fitting sequences [lh id]:", best_hypo)

            # check for production evidence
            if self.production_candidate is not None and\
                    self.lower_layer_new_hypo[0] is not None:

                evidence_difference = self.production_candidate - self.lower_layer_new_hypo[0]
                # if evidence_difference < 0.01 or len(self.tmp_seq) < 2:
                # if len(self.tmp_seq) < 2:
                #     self.log(1, "First movement of intended production is always just a preparation, so if this was send, target was reached.")
                self.log(1, "Received good-enough production evidence for", self.production_candidate, "evidence:", self.lower_layer_new_hypo[0], "diff:", evidence_difference)
                self.production_candidate = None
                # else:
                #     self.log(1, "Received insufficient production evidence:", self.lower_layer_evidence[0], "target:", self.production_candidate, "diff:", evidence_difference)



    def td_inference(self):
        """ Receive and integrate higher layer and long range projections.
        """
        if self.long_range_projection is not None:
            self.log(3, "long range projection:", self.long_range_projection)
            if "intention" in self.long_range_projection:
                lrp = self.long_range_projection["intention"]

                # get idx of intended hypothesis:
                lrp_idx = self.hypotheses.reps[lrp].dpd_idx

                # create motor intention
                self.hypotheses.equalize()
                avg_P = np_mean(self.hypotheses.dpd[:, 0])
                var_P = 0.2
                critical_intention = avg_P + var_P  # only +1 var
                self.log(1, "setting intention", lrp, "P at critical:", critical_intention)
                # idx from rep id
                self.td_posterior = copy(self.hypotheses.dpd)
                self.td_posterior = set_hypothesis_P(self.td_posterior, lrp_idx, critical_intention)
                self.td_posterior = norm_dist(self.td_posterior, smooth=True)
                self.intention = lrp
                # reset production candidate
                self.production_candidate = None

            if "observe" in self.long_range_projection:
                self.hypotheses.equalize()
                self.production_candidate = None
                self.tmp_seq = []
                self.tmp_delay = []
                self.lower_layer_evidence = None
                self.lower_layer_new_hypo = None
                self.log(1, "setting level to reset residual observation influences")


            if "done" in self.long_range_projection:
                self.tmp_seq = []
                self.tmp_delay = []

        # and self.params["self_supervised"]: # TODO: allow higher level influence during learning?
        elif self.higher_layer_prediction is not None: 
            self.log(4, "higher layer projection:", self.higher_layer_prediction)
            higher_layer = copy(self.higher_layer_prediction)
            P_C = higher_layer[0]  # higher layer posterior distribution
            matrix = higher_layer[1]  # likelihood matrix

            # perform top-down inference from mixture of experts
            self.td_posterior = copy(self.hypotheses.dpd)
            """ correct td_posterior from P(S_C, C), including the normalized P(S|C_i) """
            # P(S_C, C) = sum_j( P(S_i|C_j) P(C_j) )
            self.td_posterior = mixture_experts(self.hypotheses.dpd, P_C, matrix, smooth=True)

            # self.log(0, "updated top-down posterior from higher layer:\n", self.td_posterior)



    def bu_inference(self):
        """ Calculate the posterior for the sequence layer, based on evidence from
        predicted lower level activity.
        """
        # careful, updating likelihood distribution here!
        # TODO: calculate time precision with regard to predicted observation or for all next elements' timing?
        # gain = 0.85  # scales 0.5 likelihood at 0.2 seconds and 0.05 likelihood at 0.5 seconds
        # gain = 0.96  # scales 0.5 likelihood at 0.115 seconds (was used for skahl & skopp (2018) simulations)
        # gain = 0.997  # scales 0.5 likelihood at 0.005 seconds
        gain = 0.9

        # local_K = self.K = kalman_gain(self.free_energy, self.PE.precision)

        if len(self.tmp_delay) > 0 and self.tmp_delay[-1] > 0.:
            if self.lower_layer_new_hypo is not None:  # or self.lower_layer_evidence is not None:
                # self-other differentiation
                # what is the evidence for intended motor belief?
                if self.intention is not None:
                    t_prec = time_precision(self.production_delta_t, self.tmp_delay[-1], gain)
                    self.likelihood[:, 0] *= t_prec
                    self.log(1, "time precision:", t_prec, "predicted delta_t:", self.production_delta_t, "received delta_t:", self.tmp_delay[-1])

                    intent_idx = self.hypotheses.reps[self.intention].dpd_idx
                    norm_LH = norm_dist(self.likelihood)
                    intent_lh = norm_LH[intent_idx]
                    max_idx = np_argmax(norm_LH[:, 0])
                    max_id = norm_LH[max_idx]

                    if self.intention == max_id[1]:
                        # A filtered signal gives nice plots, but may be unnecessary for evaluating self-other distinction
                        self.self_estimate, _ = kalman_filter(self.self_estimate, t_prec, self.K)
                    else:
                        # A filtered signal gives nice plots, but may be unnecessary for evaluating self-other distinction
                        self.self_estimate, _ = kalman_filter(self.self_estimate, intent_lh[0], self.K)
                    self.log(1, "Likelihood statistics (intent hypo / max hypo):", intent_lh, '/', max_id, "K:", self.K)
                elif len(self.tmp_delay) > 2: # need a minimum of two observations before a new judgement
                    self.self_estimate, _ = kalman_filter(self.self_estimate, 0., self.K)

                # store self estimate for last action
                if self.personmodel is not None:
                    self.personmodel.self_estimate = self.self_estimate
                    
        # calculate bottom-up posterior
        self.bu_posterior = posterior(self.hypotheses.dpd, copy(self.likelihood), smooth=True)
        # self.log(0, "current P_bu:\n", self.bu_posterior)
        # print("seq bu-posterior:\n", self.bu_posterior)



    # @profile
    def update_sequences(self, sequences_dict, tmp_seq):
        """ Compare a dictionary of sequences with a new sequence.
        """
        new_seq = np.array(tmp_seq)

        # take new sequence and compare with other sequences (sequence level comparison)
        sequences = np.array([[seq, key_id] for key_id, seq in sequences_dict.items()])
        diffs_LH = np.zeros((sequences.shape[0], 2))
        diffs_LH[:, 1] = sequences[:, 1]  # create result structure "diffs_LH"

        # create temporary sequence discretization
        _seq = Sequence()
        _seq['seq'] = new_seq

        # try to accelerate processing here:
        # diffs_LH[:, 0] = [diff_sequences(_seq, seq_b[0]) for seq_b in sequences]

        for idx, seq in enumerate(sequences):
            # extend the new_seq sequence with the sequence that it will be compared to
            # the better fit will still win
            
            # s_shape = seq[0].seq.shape[0]
            # tmp_shape = _seq.seq.shape[0]

            _diff_lh = diff_sequences(_seq, seq[0])

            # check if the size of the sequences are sufficient
            # if tmp_shape <= s_shape:
            #     # weight diff with respect to the sequence length's fraction of the sequence compared to
            #     # scale_to_len = tmp_shape / s_shape

            #     _diff_lh = diff_sequences(_seq, seq[0])  # * scale_to_len

            # else:
            #     # if probability distributions are too small, rate minimal comparability
            #     # actually, this shouldn't happen
            #     _diff_lh = 0.0001

            diffs_LH[idx, 0] = _diff_lh

        diffs_LH = diffs_LH[diffs_LH[:, 1].argsort()]  # in place sorting
        idx_max_P = np_argmax(diffs_LH[:, 0])  # so max is BEST

        return diffs_LH, sequences_dict[diffs_LH[idx_max_P, 1]], idx_max_P


    def extension(self):
        """ Decide on and do hypothesis extension and
        decide on evidence for next higher layer.
        """

        gamma_extension_threshold = self.params["gamma"]

        ltmpseq = len(self.tmp_seq)

        if ltmpseq > 0:
            # if there is a surprisal, we probably perceived an unexpected coordinate
            # or it could have been received from next higher layer

            if len(self.tmp_delay) > 1:
                average_delay = self.tmp_seq_average_delay + gamma_extension_threshold
                # seems necessary for production self evidenced delay surprise
                self.time_since_evidence += self.params['time_step']
                cur_delay = self.tmp_delay[-1] + self.time_since_evidence
                self.log(3, "current delay since last update:", cur_delay)
            else:
                average_delay = 0. 
                cur_delay = 0.

            """ Add new sequence if best_hypo precision is lower than average
            and its free-energy is higher than average layer free-energy
            and the current time delay is bigger than average delay
            """
            # self.log(4, "delay:\t", cur_delay, '\t>\t', average_delay, '\n',
            # self.log(3, "\t\tsurprise:\t", self.PE)
            
            # not expecting the delay duration of a jump and average delay is smaller than current delay
            if self.params["self_supervised"]\
                    and ltmpseq > 3 and cur_delay > average_delay\
                    and (len(self.hypotheses.reps) < 3 or self.PE.is_surprising()):

                self.log(0, "\t\tsurprise:\t", self.PE)
                self.log(0, "="*10,">  ADDED!\n")

                # add new sequence
                new_hypo = self.hypotheses.add_hypothesis(Sequence, P=0.1)

                # store the index of the max probability for each stored coordinate (without last one!)
                new_hypo['seq'] = np.array(self.tmp_seq)
                new_hypo['delta_t_seq'] = self.tmp_delay
                # new_hypo['discretized'], new_hypo['alphabet'] = discretize_coords_sequence(self.tmp_seq, d=self.params["word_length"])

                # inform next higher layer about new sequence hypothesis
                self.layer_new_hypo = new_hypo

                # clear temporary sequence collection
                self.tmp_seq = []
                self.tmp_delay = []
                self.time_since_evidence = 0.

                self.log(1, "Surprising PE in hypos:\t", self.PE)
                self.log(1, "Added new sequence with id", new_hypo['id'], '(', len(new_hypo['seq']), ')')
                # self.log(1, "last element:", new_hypo['seq'][-1])
                self.log(3, new_hypo['seq'])
                # self.PE.clear()

                # signal surprise to other layers
                self.layer_long_range_projection = {}
                self.layer_long_range_projection["Schm"] = {"surprise": "Seq"}
                # self.layer_long_range_projection["Realizations"] = {"surprise": "Seq"}
                self.layer_long_range_projection["Vision"] = {"done": "Seq"}

            elif self.intention_hypo is not None and len(self.prod_seq) >= len(self.intention_hypo["seq"]) and cur_delay > average_delay:
                # no next step possible and no surprisal
                # should not be possible, so mention it
                self.log(1, "No known next steps in current intended motor belief", self.intention_hypo['id'])
                self.layer_long_range_projection = {}
                self.layer_long_range_projection["MC"] = {"done": self.intention}
                self.layer_long_range_projection["Vision"] = {"done": self.intention}
                self.layer_long_range_projection["Schm"] = {"done": self.intention}
                self.layer_long_range_projection["Realizations"] = {"done": self.self_estimate}
                self.intention = None
                self.intention_hypo = None
                self.tmp_seq = []
                self.tmp_delay = []
                self.prod_seq = []
                self.time_since_evidence = 0.
                # self.PE.clear()

            # TODO: this is why we can't have nice things... trust the input or not?
            elif cur_delay > average_delay:
                self.layer_long_range_projection = {}
                self.layer_long_range_projection["Schm"] = {"surprise": "Seq"}
                self.layer_long_range_projection["Realizations"] = {"surprise": "Seq"}
                self.layer_long_range_projection["Vision"] = {"done": "Seq"}
                self.log(1, "Only surprising delay:", cur_delay, ">", average_delay)
                self.log(1, self.PE, "with", ltmpseq, "tmp elements. clearing visual buffer!")
                self.log(3, "tmp delay elements:", self.tmp_delay, "+", self.time_since_evidence)
                # self.PE.clear()
                # clear temporary sequence collection
                self.tmp_seq = []
                self.tmp_delay = []
                self.time_since_evidence = 0.

        self.layer_evidence = [self.hypotheses, self.params["time_step"], self.self_estimate]



    def prediction(self):
        """ Decide on predicted next lower layer activity in best predicted sequence.
        """

        """ Produce a lower layer influencing prediction if there is a best hypothesis or intent.
        """
        if self.intention is not None or len(self.hypotheses.reps) > 1:

            # check if higher layer intention was only just communcated:
            if self.long_range_projection is not None and "intention" in self.long_range_projection:
                # then clear temporary sequences to make room for indented production
                self.tmp_seq = []
                self.tmp_delay = []

            if self.intention is not None:
                self.intention_hypo = self.hypotheses.reps[self.intention]
                sequence = self.intention_hypo['seq']
                delta_ts = self.intention_hypo['delta_t_seq']

                if self.production_candidate is None:
                    self.log(2, "Projecting sequence " + str(self.intention_hypo['id']))
                    self.log(3, sequence)
                    self.log(3, "cur tmp seq:", self.tmp_seq)

                    # check if a next action is possible at all
                    lpredseq = len(self.prod_seq)
                    # TODO: skip last element
                    if lpredseq < len(sequence):
                        # select next action
                        self.production_candidate = sequence[lpredseq]  # np_sum(sequence[:ltmpseq + 1], axis=0)  # coordinate and id
                        
                        # the time taken for jump can be unexpected so better ignore it mostly
                        if not self.production_candidate.drawing:
                            self.production_delta_t = 0.01
                            self.log(1, "preparing to expect jump duration")
                        else:
                            self.production_delta_t = delta_ts[lpredseq]
                        motor_delta_t = self.production_delta_t  # np_sum(delta_ts) / (len(sequence) - 1) / 2]  # [ltmpseq]
                        
                        

                        if lpredseq == 0:
                            self.production_delta_t = 0
                            self.production_candidate.drawing = False

                        if lpredseq + self.production_look_ahead < len(sequence):
                            # check for undrawn part in between
                            stop_idx = 0
                            for idx, step in enumerate(list(sequence[lpredseq:lpredseq + self.production_look_ahead + 1])):
                                stop_idx = idx
                                if not step.drawing:
                                    # print("jump at idx", stop_idx)
                                    break

                            if stop_idx > 0:
                                look_ahead_candidate = list(sequence[lpredseq:lpredseq + stop_idx])  # self.production_look_ahead + 1])  # np_sum(sequence[:ltmpseq + self.production_look_ahead + 1], axis=0)
                            else:
                                look_ahead_candidate = None

                        else:
                            look_ahead_candidate = list(sequence[lpredseq:])  # np_sum(sequence, axis=0)
                        self.log(2, self.production_look_ahead, "step look ahead goal:", look_ahead_candidate)

                        # # let timing take effect
                        # if (time() - self.delta_t_production) >= delta_t:
                        # self.delta_t_production = 0.

                        if self.intention is not None:
                            # here, what actually has to be predicted is the angle in the next lower layer (vision)
                            # or the next coordinate (MotorControl)
                            self.layer_long_range_projection = {}
                            self.layer_long_range_projection["MC"] = {"goal": [self.production_candidate, motor_delta_t]}
                            self.layer_long_range_projection["MC"].update({"look_ahead_goal": look_ahead_candidate})
                            self.layer_long_range_projection["Vision"] = {"goal": [self.production_candidate]}
                        
                        # store current production candidate in temporary sequence
                        self.prod_seq.append(self.production_candidate)


        # regardless of any intention signal, calculate distribution from the posterior probability that
        # can be understood by Vision and Motor layers
        if len(self.tmp_seq) >= 1 and self.bu_posterior is not None:
            # max_idx = np_argmax(self.bu_posterior[:, 0])
            max_id = self.hypotheses.max()[1]  # self.bu_posterior[max_idx, 1]
            best_s = self.hypotheses.reps[max_id]  # is this different than an available intention?

            _seq = Sequence()
            _seq['seq'] = self.tmp_seq
            # _seq['discretized'], _seq['alphabet'] = discretize_coords_sequence(_seq['seq'], d=self.params["word_length"])
            last_fit = diff_sequences(_seq, best_s)
            # predefine extended sequences
            extended_seqs = []
            for v in self.lower_layer_hypos.dpd:
                _seq = Sequence()
                _seq['seq'] = self.tmp_seq + [Coord(1, v[1], True)]
                # _seq['discretized'], _seq['alphabet'] = discretize_coords_sequence(_seq['seq'], d=self.params["word_length"])
                extended_seqs.append([_seq, v[1]])
            
            # communicating P(M|S) and P(V|S) likelihoods:

            # trying to accelerate comparisons
            # calculate difference with default length=1 and drawing=True for all movement angles v
            MV_S = np.array([[diff_sequences(seq_a, best_s), v] for seq_a, v in extended_seqs])
            MV_S[:, 0] = MV_S[:, 0] / last_fit
            self.layer_prediction = norm_dist(MV_S, smooth=True)

            # define a predicted delta_t from current max hypo
            delta_t_seq = best_s['delta_t_seq']
            if len(self.tmp_seq) < len(delta_t_seq) - 1:
                self.production_delta_t = delta_t_seq[len(self.tmp_seq)]
            else:
                self.production_delta_t = 0
        elif self.lower_layer_hypos is not None:
            lower_layer_dpd = copy(self.lower_layer_hypos.dpd)
            self.layer_prediction = lower_layer_dpd


        self.log(4, "prediction:", self.layer_prediction)


