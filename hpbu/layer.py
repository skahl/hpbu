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

""" Layer
Created on 15.08.2017

@author: skahl
"""
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals
from .functions import *
from .representations import *
from .prediction_error import PredictionError


class Layer(object):
    """ Base Layer class for predictive processing hierarchy.
    """

    def __init__(self, name):
        # layer identity and parameters
        self.name = name
        self.enabled_processing = True
        self.logger = None
        self.params = {}
        self.is_top_layer = False
        self.personmodel = None

        # layer state
        self.hypotheses = Hypotheses()
        self.reset()

        # hierarchy functions
        self.hierarchy_sleep = None


    
    def finalize(self):
        """ finalize gets called upon the end of the hierarchy main loop
        """
        pass



    def reset(self):

        # update control
        self.dreaming = False  # control the layer's dream state
        self.direct_mode = False  # faster update needed, update distributions but circumvent slow Bayes
        self.personmodel_influence = False  # set True if knowledge from PersonModel should influence belief updates 

        # distributions, inferences, and statistics
        self.td_posterior = None  # posterior calculated from top-down projection
        self.bu_posterior = None  # posterior calculated from bottom-up evidence

        self.PE = PredictionError()
        self.K = 0.5  # layer Kalman gain
        # self.precision = 1.
        # self.lower_layer_K = 0.5
        self.higher_layer_K = 0.5

        self.free_energy = 0.1  # layer free energy calculated between available posteriors
        self.transient_PE = deque(maxlen=20)  # store past prediction-error
        self.likelihood = []  # calculated likelihood distribution given the bottom-up evidence
        self.self_estimate = 0.  # sense of agency estimate for current production
        self.intention = None  # intention is set by long range projections

        # convenient variables
        self.best_hypo = None  # currently best hypothesis
        self.last_t_hypotheses = None

        # layer communication
        self.higher_layer_prediction = None  # in
        self.higher_layer_pruning = None  # in

        self.layer_evidence = None  # out
        self.layer_prediction = None  # out
        self.layer_new_hypo = None  # out
        self.layer_pruning = None  # out

        self.lower_layer_hypos = None  # to be initialized
        self.lower_layer_evidence = None  # initialized
        self.lower_layer_new_hypo = None  # in
        self.long_range_projection = None  # in
        print("(Re-)initialized base Layer")



    def __str__(self):
        return "Layer '" + self.name + "'"



    def set_logger(self, color):
        """ Initialize and set log and error methods.
        """
        self.logger = Logger(color, self.name)
        self.log = self.logger.log
        self.error = self.logger.error



    def set_parameters(self, parameters):
        """ Initialize layer parameters from configuration.
        """
        self.params = parameters



    def update(self):
        """ All the prediction-error handling logic.
        """

        # in order to disable processing from this layer and above, set this to False
        if not self.enabled_processing:
            return

        # store current hypothesis for later comparison
        self.last_t_hypotheses = copy(self.hypotheses.dpd)

        # if self.higher_layer_pruning is not None \
        #         or self.lower_layer_evidence is not None \
        #         or self.lower_layer_new_hypo is not None  \
        #         or self.long_range_projection is not None:
        # incorporate new evidence
        self.integrate_evidence()

        if self.higher_layer_prediction is not None \
                or self.long_range_projection is not None:
            # incorporate higher layer and long distance projections
            self.td_inference()  # calculate the top-down posterior
        
        # calculate inference
        if len(self.hypotheses.reps) > 0 and\
                self.likelihood is not None and\
                (self.lower_layer_evidence is not None or
                    self.lower_layer_new_hypo is not None):
            # Bayesian update of posterior distribution
            self.bu_inference()  # calculate the bottom-up posterior

        if len(self.hypotheses.reps) > 1 and self.td_posterior is not None and self.bu_posterior is not None:

            # calculate free-energy and K
            if self.higher_layer_prediction is not None or\
                    self.long_range_projection is not None or\
                    self.lower_layer_evidence is not None or\
                    self.lower_layer_new_hypo is not None:

                # active inference case, where top-down drives change
                prior = self.td_posterior[:, 0]    # driving signal
                post = self.bu_posterior[:, 0]     # adapted posterior

                try:
                    # calculate free energy
                    F = free_energy(P=prior, Q=post)
                    # print("intention:", self.intention, "post:", post.shape, "prior:", prior.shape)
                    self.free_energy = F[0]
                    self.PE.new(surprise=F[3], P=prior, Q=post)
                    # self.log(3, "max for driving signal:", max_p, "posterior:", max_q)
                    self.log(4, "free energy:", self.free_energy, "surprise:", F[1], "cross-entropy:", F[2])

                    # check if all posteriors add up to one, and tell warning if not
                    sum_bu = np_sum(post)
                    if sum_bu > 1.1 or sum_bu < 0.9:
                        self.error("bu_posterior not normalized:", sum_bu)

                    sum_td = np_sum(prior)
                    if sum_td > 1.1 or sum_td < 0.9:
                        self.error("td_posterior not normalized:", sum_td)

                    # if self.higher_layer_K != self.K:
                    #     self.log(0, "switching to different K:", self.K, "->", self.higher_layer_K)
                    # shorten this...
                    # hl_k = self.higher_layer_K if self.higher_layer_K is not None else self.K
                    hl_k = self.K
                    # ll_k = self.lower_layer_K if self.lower_layer_K is not None else self.K

                    self.log(4, "calculating belief update with K =", hl_k)
                    # we are interested in the top-down adaptation
                    if self.personmodel_influence and len(self.personmodel.agents) > 0:
                        # self.log(1, "PersonModel Influence in layer", self.name, "for agents:", self.personmodel.agents)
                        # allow for multiple signals from PersonModel prior knowledge to influence belief updates
                        self.personmodel.inform_sequence_mapping(self.hypotheses.reps)
                        self.hypotheses.dpd = inhibition_belief_update(self.td_posterior, self.bu_posterior, hl_k, self.personmodel)
                    else:
                        # update beliefs without prior PersonModel knowledge
                        self.hypotheses.dpd = inhibition_belief_update(self.td_posterior, self.bu_posterior, hl_k)

                    # self.log(0, "K:", hl_k, "\nTD:\n", self.td_posterior, "\nBU:\n", self.bu_posterior, "\nP:\n", self.hypotheses.dpd) 
                    
                    # if self.name == "Realizations":
                    #     self.log(1, "Realizations K that was applied:", hl_k)
                    #     self.log(1, "from td:", self.td_posterior, "\nand bu:", self.bu_posterior)
                    #     self.log(1, "resulting in:", self.hypotheses.dpd)

                    # update dpd_idx mapping
                    self.hypotheses.update_idx_id_mapping()
                except Exception as e:
                    self.error(str(e))
                    sys.exit(1)

            self.log(4, "full belief update activity")
        elif len(self.hypotheses.reps) > 1 and self.bu_posterior is not None:
            # default free energy if no distribution for comparison is given, so we compare what we have
            # with the last posterior
            self.log(4, "calculating partial free energy update")
            post = self.bu_posterior[:, 0]
            prior = self.hypotheses.dpd[:, 0]
            self.log(5, "post:", self.bu_posterior, "prior:", self.hypotheses.dpd)

            # calculate free energy
            F = free_energy(P=prior, Q=post)
            self.free_energy = F[0]
            self.PE.new(surprise=F[3], P=prior, Q=post)
            self.log(4, "free energy:", self.free_energy, "surprise:", F[1], "cross-entropy:", F[2])

            self.hypotheses.dpd = inhibition_belief_update(self.hypotheses.dpd, self.bu_posterior, self.K)
            # self.hypotheses.dpd = self.bu_posterior

            # update dpd_idx mapping
            self.hypotheses.update_idx_id_mapping()
            # self.log(1, self.hypotheses.max())
            self.log(4, "partial belief update activity")
        elif len(self.hypotheses.reps) > 1 and self.td_posterior is not None:
            # default free energy if no distribution for comparison is given, so we compare what we have
            # with the last posterior
            self.log(4, "calculating partial free energy update")

            # if self.higher_layer_K != self.K:
            #     self.log(0, "switching to different K:", self.K, "->", self.higher_layer_K)
            # shorten this...
            # hl_k = self.higher_layer_K if self.higher_layer_K is not None else self.K
            hl_k = self.K
            # ll_k = self.lower_layer_K if self.lower_layer_K is not None else self.K

            self.log(4, "calculating belief update with K =", hl_k)
            prior = self.td_posterior[:, 0]
            post = self.hypotheses.dpd[:, 0]
            self.log(5, "post:", self.hypotheses.dpd, "prior:", self.td_posterior)

            # calculate free energy
            F = free_energy(P=prior, Q=post)
            self.free_energy = F[0]
            self.PE.new(surprise=F[3], P=prior, Q=post)
            self.log(4, "free energy:", self.free_energy, "surprise:", F[1], "cross-entropy:", F[2])

            self.hypotheses.dpd = inhibition_belief_update(self.td_posterior, self.hypotheses.dpd, hl_k)
            # self.hypotheses.dpd = self.td_posterior

            # update dpd_idx mapping
            self.hypotheses.update_idx_id_mapping()
            # self.log(1, self.hypotheses.max())
            self.log(4, "partial belief update activity")
        else:
            self.log(4, "no belief update!")
        
        # fixate the intention dependent kalman gain
        self.set_intention_dependent_kalman_gain()
        # self.set_variable_kalman_gain()
        # self.K = kalman_gain(self.free_energy, self.PE.precision)
    
        # self.log(1, "gain bias:", bias, "resulting K:", self.K)
        # gain_gain is how strong the bias is enforced

        # apply extensions
        self.extension()

        # reduce by 0.9
        # self.hypotheses.dpd[:, 0] *= 0.75
        # self.hypotheses.dpd = norm_dist(self.hypotheses.dpd, smooth=True)

        # new prediction
        if self.hypotheses.dpd.shape[0] > 0:
            self.prediction()

        self.log(4, "extension and prediction activity")



    def set_variable_kalman_gain(self):
        """ Allow for dynamic Kalman gain based on free-energy and precision of prediction-error.
        Distribute a linear gain-bias for bottom-up information, decreasing with every hierarchical level.
        """
        # if "bias_gain" in self.params:
        #     gain_gain = self.params['bias_gain']
        # else:
            
        # gain_gain = 0.5
        
        # if self.intention is None:
        #     # during action perception
        #     bias = [0.2, 0.6, 0.6, 0.8, 0.8, 0.8]
        # else:
        #     # during action production
        #     bias = [0.2, 0.2, 0.4, 0.6, 0.8, 0.8]
        
        # bias = [0.8] * 6  # same bias

        # general intention-dependent bias 
        if self.intention is None:
            bias = 0.7
        else:
            bias = 0.3

        self.K = kalman_gain(self.free_energy, self.PE.precision, bias, gain_gain=0.66)
        
        # if self.name == "Goals":
        #     self.K = kalman_gain(self.free_energy, self.PE.precision, bias[0], gain_gain=gain_gain)
        # elif self.name == "Realizations":
        #     self.K = kalman_gain(self.free_energy, self.PE.precision, bias[1], gain_gain=gain_gain)
        # elif self.name == "Schm":
        #     self.K = kalman_gain(self.free_energy, self.PE.precision, bias[2], gain_gain=gain_gain)
        # elif self.name == "Seq":
        #     self.K = kalman_gain(self.free_energy, self.PE.precision, bias[3], gain_gain=gain_gain)
        # elif self.name == "Vision":
        #     self.K = kalman_gain(self.free_energy, self.PE.precision, bias[4], gain_gain=gain_gain)
        # elif self.name == "MC":
        #     self.K = kalman_gain(self.free_energy, self.PE.precision, bias[5], gain_gain=gain_gain)



    def set_intention_dependent_kalman_gain(self):
        """ fixate K depending on intention setting.
        belief' = td_posterior + K * (bu_posterior - td_posterior)
        """
        if "bias_gain" in self.params:
            gain_bias = self.params['bias_gain']
        else:
            gain_bias = 0.5

        if self.intention is None:
            # during PERCEPTION
            self.K = kalman_gain(self.free_energy, self.PE.precision, 1 - gain_bias)

            # if self.name in ["Realizations", "Schm", "Seq"]:
            #     # increase Kalman Gain to make these layer susceptible to evidence
            #     self.K = gain_bias # 0.9
            # elif self.name in ["Goals", "MC", "Vision"]:
            #     # decrease Kalman Gain for these layers
            #     self.K = 1 - gain_bias # 0.1
            if self.name == "Vision":
                self.K = 1.0
        elif self.intention is not None:
            # during PRODUCTION

            self.K = kalman_gain(self.free_energy, self.PE.precision, gain_bias)

            # if self.name in ["Goals", "Schm", "Seq"]:
            #     # decrease Kalman Gain for these layers to be less susceptible to evidence
            #     self.K = 1 - gain_bias # 0.1
            # elif self.name in ["Realizations", "Vision", "MC"]:
            #     # increase Kalman Gain for these layers
            #     self.K = gain_bias # 0.9

        
        self.log(3, "K =", self.K, "for", self.name)



    def prediction(self):
        """ Sample new prediction based on updated state.
        Return prediction for sending.
        """
        raise NotImplementedError("Should have implemented this")



    def integrate_evidence(self):
        """ Receive and integrate the evidence with previous evidence.
        """
        raise NotImplementedError("Should have implemented this")



    def td_inference(self):
        """ Receive and integrate higher layer and long range projections.
        """
        raise NotImplementedError("Should have implemented this")


    def bu_inference(self):
        """ Infer new bottom-up posterior based on prior and evidence.
        """
        raise NotImplementedError("Should have implemented this")



    def extension(self):
        """ Decide on and do hypothesis extension and
        decide on evidence for next higher layer.
        """
        raise NotImplementedError("Should have implemented this")



    def receive_prediction(self, prediction):
        """ Receive prediction from next higher layer.
        """
        self.higher_layer_prediction = prediction[0]
        self.higher_layer_K = prediction[1]



    def receive_evidence(self, evidence):
        """ Receive evidence from next lower layer.
        """
        self.lower_layer_evidence = evidence[0]
        self.lower_layer_new_hypo = evidence[1]
        # self.lower_layer_K = evidence[2]



    def receive_prunable(self, prunable):
        """ Receive prunable representation IDs from next higher layer.
        """
        self.higher_layer_pruning = prunable



    def receive_long_range_projection(self, projection):
        """ Receive long range projections from other sources.
        """
        self.long_range_projection = projection



    def receive_lower_level_hypos(self, ll_hypos):
        """ Inform level about lower level's hypotheses
        """
        self.lower_layer_hypos = ll_hypos


    def send_level_hypos(self):
        """ Send level's hypos to next higher level
        """
        return self.hypotheses



    def send_prediction(self):
        """ Send prediction for next lower layer.
        """
        return [self.layer_prediction, self.K]



    def send_evidence(self):
        """ Send evidence for next higher layer.
        """
        return [self.layer_evidence, self.layer_new_hypo]



    def send_prunable(self):
        """ Send IDs of prunable representations for next lower layer.
        """
        return self.layer_pruning



    def send_long_range_projection(self):
        """ Send long range projection to another layer by name.
        """
        return self.layer_long_range_projection



    def clean_up(self):
        """ Clean up used object attributes after hierarchy update.
        """
        # remote references
        self.higher_layer_pruning = None
        self.higher_layer_prediction = None
        self.higher_layer_K = None
        self.long_range_projection = None
        self.lower_layer_new_hypo = None
        self.lower_layer_evidence = None

        # local attributes
        # if self.params["self_supervised"]:
        self.bu_posterior = None  
        self.td_posterior = None  
        self.likelihood = None
        # self.best_hypo = None
        self.lower_layer_hypos = None
        self.layer_evidence = None
        self.layer_LH = None  # otherwise updated dependend on word length won't work
        self.layer_prediction = None
        self.layer_new_hypo = None
        self.layer_pruning = None
        self.layer_long_range_projection = None

