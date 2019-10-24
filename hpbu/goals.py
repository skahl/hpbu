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

""" DesiredSocialStates
Created on 12.03.2018

@author: skahl
"""
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals


from .clusterlayer import *
from .ment import *


class Goals(ClusterLayer):


    def __init__(self, name):
        super(Goals, self).__init__(name)
        self.type = 'Goals'
        self.reset()



    def reset(self):
        super(Goals, self).reset()
        self.layer_LH = {}  # cluster layer likelihood P(C|S)

        # store last temporal delta between surprisal events
        self.delta_t = 0.
        # store last surprisal time
        self.last_surprisal_time = None
        print("(Re-)initialized layer", self.name)



    def __str__(self):
        return "Layer '" + self.name + "', type:" + self.type


    def finalize(self):
        """ finalize gets called upon the end of the hierarchy main loop
        So for clustering during training, re-cluster at the end of the training session.
        """
        pass



    def print_out(self):
        _str_ = self.name
        _str_ += "\nhypotheses ("
        _str_ += str(len(self.hypotheses))
        _str_ += "):\n"
        _str_ += str(self.hypotheses)
        return _str_



    def integrate_evidence(self):
        """ Integrate evidence from next lower layer.
        Here, only prepare for new posterior inference if new hypothesis was added in next lower layer
        and calculate likelihood matrix P(C|S)
        """

        if self.lower_layer_evidence is not None:
            self.log(4, "received lower layer evidence")

            # store next lower layer sense of agency estimate
            self.self_estimate = self.lower_layer_evidence[2]

            # add timing information if there is any
            if self.lower_layer_evidence[1] is not None:
                self.last_surprise_time += self.lower_layer_evidence[1]
                self.last_production_time += self.lower_layer_evidence[1]

        # construct sparse LH matrix from part-of relationships with space for new sequence hypothesis
        self.likelihood = self.layer_LH = defaultdict(list)
        for cl_id, rep in self.hypotheses.reps.items():
            for rule_id in rep.realizations:
                self.layer_LH[cl_id].append(rule_id)


        if self.lower_layer_new_hypo is None and len(self.lost_sequences) > 0:
            self.lower_layer_new_hypo = self.lost_sequences.pop()
            self.log(1, "Adding lost sequence as lower_layer_new_hypo:", self.lower_layer_new_hypo['id'])



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
                avg_P = np_mean(self.hypotheses.dpd[:, 0])
                var_P = np_var(self.hypotheses.dpd[:, 0])
                critical_intention = avg_P + var_P  # only +1 var

                # print("pre influence id:", lrp, self.hypotheses.dpd[self.hypotheses.reps[lrp].dpd_idx])

                # idx from rep id
                self.td_posterior = np.zeros((self.hypotheses.dpd.shape[0], 2))  # copy(self.hypotheses.dpd)
                self.td_posterior[:, 1] = self.hypotheses.dpd[:, 1]
                self.td_posterior = set_hypothesis_P(self.td_posterior, lrp_idx, critical_intention)
                self.td_posterior = norm_dist(self.td_posterior)
                self.intention = lrp
                # print("projected id:", lrp, self.td_posterior[lrp_idx])

            # if successful intention production was communicated via long range projection:
            if "done" in self.long_range_projection:
                self.log(1, "intention production of state-goal pair", self.intention, "was finished!")
                self.intention = None

        if self.higher_layer_prediction is not None:
            self.log(4, "higher layer projection:", self.higher_layer_prediction)
            higher_layer = copy(self.higher_layer_prediction)

            self.td_posterior = posterior(self.hypotheses.dpd, higher_layer, smooth=True)

            self.log(4, "updated top-down posterior from higher layer:\n", self.td_posterior)



    def extension(self):
        pass



    def prediction(self):
        """ STATE NODE: Prepare prediction.
        """
        if len(self.hypotheses.reps) > 1 and self.layer_LH is not None and len(self.layer_LH) > 0:

            self.layer_prediction = [self.hypotheses.dpd, self.layer_LH]
            # print("hypos:", self.hypotheses.dpd, "goals matrix:", self.layer_LH)
            

