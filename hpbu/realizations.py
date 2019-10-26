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

""" GoalSequences
Created on 12.03.2018

@author: skahl
"""
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from .sequencelayer import *
from .ment import *


class Realizations(SequenceLayer):


    def __init__(self, name):
        super(Realizations, self).__init__(name)
        self.type = 'Sequence'
        self.reset()



    def reset(self):
        super(Realizations, self).reset()
        self.last_lower_layer_evidence = None

        # temporarily store the most probable realization
        self.intention_prediction = None
        # initialize wait signal for after own productions to wait for other agent's
        self.should_wait = False
        self.action_done = True
        self.surprise_received = False
        # store lower layer sense of agency estimate
        self.self_estimate = 0.
        self.max_lower_level_hypo = None
        # store currently available info about own and other's mental states
        self.current_mentalstates = {}
        self.current_mentalstate = None
        self.personmodel = None
        # store temporary realization
        self.tmp_realization = []  # will be extended with a mental state if one was used for a decision 

        print("(Re-)initialized layer", self.name)



    def __str__(self):
        return "Layer '" + self.name + "', type:" + self.type



    def print_out(self):
        _str_ = self.name
        _str_ += "\nMentalStates:\n"
        _str_ += str(self.current_mentalstates)
        _str_ += "\nhypotheses ("
        _str_ += str(len(self.hypotheses))
        _str_ += "):\n"
        _str_ += str(self.hypotheses)
        return _str_




    def integrate_evidence(self):
        """ Integrate evidence from next lower layer.
        Here, update the stored core PersonModel with lower layer argmax hypothesis IDs, which
        identify actions coming from self (me) or other (you).
        In that, the mapping from PersonModel beliefs to MentalStates in Realizations is done.

        Given identified (possibly mutiple) You, update information in the context of possible
        social interactions. If no You is available, no social interaction should be possible.
        """

        self.log(4, "new evidence:", self.lower_layer_evidence)
        updated_tmp_relization = False
        action_done = False
        self_estimate = self.self_estimate
        max_hypo = None

        # only integrate information and belief states if there is another agent present
        if len(self.personmodel.agents) > 0:

            # check if intention_prediction exists
            if self.intention_prediction is None:
                self.intention_prediction = {"idx": 0, "realization": None}

            if self.lower_layer_evidence is not None and self.lower_layer_evidence[0] is not None and self.surprise_received:
                # self.surprise_received

                self.last_lower_layer_evidence = copy(self.lower_layer_evidence)  # copy for later use
                self.lower_layer_hypos = self.last_lower_layer_evidence[0]
                max_hypo_P, max_hypo = self.lower_layer_hypos.max()  # tuple of (P, id)

                self.time_since_evidence += self.last_lower_layer_evidence[1]
                self_estimate = self.last_lower_layer_evidence[3]

                # TODO: check if it is possible that the perceived max lh is similar to intention
                # compare perceived you belief to all cluster prototypes:
                # mean_P = np_mean(self.lower_layer_hypos.dpd[:, 0])
                # self.lower_layer_hypos.dpd[cluster.dpd_idx, 0] > mean_P and

                if self.current_mentalstate is not None and self.current_mentalstate.me != []:
                    # TODO: if clusters don't change this could be calculated once in a lookup table
                    # check what other clusters could be expected, given the current me belief as the communication goal
                    me_cluster = self.current_mentalstate.me[0]
                    if me_cluster != max_hypo:
                        # get the top 20% of likely clusters
                        ar_likely_clusters = np.array(self.lower_layer_hypos.dpd)
                        nth_part = int(ar_likely_clusters.shape[0] * 0.2)+1
                        sorted_likely_clusters = ar_likely_clusters[ar_likely_clusters[:, 0].argsort()]
                        top_percent_clusters = sorted_likely_clusters[ar_likely_clusters.shape[0] - nth_part:]
                        self.log(3, "index of the percentile split:", nth_part)

                        self.log(1, "Top 20 percent of other likely clusters:", top_percent_clusters)
                        if me_cluster in top_percent_clusters[:, 1]:
                            self.log(1, "Perceived cluster is closely similar to intended cluster! Assuming similarity between:", me_cluster, max_hypo)
                            max_hypo = me_cluster
                            max_hypo_P = self.lower_layer_hypos.dpd[self.lower_layer_hypos.reps[me_cluster].dpd_idx, 0]
                        else:
                            self.log(1, "Perceived cluster is not similar enough to intended cluster!")
                        # self.log(0, self.lower_layer_hypos.dpd)


                # print("received self_estimate:", self_estimate)#
                self.log(1, "max hypo:", max_hypo, max_hypo_P)
                self.log(1, "self estimate:", self_estimate, "action finished:", action_done, "intention:", self.intention)

                if self.action_done and self.surprise_received and self.intention is not None:
                    self_estimate = self.self_estimate
                    # max_hypo = self.max_lower_level_hypo
                    action_done = True
                else:
                    action_done = False

                # update current mental state in PersonModel
                if self.intention in [Intention("_you", True), Intention("_you", False), Intention("_me", True), Intention("_me", False)] and action_done and max_hypo is not None:
                    # there was an intention to act, intent was acted upon
                    # am I still sure that it was me who acted?
                    self.log(1, "successfully acted upon action intention, update realization likelihoods")

                    # ! ! ! there is a bias towards self-hood, so to make sure bias is 0.1
                    # you:
                    if self_estimate < 0.4:
                        # update current mentalstate
                        self.current_mentalstate.you = [max_hypo]
                    # me:
                    else:
                        self.current_mentalstate.me = [max_hypo]
                    # me: no change necessary
                    self.log(1, "current mental state:", self.current_mentalstate)

                    # extend the tmp realization with the successfully acted upon intention
                    self.tmp_realization.append( copy(self.intention) )
                    updated_tmp_relization = True
                    # move to next substate of most probable realization
                    self.intention_prediction["idx"] += 1

                    # enough evidence collected, reset intention, as action was done
                    self.intention = None
                    self.self_estimate = 0.
                    # self.action_done = True

                elif self.intention in [None, Intention("_wait", False), Intention("_thumbsup", False)] and self.should_wait and self.surprise_received:
                    # we are actively perceiving an action
                    # there is no intention to act, but maybe this was still me?
                    self.log(1, "perceived action understanding is stable, update realization likelihoods")
                    # self.log(1, "after perceived action, wait one second...")
                    # self.hierarchy_sleep(1)

                    # ! ! ! there is a bias towards other-hood, so to make sure bias is 0.1
                    # you:
                    if self_estimate < 0.4:
                        # update current mentalstate
                        self.current_mentalstate.you = [max_hypo]
                    # me:
                    else:
                        self.current_mentalstate.me = [max_hypo]

                    self.log(1, "current mental state:", self.current_mentalstate)
                    if self.intention == Intention("_thumbsup", False):
                        self.log(1, "Didn't receive the _thumbsup signal, but an action was performed instead!")

                    # extend the tmp realization with the "waiting" that was done here
                    self.tmp_realization.append( Intention("_wait", False) )
                    updated_tmp_relization = True
                    # move to next substate of most probable realization
                    self.intention_prediction["idx"] += 1

                    # waited long enough, reset waiting variable
                    self.should_wait = False
                    self.surprise_received = False
                    # reset intention, as action was perceived or thumbsup done
                    self.intention = None

                self.surprise_received = False

            # check for just received meta-communicative signals 
            if self.personmodel.me_focus is not None:
                focus_meta_signals = self.personmodel.meta_communication.agents_meta[self.personmodel.me_focus]

                if self.intention in [Intention("_thumbsup", False), Intention("_gaze", False)] and len(focus_meta_signals) > 0:

                    self.log(3, "checking focused agent for meta-signals:", self.personmodel.me_focus, focus_meta_signals)
                    # self.personmodel.meta_communication.set_information_received()

                    if self.intention == Intention("_thumbsup"):
                        if "_thumbsup" in focus_meta_signals:
                            self.log(1, "Received the mutual _thumbsup signal that I was waiting for!")
                            self.tmp_realization.append( Intention("_thumbsup", False) )
                            updated_tmp_relization = True
                            self.intention_prediction["idx"] += 1
                            self.should_wait = False
                            # reset intention, thumbsup done
                            self.intention = None
                        else:
                            self.log(4, "Didn't receive the _thumbsup signal, instead received:", focus_meta_signals)

                    if self.intention == Intention("_gaze"):
                        if "_gaze" in focus_meta_signals:
                            self.log(1, "Received the mutual _gaze signal that I was waiting for!")
                            self.tmp_realization.append( Intention("_gaze", False) )
                            updated_tmp_relization = True
                            self.intention_prediction["idx"] += 1
                            self.should_wait = False
                            # reset intention, gaze done
                            self.intention = None

                            # if some time has passed, in multi agent scenarios, just to make sure, gaze back again
                            # self.personmodel.meta_communication.gaze_at(self.personmodel.me_focus)
                        else:
                            self.log(4, "Didn't receive the _gaze signal, instead received:", focus_meta_signals)

            # A realization sequence always starts with a MentalState and a current MentalState should be available
            if self.current_mentalstate is not None:
                if self.intention_prediction is not None and self.intention_prediction["realization"] is not None:
                    cur_idx = self.intention_prediction["idx"]
                    len_substates = len(self.intention_prediction["realization"].substates)
                    
                    # add current_mentalstate if 
                    if cur_idx < len_substates and\
                        type(self.tmp_realization[-1]) != MentalState and\
                        type(self.intention_prediction["realization"].substates[cur_idx]) == MentalState:
                        # if next predicted substate is a MentalState
                        # add a generalized form of the current mentalstate to tmp realization
                        generalized_mentalstate = self.generalize_current_mentalstate()
                        self.tmp_realization.append( generalized_mentalstate )
                        updated_tmp_relization = True

                        # advance the realization index, since mental states cannot intent actions
                        self.intention_prediction["idx"] += 1

                        self.log(1, "Matching coordination sequence:", self.tmp_realization)

                # update likelihood of realizations, given current temporary realization
                if updated_tmp_relization:
                    self.likelihood, best_hypo, best_idx = self.update_realizations(self.hypotheses.reps, self.tmp_realization)

            if self.likelihood is not None:
                self.log(3, "Coordination sequence likelihood:\n", self.likelihood)



    def td_inference(self):
        """ Receive and integrate higher layer and long range projections.
        """
        if self.long_range_projection is not None:
            self.log(1, "long range projection:", self.long_range_projection)
            if "intention" in self.long_range_projection:
                lrp = self.long_range_projection["intention"]

                # only generate intentions for coordination if there is at least one other agent present
                if lrp is not None and len(self.personmodel.agents) > 0:
                    # if other agents are available, switch to next best agent for interaction and update current_mentalstate
                    self.current_mentalstates = {agent_id: MentalState(me=[lrp]) for agent_id in self.personmodel.agents.keys()}

                    # bootstrapped interaction with Agent A, if you are in a follower role
                    if lrp == "follow":
                        self.log(1, "\n\tInitializing follower role")
                        if "Agent A" in self.current_mentalstates:
                            self.personmodel.set_focus("Agent A")
                            self.current_mentalstate = self.current_mentalstates["Agent A"]
                            self.current_mentalstate = MentalState()
                        else:
                            self.error("Agent A is not present")
                            sys.exit(1)
                    else:
                        self.log(1, "\n\tInitializing leader role")
                        # start interaction with any agent that you have no understanding with
                        for agent_id, MS in self.current_mentalstates.items():
                            if MS.we != MS.me and agent_id != self.personmodel.my_id:
                                self.personmodel.set_focus(agent_id)
                                self.current_mentalstate = MS
                                break

                        self.current_mentalstate = MentalState(me=[lrp])
                        self.personmodel["me"] = [lrp]
                    
                    generalized_mentalstate = self.generalize_current_mentalstate()
                    # remember the new intended mentalstate goal for this interaction
                    self.tmp_realization = [generalized_mentalstate]
                    # check if intention_prediction exists
                    if self.intention_prediction is None:
                        self.intention_prediction = {"idx": 0, "realization": None}
                    # advance the realization index, since mental states cannot intent actions
                    self.intention_prediction["idx"] += 1

                    # calculate rule match for PersonModel, disregarding timing
                    self.likelihood, best_hypo, best_idx = self.update_realizations(self.hypotheses.reps, self.tmp_realization)
                    self.td_posterior = norm_dist(self.likelihood)

                    self.log(3, "Comparing known to temporary coordination sequence", self.tmp_realization)
                    self.log(3, "posterior:", self.td_posterior)

            if "surprise" in self.long_range_projection:
                self.log(1, "Received delay-surprise signal from level:", self.long_range_projection["surprise"])
                self.surprise_received = True

            # receive information about an intended action's completion
            if "done" in self.long_range_projection:
                # self.max_lower_level_hypo = self.intention
                self.self_estimate = self.long_range_projection["done"]  # self.personmodel.self_estimate
                self.action_done = True
                self.surprise_received = True

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

            self.log(4, "updated top-down posterior from higher layer:\n", self.td_posterior)



    def bu_inference(self):
        """ Calculate the posterior for the sequence layer, based on evidence from
        predicted lower level activity.
        """
        if self.hypotheses.dpd is not None and len(self.hypotheses.dpd) > 0:
            if self.likelihood is not None:
                self.log(4, "evidence:", self.likelihood)
                self.bu_posterior = posterior(self.hypotheses.dpd, self.likelihood, smooth=True)
                self.log(3, "updated bottom-up posterior:\n", self.bu_posterior)



    def update_realizations(self, realizations_dict, tmp_realization):
        """ Compare a dictionary of realizations with a new realization and the current mental state.
        """
        new_seq = np.array(tmp_realization)

        # take new sequence and compare with other realizations (realizations level comparison)
        realizations = np.array([[seq, key_id] for key_id, seq in realizations_dict.items()])
        diffs_LH = np.zeros((realizations.shape[0], 2))
        diffs_LH[:, 1] = realizations[:, 1]  # create result structure "diffs_LH"

        # create temporary sequence discretization
        for idx, seq in enumerate(realizations):
            # the better fit will still win
            s_shape = len(seq[0].substates)
            tmp_shape = len(new_seq)

            # self.log(1, "tmp:", new_seq, "\n\tknown:", seq[0].substates)
            # check if the size of the realizations are sufficient
            if tmp_shape <= s_shape:
                # weight diff with respect to the sequence length's fraction of the sequence compared to
                # scale_to_len = tmp_shape / s_shape
                _diff_lh = diff_levenshtein(new_seq, seq[0].substates) # * scale_to_len

            else:
                # if probability distributions are too small, rate minimal comparability
                # actually, this shouldn't happen
                _diff_lh = 0.0001

            diffs_LH[idx, 0] = _diff_lh

        diffs_LH = diffs_LH[diffs_LH[:, 1].argsort()]  # in place sorting
        idx_max_P = np_argmax(diffs_LH[:, 0])  # so max is BEST

        return diffs_LH, realizations_dict[diffs_LH[idx_max_P, 1]], idx_max_P



    def generalize_current_mentalstate(self):
        """ To make the current mental state comparable,
        content has to be converted to more general variables.
        """

        # TODO: ignoring WE for now

        _me = self.current_mentalstate.me[0] if len(self.current_mentalstate.me) > 0 else None
        _you = self.current_mentalstate.you[0]if len(self.current_mentalstate.you) > 0 else None
        # _we = self.current_mentalstate.we[0]if len(self.current_mentalstate.we) > 0 else None

        generalized_mentalstate = MentalState()
        generalized_mentalstate.me = [Belief("_a")] if _me is not None else [Belief(None)]

        if _me == _you and _me is not None:
            generalized_mentalstate.you = [Belief("_a")]
        elif _me is None and _you is not None:
            generalized_mentalstate.you = [Belief("_a")]
        elif _you is not None:
            generalized_mentalstate.you = [Belief("_b")]
        else:
            generalized_mentalstate.you = [Belief(None)]

        # if _we == _me and _me is not None:
        #     generalized_mentalstate.we = [Belief("_a")]
        # elif _we == _you and _you is not None:
        #     generalized_mentalstate.we = [Belief("_b")]
        # elif _we is not None:
        #     generalized_mentalstate.we = [Belief("_c")]
        # else:
        #     generalized_mentalstate.we = [Belief(None)]

        return generalized_mentalstate



    def extension(self):
        """ Decide on and do hypothesis extension and
        decide on evidence for next higher layer.
        """

        if self.likelihood is not None:
            self.layer_evidence = [self.hypotheses, self.time_since_evidence, self.self_estimate]



    def prediction(self):
        """ Decide on predicted next lower layer activity in best predicted sequence.
        """

        # print("likelihood not None?:", self.likelihood is not None, "not should_wait?:", not self.should_wait, "ll hypos not None?:", self.lower_layer_hypos is not None)
        # only predict a next step in social interaction, if there is another agent present
        if len(self.personmodel.agents) > 0 and self.current_mentalstate is not None and self.lower_layer_hypos is not None:
            #  and self.lower_layer_hypos is not None and len(self.lower_layer_hypos) > 0
            
            if self.intention is None and not self.should_wait and self.action_done:
                # update self.intention_prediction
                if self.intention_prediction is None:
                    self.intention_prediction = {"idx": 0, "realization": None}
                max_id = self.hypotheses.max()[1]
                self.intention_prediction["realization"] = self.hypotheses.reps[max_id]  # is this different than an available intention?

                if self.current_mentalstate is not None:
                    generalized_MS = self.generalize_current_mentalstate()

                if self.intention_prediction["idx"] < len(self.intention_prediction["realization"].substates) - 1:
                    next_intention_variable = self.intention_prediction["realization"].substates[self.intention_prediction["idx"]]
                    self.log(1, "Next intention variable from Realization", max_id, ":", next_intention_variable)

                    # the next step in the realization will be passed on to lower levels
                    if type(next_intention_variable) == MentalState:
                        self.log(1, "Something has gone terribly wrong: MentalState shouldn't be a next realization step")
                        self.intention = None
                        self.layer_prediction = None
                        sys.exit(2)
                    else:
                        
                        if Intention("_gaze", False) == next_intention_variable:
                            # send the gaze signal to the currently focused present agent
                            if self.personmodel.me_focus is not None:
                                self.log(1, "Now setting gaze at agent:", self.personmodel.me_focus)
                                self.personmodel.meta_communication.gaze_at(self.personmodel.me_focus)

                                # implicitly we also wait for mutual gaze on a sent gaze signal
                                self.should_wait = True
                                self.intention = next_intention_variable
                            else:
                                self.error("me_focus is not set! Cannot gaze at unknown agent!")

                        elif Intention("_thumbsup", False) == next_intention_variable:
                            # send the thumbsup signal to the currently focused present agent
                            if self.personmodel.me_focus is not None:
                                self.log(1, "Now signaling thumbsup to agent:", self.personmodel.me_focus)
                                self.personmodel.meta_communication.thumbsup_at(self.personmodel.me_focus)

                                # implicitly we also wait for the return thumbsup
                                self.should_wait = True
                                self.intention = next_intention_variable

                                # clear distracting influences and signal lower levels to just observe 
                                # and possibly reset their influencing factors
                                self.lower_layer_hypos.equalize()
                                self.layer_prediction = self.lower_layer_hypos.dpd
                                self.layer_long_range_projection = {
                                                                        "Schm":
                                                                        {
                                                                            "observe": None
                                                                        },
                                                                        "Seq":
                                                                        {
                                                                            "observe": None
                                                                        }
                                                                    }
                            else:
                                self.error("me_focus is not set! Cannot gaze at unknown agent!")

                        elif Intention("_wait", False) == next_intention_variable:
                            # configure this level to wait for a stable lower level hypothesis
                            self.log(1, "Now waiting for a stable percept...")
                            self.should_wait = True
                            self.intention = next_intention_variable

                            # clear distracting influences and signal lower levels to just observe 
                            # and possibly reset their influencing factors
                            # self.lower_layer_hypos.equalize()
                            self.layer_prediction = self.lower_layer_hypos.dpd
                            # self.layer_long_range_projection = {
                            #                                         "Schm":
                            #                                         {
                            #                                             "observe": None
                            #                                         },
                            #                                         "Seq":
                            #                                         {
                            #                                             "observe": None
                            #                                         }
                            #                                     }

                        elif self.lower_layer_hypos is not None and len(self.lower_layer_hypos) > 0:
                            if self.current_mentalstate.me != [] and\
                                    (next_intention_variable == Intention("_me", False) or next_intention_variable == Intention("_me", True)):
                                # decode generalized intent to current MentalState
                                next_intention_id = self.current_mentalstate.me[0]
                            elif self.current_mentalstate.you != [] and\
                                    (next_intention_variable == Intention("_you", False) or next_intention_variable == Intention("_you", True)):
                                # decode generalized intent to current MentalState
                                next_intention_id = self.current_mentalstate.you[0]
                            else:
                                self.log(1, "No actionable motor belief available!")
                                next_intention_id = None

                            # print("next intention id:", next_intention_id)
                            if next_intention_id is not None:
                                # make a prediction for the complete lower level distribution
                                
                                # clear distracting influences
                                self.lower_layer_hypos.equalize()

                                avg_P = np_mean(self.hypotheses.dpd[:, 0])
                                var_P = 0.2
                                critical_intention = avg_P + var_P  # only +1 var

                                lower_layer_dpd = copy(self.lower_layer_hypos.dpd)
                                intention_idx = self.lower_layer_hypos.reps[next_intention_id].dpd_idx
                                tmp_layer_prediction = set_hypothesis_P(lower_layer_dpd, intention_idx, critical_intention)  # TODO: criticality?
                                self.layer_prediction = norm_dist(tmp_layer_prediction)

                                # intention for 'me' to produce entails an actual intention signal:
                                # in addition and to let intention percolate down the hierarchy, send a LRP
                                self.layer_long_range_projection = {"Schm":
                                                                    {
                                                                        "intention": next_intention_id,
                                                                        "signaling_distractor": self.current_mentalstate.you[0] if next_intention_variable.signaling else None
                                                                    }}

                                self.intention = next_intention_variable  # the full Intention object
                                self.action_done = False
                            else:
                                # if no intention prediction available, just commit to the available information
                                self.layer_prediction = self.lower_layer_hypos.dpd

                elif self.intention_prediction["realization"].goal == generalized_MS:
                    self.log(1, "My own realization is supposedly finished with", self.personmodel.me_focus)  #, "\nSending thumbsup signal...")
                    self.intention_prediction = None  # {"realization": None, "idx": 0}
                    self.tmp_realization = []
                    self.current_mentalstate.we = copy(self.current_mentalstate.me)
                    self.current_mentalstates[self.personmodel.me_focus] = copy(self.current_mentalstate)

                    lrp = copy(self.current_mentalstate.me[0])
                    self.current_mentalstate = None

                    # # after established common-ground, send the finalizing thumbsup
                    # self.personmodel.meta_communication.thumbsup_at(self.personmodel.me_focus)

                    # if other agents are available, switch to other agent for interaction and update current_mentalstate
                    # check for role being follower or leader. followers don't switch!
                    if self.personmodel.me_focus != "Agent A":
                        for agent_id, MS in self.current_mentalstates.items():
                            if agent_id != self.personmodel.me_focus:
                                if MS.we != MS.me:
                                    self.personmodel.set_focus(agent_id)
                                    self.current_mentalstate = MS
                                    self.log(1, "Starting interaction with:", agent_id)
                                    break
                                else:
                                    self.personmodel.me_focus = None
                                    self.current_mentalstate = None

                        if self.current_mentalstate is not None:
                            self.current_mentalstate.me = [lrp]
                            self.personmodel["me"] = [lrp]
                        
                            generalized_mentalstate = self.generalize_current_mentalstate()
                            # remember the new intended mentalstate goal for this interaction
                            self.tmp_realization = [generalized_mentalstate]
                            # check if intention_prediction exists
                            if self.intention_prediction is None:
                                self.intention_prediction = {"idx": 0, "realization": None}
                            # advance the realization index, since mental states cannot intent actions
                            self.intention_prediction["idx"] += 1

                            # self.log(1, "Comparing known to temporary coordination sequence:", self.tmp_realization)
                            # # calculate rule match for PersonModel, disregarding timing
                            # self.likelihood, best_hypo, best_idx = self.update_realizations(self.hypotheses.reps, self.tmp_realization)
                            # self.td_posterior = norm_dist(self.likelihood)

                    # to end the interaction, send a "done" signal to the Goals layer
                    if self.current_mentalstate is None:
                        self.log(1, "\n\nAll interaction done for me!")
                        self.layer_long_range_projection = {"Goals": {"done", True}}

            elif self.lower_layer_hypos is not None:
                # if no intention prediction available, just commit to the available information
                self.layer_prediction = self.lower_layer_hypos.dpd


