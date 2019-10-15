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

""" MENT module for parsing and storing xml mental state rules
Created on 06.03.2018

@author: skahl
"""

# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals


from .representations import *
from collections import defaultdict


class Goals(Representation):
    def __init__(self, idx, comment, state=None, realizations=None, goal=None):
        super(Goals, self).__init__(idx)
        self.comment = comment
        self.state = state
        self.goal = goal
        self.realizations = realizations


    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        mystr = "\nGoal ID: " + str(self.id) + "\n"
        mystr += "state:\n" + str(self.state) + "\n"
        mystr += "goal:\n" + str(self.goal) + "\n"
        mystr += "realizations: " + str(self.realizations) + "\n"
        return mystr





class Belief(object):
    """ Representation of a belief in a MentalState, used to decide if
    one belief is different from another, or the same.
    """
    def __init__(self, var):
        self.var = var


    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        return "belief:"+str(self.var)


    def __eq__(self, other):
        if type(other) == Belief:
            return self.var == other.var
        else:
            return False



class Intention(object):
    """ Representation of an intention, used for having a overloaded function for equality check.
    """

    def __init__(self, intent, signaling=False):
        """ set the initial intention string or ID string
        """
        self.intent = intent
        self.signaling = signaling


    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        return "intent:"+str(self.intent)+"|signaling:"+str(self.signaling)


    def __eq__(self, other_intent):
        if type(other_intent) == Intention:
            # print(self, other_intent, "eq:", self.intent == other_intent.intent)
            return self.intent == other_intent.intent and self.signaling == other_intent.signaling
        else:
             #print(other_intent, "is no Intention")
            return False




class MentalState(object):
    """ Store beliefs for comparison
    """

    def __init__(self, me=[], you=[], we=[]):
        """ beliefs to compare
        """
        self.me = me
        self.you = you
        self.we = we


    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        mystr = ""
        if self.me != []:
            mystr += "me: "+str(self.me)+" "
        if self.you != []:
            mystr += "you: "+str(self.you)+" "
        if self.we != []:
            mystr += "we: "+str(self.we)+" "
        return mystr


    def __eq__(self, mentalstate):
        if type(mentalstate) == MentalState:
            # belief comparison of fit of mentalstate given this mentalstate
            if self.me == mentalstate.me and\
                self.you == mentalstate.you:
                
                # TODO: ignoring WE for now:  and self.we == mentalstate.we:

                # print("self:", self, "ms:", mentalstate, "TRUE")
                return True
            else:
                # print("self:", self, "ms:", mentalstate, "FALSE")
                return False
        else:
            # print(mentalstate, "is no MentalState")
            return False



class PersonModel(dict):
    """ Store mental states for me, you and we personmodels
    """
    def __init__(self, me=None, you=None, we=None, agents=None):

        self.meta_communication = None
        self.my_id = None

        self.me_focus = None
        self.interaction_influence_mode = None
        self.sequence_index_mapping = None
        self.self_estimate = 0.

        self.me = me
        if me is None:
            self.me = []
        self.you = you
        if you is None:
            self.you = defaultdict(list)
        self.we = we
        if we is None:
            self.we = []
        self.agents = agents
        if agents is None:
            self.agents = {}

        # if type(self.you) == defaultdict:
        #     self.you_dpd = np.array([[1/len(self.you), you_id] for you_id in self.you.keys() if you_id in self.agents.keys()])
        # else:
        #     self.you_dpd = None if len(self.you) == 0 else np.array([1.])


    def set_focus(self, agent_id):
        """ Set the PersonModel focus variable and
        adapt the influence distribution, given the current interaction influence mode.
        """
        if self.interaction_influence_mode is not None:
            if agent_id in self.agents:
                self.me_focus = agent_id
            else:
                print("PersonModel Error: focus id is not found in list of present agents.")
                sys.exit(1)

            # set influence distribution, given current focus and mode
            if self.interaction_influence_mode == "focus_only":
                # set a very focused influence gain on the current focus
                self.set_agent_influence(self.me_focus, 1.)
            elif self.interaction_influence_mode == "equal":
                # equally distribution influence gain
                ratio_per = 1 / len(self.agents)
                all_influence_dict = {a_id:ratio_per for a_id in self.agents.keys()}
                self.set_all_agents_influence(all_influence_dict)
            elif self.interaction_influence_mode == "balanced":
                # set a strong influence gain on strong focus, and distribute the rest
                self.set_agent_influence(self.me_focus, 0.66)
            else:
                print("PersonModel Warning: Interaction influence mode was not set correctly!")
        else:
            print("PersonModel Error: Interaction influence mode parameter was not set!")
            sys.exit(1)


    def inform_sequence_mapping(self, reps):
        self.sequence_index_mapping = reps


    def get_agent_influence_indices(self):
        if self.sequence_index_mapping is not None:
            you_indices = defaultdict(list)
            for you_id, ids in self.you.items():
                you_indices[you_id] = [self.sequence_index_mapping[seq_id].dpd_idx for seq_id in ids]

            return you_indices
        else:
            print("PersonModel Warning: Sequence_index_mapping wasn't set!")
            return None


    def set_agent_influence(self, agent_id, influence_ratio, max_cum_gain=1.):
        if agent_id in self.agents:
            if influence_ratio <= max_cum_gain:
                # set target agent influence
                self.agents[agent_id] = influence_ratio
                if len(self.agents) > 1:
                    # adapt other_agents influence if there are any
                    leftover_influence = max_cum_gain - influence_ratio
                    if leftover_influence > 0.:
                        per_agent_influence = leftover_influence / (len(self.agents) - 1)
                        for other_agent_id in self.agents.keys():
                            if agent_id != other_agent_id:
                                self.agents[other_agent_id] = per_agent_influence


    def set_all_agents_influences(self, agent_influence_dict, max_cum_gain=1.):
        combined_influence = 0.
        for agent_id, influence in agent_influence_dict.items():
            if agent_id in self.agents:
                self.agents[agent_id] = influence
                combined_influence += influence

        if combined_influence > max_cum_gain:
            print("PersonModel Warning: Combined influence of present agents is greater than 1!")


    def set_meta_communication(self, meta_communication):
        self.meta_communication = meta_communication


    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        mystr = "\t" + str(self.my_id) + "- personmodel:\n"
        if self.me is not None:
            mystr += "\t\tme: " + str(self.me) + "\n"
        if self.you is not None:
            mystr += "\t\tyou: " + str(self.you) + "\n"
            # mystr += "\t\tDPD: " + str(self.you_dpd) + "\n"
        if self.we is not None:
            mystr += "\t\twe: " + str(self.we) + "\n"
        if self.agents is not None:
            mystr += "\t\tpresent agents: " + str(self.agents) + "\n"
        return mystr


    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)


    def __setattr__(self, name, value):
        self[name] = value


    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)



class Realization(Representation):
    """ Store a realization of a goal-state sequence, consisting of intentions and mentalstate subgoals.
    """

    def __init__(self, idx, comment, state, goal, substates=[]):
        super(Realization, self).__init__(idx)
        self.comment = comment
        self.state = state
        self.goal = goal
        self.substates = substates


    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        mystr = "\nID: "+str(self.id)+"\t"+str(self.comment)+"\n"
        mystr += "State:\n"
        mystr += str(self.state)+"\n"
        mystr += "Substates:\n"
        mystr += str(self.substates)+"\n"
        mystr += "Goal:\n"
        mystr += str(self.goal)+"\n"

        return mystr




from collections import deque

class MetaCommunication(object):
    """ Represent available meta-communicative signals and their
    reception and sending. Specifically gaze- and confirmation are of interest here.
    An object has to receive a deque object that acts as a channel for communication.
    """

    def __init__(self, my_id, agents, comm_channel):

        if type(comm_channel) == deque:
            self.comm_channel = comm_channel  # set this deque object for communication
        else:
            print("MetaCommunication ERROR: comm_channel is no collections.deque object!")

        self.has_new_information = False
        self.my_id = my_id
        self.agents_available = agents
        self.cur_meta = ""
        self.cur_agent = ""
        self.agents_meta = {agent_id:[] for agent_id in agents}
        self.meta_signals = ["_gaze", "_thumbsup"]
        print("MetaCommunication was set up with the following agents:", self.agents_available)


    def gaze_at(self, agent):
        if agent in self.agents_available:
            self.cur_agent = agent
            self.cur_meta = "_gaze"

            # send gaze signal
            self._send_meta()


    def thumbsup_at(self, agent):
        if agent in self.agents_available:
            self.cur_agent = agent
            self.cur_meta = "_thumbsup"

            # send thumbsup signal
            self._send_meta()


    def _send_meta(self):
        if self.cur_agent in self.agents_available and self.cur_meta in self.meta_signals:
            # send signal
            self.comm_channel.appendleft({"out": {"agent": self.cur_agent, "meta": self.cur_meta}})
            print("MetaCommunication: sent", {"agent": self.cur_agent, "meta": self.cur_meta})


    def rcv_meta(self):
        signal = None

        if len(self.comm_channel) > 0:
            if "in" in self.comm_channel[-1]:
                # set boolean for new information
                self.has_new_information = True
                # received a meta-communicative signal
                message = self.comm_channel.pop()
                signal = message["in"]
                if "agent_name" in message:
                    source_agent = message["agent_name"]
                    if source_agent not in self.agents_available:
                        print("MetaCommunication ERROR:", source_agent, "not in list of available agents:", self.agents_available)
                        return None
                    elif "meta" in signal:
                        if signal["meta"] not in self.meta_signals:
                            print("MetaCommunication ERROR:", signal["meta"], "not in list of available signals!")
                            return None
                        elif "agent" in signal and signal["agent"] == self.my_id:
                            print("MetaCommunication: received", signal["meta"], "from agent", source_agent)
                            if self.agents_meta[source_agent] is not [] and signal["meta"] not in self.agents_meta[source_agent]:
                                self.agents_meta[source_agent].append(signal["meta"])

        return signal


    def set_information_received(self):
        self.has_new_information = False






