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

""" Rule_parser module for parsing xml mental state rules
Created on 22.02.2018

@author: skahl
"""

# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

import sys
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET

from .ment import *


class Parser(object):

    def __init__(self, filename):
        try:
            self.filename = filename
            self.tree = ET.parse(self.filename)
            self.root = self.tree.getroot()
        except ET.ParseError as v:
            row, column = v.position
            print(str(v))
            return None
        except IOError as e:
            print(str(e))
        except:
            print(str(sys.exc_info()[0]))
            return None


    def select_parser(self):
        node = self.root

        if node.tag == "RULES":
            parser_id = node.get("parser", None)
            if parser_id is not None:
                if parser_id == "goals":
                    return GoalsParser(self.filename)

                if parser_id == "realizations":
                    return RealizationsParser(self.filename)

                if parser_id == "personmodel":
                    return PersonmodelParser(self.filename)
            else:
                print("No parser information found in xml. Exit!")
                sys.exit(1)



class PersonmodelParser(object):

    def __init__(self, filename):    
        try:
            self.filename = filename
            self.tree = ET.parse(self.filename)
            self.root = self.tree.getroot()
        except ET.ParseError as v:
            row, column = v.position
            print(str(v))
            return None
        except IOError as e:
            print(str(e))
        except:
            print(str(sys.exc_info()[0]))
            return None


    def parse(self, node=None):

        my_personmodels = {}

        if node is None:
            node = self.root

        if node.tag == "RULES":
            for personmodel in node:
                my_id = personmodel.get("id", None)
                # personmodel ID is necessary to identify agents in multiagent settings
                if my_id is not None:
                    new_personmodel = {}

                    new_personmodel["me"] = []
                    new_personmodel["you"] = []
                    new_personmodel["we"] = []
                    new_personmodel["agents"] = {}

                    you_models = defaultdict(list)
                    # you_model contains several you, with several schemas, containing sequences
                    for state in personmodel:
                        if state.tag == "me":
                            for blf in state:
                                new_personmodel["me"].append( blf.text )

                        if state.tag == "we":
                            for blf in state:
                                new_personmodel["we"].append( blf.text )

                        if state.tag == "you":
                            you_id = state.get("id", None)
                            if you_id is not None:
                                you_id = you_id
                                for seq in state:
                                    you_models[you_id].append( int(seq.text) )

                                you_present = state.get("present", None)
                                if you_present is not None:
                                    if you_present == "true":
                                        new_personmodel["agents"].update( {you_id: 0.} )

                    new_personmodel["you"] = you_models

                    my_personmodels[my_id] = PersonModel(me=new_personmodel["me"],
                                                        you=new_personmodel["you"],
                                                        we=new_personmodel["we"],
                                                        agents=new_personmodel["agents"])

        return my_personmodels



class GoalsParser(object):

    def __init__(self, filename):    
        try:
            self.filename = filename
            self.tree = ET.parse(self.filename)
            self.root = self.tree.getroot()
        except ET.ParseError as v:
            row, column = v.position
            print(str(v))
            return None
        except IOError as e:
            print(str(e))
        except:
            print(str(sys.exc_info()[0]))
            return None


    def parse(self, node=None):

        goals_dict = {}

        if node is None:
            node = self.root

        if node.tag == "RULES":
            for stategoalpair in node:
                new_goal = {}
                new_goal["id"] = int(stategoalpair.get("id", None))
                new_goal["comment"] = stategoalpair.get("comment", None)

                realizations = []
                for realization in stategoalpair:
                    if realization.tag == "state":
                        ms = realization.find("mentalstate")
                        if ms is not None:
                            me_blfs = []
                            me = ms.find("me")
                            if me is not None:
                                for blf in me:
                                    me_blfs.append( Belief(blf.text) )

                            you_blfs = []
                            you = ms.find("you")
                            if you is not None:
                                for blf in you:
                                    you_blfs.append( Belief(blf.text) )

                            we_blfs = []
                            we = ms.find("we")
                            if we is not None:
                                for blf in we:
                                    we_blfs.append( Belief(blf.text) )

                            new_goal["state"] = MentalState(me=me_blfs, you=you_blfs, we=we_blfs)

                    if realization.tag == "goal":
                        ms = realization.find("mentalstate")
                        if ms is not None:
                            me_blfs = []
                            me = ms.find("me")
                            if me is not None:
                                for blf in me:
                                    me_blfs.append( Belief(blf.text) )

                            you_blfs = []
                            you = ms.find("you")
                            if you is not None:
                                for blf in you:
                                    you_blfs.append( Belief(blf.text) )

                            we_blfs = []
                            we = ms.find("we")
                            if we is not None:
                                for blf in we:
                                    we_blfs.append( Belief(blf.text) )

                            new_goal["goal"] = MentalState(me=me_blfs, you=you_blfs, we=we_blfs)

                    if realization.tag == "realization":
                        realizations.append(int(realization.get("id")))

                new_goal["realizations"] = realizations

                goals_dict[new_goal["id"]] = Goals(idx=new_goal["id"],
                                                   comment=new_goal["comment"],
                                                   state=new_goal["state"],
                                                   realizations=new_goal["realizations"],
                                                   goal=new_goal["goal"])

        return goals_dict



class RealizationsParser(object):

    def __init__(self, filename):
        try:
            self.filename = filename
            self.tree = ET.parse(self.filename)
            self.root = self.tree.getroot()
        except ET.ParseError as v:
            row, column = v.position
            print(str(v))
            return None
        except IOError as e:
            print(str(e))
        except:
            print(str(sys.exc_info()[0]))
            return None


    def parse(self, node=None):

        realizations_dict = {}

        if node is None:
            node = self.root

        """ Outermost tag == RULES """
        if node.tag == "RULES":
            for realization in node:
                new_realization = {}
                new_realization["id"] = int(realization.get("id", None))
                new_realization["comment"] = realization.get("comment", None)

                # state and goal pairs will also be contained in the substates list
                substates = []

                for state in realization:
                    if state.tag == "state":
                        ms = state.find("mentalstate")
                        if ms is not None:
                            me_blfs = []
                            me = ms.find("me")
                            if me is not None:
                                for blf in me:
                                    me_blfs.append( Belief(blf.text) )

                            you_blfs = []
                            you = ms.find("you")
                            if you is not None:
                                for blf in you:
                                    you_blfs.append( Belief(blf.text) )

                            we_blfs = []
                            we = ms.find("we")
                            if we is not None:
                                for blf in we:
                                    we_blfs.append( Belief(blf.text) )

                            new_realization["state"] = MentalState(me=me_blfs, you=you_blfs, we=we_blfs)

                    if state.tag == "goal":
                        ms = state.find("mentalstate")
                        if ms is not None:
                            me_blfs = []
                            me = ms.find("me")
                            if me is not None:
                                for blf in me:
                                    me_blfs.append( Belief(blf.text) )

                            you_blfs = []
                            you = ms.find("you")
                            if you is not None:
                                for blf in you:
                                    you_blfs.append( Belief(blf.text) )

                            we_blfs = []
                            we = ms.find("we")
                            if we is not None:
                                for blf in we:
                                    we_blfs.append( Belief(blf.text) )

                            new_realization["goal"] = MentalState(me=me_blfs, you=you_blfs, we=we_blfs)

                    if state.tag == "substates":
                        for child in state:
                            # motor intention acting out beliefs
                            if child.tag == "intention":
                                belief = child.get("belief", None)
                                is_signaling = child.get("signaling", None)

                                if belief is None:
                                    print("error parsing goal realization with id=" + str(new_realization["id"]) + ": belief of intention cannot be None")

                                is_signaling = True if is_signaling is not None and is_signaling == "true" else False
                                intention = Intention(intent=belief, signaling=is_signaling)

                                substates.append( intention )

                            # check for intermittent mental state
                            if child.tag == "mentalstate":
                                me_blfs = []
                                me = child.find("me")
                                if me is not None:
                                    for blf in me:
                                        me_blfs.append( Belief(blf.text) )

                                you_blfs = []
                                you = child.find("you")
                                if you is not None:
                                    for blf in you:
                                        you_blfs.append( Belief(blf.text) )

                                we_blfs = []
                                we = child.find("we")
                                if we is not None:
                                    for blf in we:
                                        we_blfs.append( Belief(blf.text) )

                                substates.append( MentalState(me=me_blfs, you=you_blfs, we=we_blfs) )

                # add state and goal MentalStates to substates list for better comparability
                substates.insert(0, new_realization["state"])
                substates.append(new_realization["goal"])

                realizations_dict[new_realization["id"]] = Realization(idx=new_realization["id"],
                                                                       comment=new_realization["comment"],
                                                                       state=new_realization["state"],
                                                                       goal=new_realization["goal"],
                                                                       substates=substates)

        return realizations_dict


if __name__ == "__main__":
    realizations_path = "../../resource/goal_realizations.xml"
    goals_path = "../../resource/state_goal_tuples.xml"

    while realizations_path == "":
        realizations_path = input("path and name of goal realizations xml file: ")
    realizations_parser = RealizationsParser(filename=realizations_path)
    realizations_dict = realizations_parser.parse()

    while goals_path == "":
        goals_path = input("path and name of goal state tuples xml file: ")
    goals_parser = GoalsParser(filename=goals_path)
    goals_dict = goals_parser.parse()

    # printout
    for realization_id, realization in realizations_dict.items():
        print(realization)

    for goal_id, goal in goals_dict.items():
        print(goal)


    



                        