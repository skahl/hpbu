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

""" Hierarchy
Created on 16.08.2017

@author: skahl
"""
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals
# system modules
import threading
from collections import deque  # from Queue import Queue
import sys
import os
# own modules
from .configurator import Config
from .visionlayer import VisionLayer
from .motorcontrol import MotorControl
from .sequencelayer import SequenceLayer
from .clusterlayer import ClusterLayer
from .toplayer import TopLayer
from .realizations import Realizations
from .goals import Goals
from .ment_rule_parser import Parser
from .ment import MetaCommunication
from .functions import *

# timing for information update
from time import time

class Hierarchy(object):


    def __init__(self, config):
        """ Initialize the hierarchy by a config list, containing name and type dictionaries.
        """
        # logging setup
        self.logger = Logger('White', 'HPBU')
        self.log = self.logger.log
        self.error = self.logger.error

        # hierarchy layer setup
        self.is_stopped = False
        self.updates_enabled = False
        self.config = config
        self.layer_io = None
        self.layer_vision = None
        self.layer_between = []
        self.layer_top = None
        self.update_delay = self.config.parameters['update_delay']  # seconds

        # optional interaction layer modules
        self.personmodel = None

        self.time_step = self.config.parameters['time_step']  # seconds
        self.update_info_delay = 0.1  # seconds TODO: decrease!!
        self.time_since_update = 0
        self.last_time_info = time()
        # TODO: make use of self.config.parameters['memory_len']  # currently defaulting to 5 everywhere

        # hierarchy, possible modules
        layer_dict = {
            'MotorControl': MotorControl,
            'Vision': VisionLayer,
            'Sequence': SequenceLayer,
            'Cluster': ClusterLayer,
            'Top': TopLayer,
            'Realizations': Realizations,
            'Goals': Goals
        }

        # load and parse PersonModel from xml, if parameter is given
        if self.config.parameters["personmodel_file"] is not "":
            personmodel_file = self.config.parameters["personmodel_file"]
            # if we are processing PersonModel knowledge for 'Seq' layer, load, parse and connect it
            self.log(1, "Loading PersonModel knowledge from xml...", personmodel_file)
            resource_path = os.environ.get('RESOURCE_PATH')
            # select xml-file specific parser
            parser_selector = Parser(filename=resource_path + os.sep + personmodel_file)
            parser = parser_selector.select_parser()
            # parse all available personmodels
            personmodels = parser.parse()
            if personmodels is not None and personmodels is not {}:
                # select the one personmodel identified in config by my_id
                my_id = self.config.parameters.get("my_id", None)
                if my_id in personmodels:
                    personmodel = personmodels[my_id]
                    personmodel.my_id = my_id
                    self.log(1, "Selected PersonModel:\n", personmodel)
                    self.personmodel = personmodel
                else:
                    print(my_id)
                    self.error("Configured my_id cannot be found in available PersonModels:", my_id)
                    sys.exit(1)

        # if PersonModel was available, also try to configure a MetaCommunication object
        if self.personmodel is not None:
            meta_comm_queue = deque(maxlen=100)
            meta_communication = MetaCommunication(self.personmodel.my_id, self.personmodel.agents, meta_comm_queue)
            self.personmodel.set_meta_communication(meta_communication)
            # set PersonModel interaction_influence_mode parameter
            if self.config.parameters["interaction_influence_mode"] is not None:
                self.personmodel.interaction_influence_mode = self.config.parameters["interaction_influence_mode"]

        # load and parse configured layers
        for line in self.config.layers:
            layer = layer_dict[line['type']]

            if line['type'] in ['MotorControl']:
                self.layer_io = layer(line['name'])
                self.layer_io.set_logger(line['color'])
                self.layer_io.set_parameters(line['parameters'] if "parameters" in line else {})
                # add global parameters to local layer parameters
                for key, val in self.config.parameters.items():
                    self.layer_io.params[key] = val
                # reference personmodel, if available
                self.layer_io.personmodel = self.personmodel
                # reference to hierarchy methods
                self.layer_io.hierarchy_sleep = self.hierarchy_sleep
            elif line['type'] in ['Top', 'Goals']:
                self.layer_top = layer(line['name'])
                self.layer_top.set_logger(line['color'])
                self.layer_top.set_parameters(line['parameters'] if "parameters" in line else {})
                # add global parameters to local layer parameters
                for key, val in self.config.parameters.items():
                    self.layer_top.params[key] = val
                # reference personmodel, if available
                self.layer_top.personmodel = self.personmodel
                # reference to hierarchy methods
                self.layer_top.hierarchy_sleep = self.hierarchy_sleep

                if "filename" in line and line['filename'] is not None:
                    """ loading routine to parse MENT-knowledge and load into hierarchy
                    """
                    self.log(1, "Loading top layer knowledge from xml...", line['filename'])
                    resource_path = os.environ.get('RESOURCE_PATH')
                    # select xml-file specific parser
                    parser_selector = Parser(filename=resource_path + os.sep + line['filename'])
                    parser = parser_selector.select_parser()
                    # parse
                    _dict = parser.parse()
                    # print(line['name'], "layer xml deserialization:", _dict)
                    # load into layer structure
                    num_hypos = len(_dict)
                    for hypo_id, hypo in _dict.items():
                        self.layer_top.hypotheses.add_hypothesis_from_existing_repr(hypo, 1. / num_hypos)

            elif "Vision" == line['type']:
                self.layer_vision = layer(line['name'])
                self.layer_vision.set_logger(line['color'])
                self.layer_vision.set_parameters(line['parameters'] if "parameters" in line else {})
                # add global parameters to local layer parameters
                for key, val in self.config.parameters.items():
                    self.layer_vision.params[key] = val
                # reference personmodel, if available
                self.layer_vision.personmodel = self.personmodel
                # reference to hierarchy methods
                self.layer_vision.hierarchy_sleep = self.hierarchy_sleep

            else:
                self.layer_between.append(layer(line['name']))
                self.layer_between[-1].set_logger(line['color'])
                self.layer_between[-1].set_parameters(line['parameters'] if "parameters" in line else {})
                # add global parameters to local layer parameters
                for key, val in self.config.parameters.items():
                    self.layer_between[-1].params[key] = val
                # reference personmodel, if available
                self.layer_between[-1].personmodel = self.personmodel
                # reference to hierarchy methods
                self.layer_between[-1].hierarchy_sleep = self.hierarchy_sleep

                if "filename" in line and line['filename'] is not None:
                    """ loading routine to parse MENT-knowledge and load into hierarchy
                    """
                    self.log(1, "Loading in-between layer knowledge from xml...", line['filename'])
                    resource_path = os.environ.get('RESOURCE_PATH')
                    # select xml-file specific parser
                    parser_selector = Parser(filename=resource_path + os.sep + line['filename'])
                    parser = parser_selector.select_parser()
                    # parse
                    _dict = parser.parse()
                    # print(line['name'], "layer xml deserialization:", _dict)

                    # load into layer structure
                    num_hypos = len(_dict)
                    for hypo_id, hypo in _dict.items():
                        self.layer_between[-1].hypotheses.add_hypothesis_from_existing_repr(hypo, 1. / num_hypos)

        if self.layer_io is None or self.layer_top is None:
            self.error("At least one layer of type 'MotorControl' and 'Top' have to be defined!")
            sys.exit(1)
        else:
            self.log(1, "Successfully set up the hierarchy with the following layers:")
            self.log(1, self.layer_top)
            [self.log(1, l) for l in self.layer_between[::-1]]
            self.log(1, self.layer_vision)
            self.log(1, self.layer_io)

        # mark top-layer as top-layer
        self.layer_top.is_top_layer = True


        """ loading routine to retrieve already learned layer knowledge from data storage
        """
        if self.config.parameters["read_knowledge"]:
            self.log(1, "Restoring layer knowledge from data storage...")
            self.layer_top, self.layer_between, self.layer_io, self.layer_vision = self.config.restore_knowledge_from_storage(self.layer_top, self.layer_between, self.layer_io, self.layer_vision)


        """ Define influence, output and introspection necessities """
        self.evidence_top_layer = None
        self.long_range_projection = {}
        self.lower_level_hypotheses = {}
        # prepare lower_level_hypotheses update house keeping
        lower_level = self.layer_vision
        for l in self.layer_between:
            self.lower_level_hypotheses[l.name] = lower_level.hypotheses
            lower_level = l 
        self.lower_level_hypotheses[self.layer_top.name] = lower_level.hypotheses
        # prepare evidence update to next higher level house keeping
        self.evidences = {}
        self.evidences[self.layer_top.name] = [None, None, None]
        [self.evidences.update({l.name: [None, None, None]}) for l in self.layer_between]
        self.evidences[self.layer_io.name] = [None]
        """ Define multithreading necessities """
        self.queue_input = deque(maxlen=100)
        self.queue_output = deque(maxlen=100)
        self.queue_info = deque(maxlen=100)
        self.queue_top_influence = deque(maxlen=100)
        self.queue_long_range_projection = deque(maxlen=100)


    # @profile
    def update(self):
        """ Full hierarchy update routine.
        This is a "prediction-first" update, so we update with the TopLayer first,
        traversing the hierarchy down until we reach the IOLayer.

        Send input via queue_input using dict {"input":your input}
        Quit the update thread using dict {"control":"quit"} in queue_input.
        Receive output via queue_output.
        """

        # control prep
        control = ""
        # visualization information collection prep
        collected_info = {}
        # highly specialized variables to make sure realization intentions are stored in model info
        realizations_intention = None

        #################################
        while control != "quit":
            # input prep
            input_proprioception = None
            input_vision = None
            input_meta = None  # meta-communicative signals
            influence_top_layer = [None, None]
            prediction = [None, None]
            prunable = None
            long_range_projection = {}
            higher_layer_name = None
            lrp = None

            # highly specialized variables to make sure realization intentions are stored in model info
            realizations_intention_changed = False


            # Per-layer clean-up
            self.layer_top.clean_up()
            for layer in self.layer_between:
                layer.clean_up()
            self.layer_vision.clean_up()
            self.layer_io.clean_up()

            # timer
            self.time_since_update = time() - self.last_time_info

            # hierarchy update

            # Check queue for sensory input or control input
            # print("update input queue:", len(self.queue_input))
            if len(self.queue_input) > 0:
                _input = self.queue_input.pop()
                _val = _input[list(_input.keys())[0]]
                if "control" in _val:
                    control = _val["control"]
                if "proprioception" in _val:
                    input_proprioception = _val["proprioception"]
                if "vision" in _val:
                    # filter by attention on interaction partner agent
                    if self.personmodel is not None:
                        if "agent_name" in _val and _val["agent_name"] in [self.personmodel["me_focus"], self.personmodel["my_id"], None]:
                            input_vision = _val["vision"]
                        elif "agent_name" in _val:
                            print("unfocused agent:", _val["agent_name"], "focus:", [self.personmodel["me_focus"], self.personmodel["my_id"], None])
                        else:
                            print("received behavior from unknown agent")
                    else:
                        input_vision = _val["vision"]
                    
                if "meta" in _val:
                    input_meta = _val["meta"]
                    if self.personmodel is not None:
                        # todo: could be encapsulated in external comm_channel extending method
                        self.personmodel.meta_communication.comm_channel.appendleft({"in": input_meta, "agent_name": _val["agent_name"]})
                        self.personmodel.meta_communication.rcv_meta()

                # Check queue for influencing top-layer input
                if len(self.queue_top_influence) > 0:
                    _input = self.queue_top_influence.pop()
                    if "input" in _input:
                        influence_top_layer = _input["input"]

            # inhibit updates if necessary
            if not self.updates_enabled:
                sleep(0.1)
            else:
                # Check queue for outside long-range projection input
                if len(self.queue_long_range_projection) > 0:
                    _lr = self.queue_long_range_projection.pop()
                    for target, com in _lr.items():
                        long_range_projection[target] = com

                # print("\nlrp:", long_range_projection)

                # TopLayer update
                self.layer_top.receive_prediction(influence_top_layer)  # from next higher layer, or external source
                self.layer_top.receive_long_range_projection(long_range_projection.get(self.layer_top.name, None))
                self.layer_top.receive_lower_level_hypos(self.lower_level_hypotheses[self.layer_top.name])
                self.layer_top.receive_evidence(self.evidences[self.layer_top.name])
                self.layer_top.update()
                prediction = self.layer_top.send_prediction()
                prunable = self.layer_top.send_prunable()
                lrp = self.layer_top.send_long_range_projection()
                self.set_long_range_projection(lrp)
                higher_layer_name = self.layer_top.name
                # collect layer information for visualization
                if len(self.layer_top.hypotheses) > 0:
                    collected_info[self.layer_top.name] = {"hypotheses": self.layer_top.hypotheses.dpd.tolist(),
                                                           "free_energy": self.layer_top.free_energy,
                                                           "precision": self.layer_top.K,
                                                           "self": self.layer_top.self_estimate,
                                                           "intention": str(self.layer_top.intention)}

                # Traverse the hierarchy in reverse order (top-down)
                for layer in self.layer_between[::-1]:
                    # enable PersonModel influence in Seq layer if personmodel is available
                    if layer.name == "Seq" and self.personmodel is not None:
                        layer.personmodel_influence = True  # enable PersonModel influence to allow prior knowledge to influence belief updates
                    
                    layer.receive_prediction(prediction)
                    layer.receive_long_range_projection(long_range_projection.get(layer.name, None))
                    layer.receive_lower_level_hypos(self.lower_level_hypotheses[layer.name])
                    layer.receive_evidence(self.evidences[layer.name])
                    layer.receive_prunable(prunable)
                    layer.update()
                    self.lower_level_hypotheses[higher_layer_name] = layer.send_level_hypos()
                    self.evidences[higher_layer_name] = layer.send_evidence()  # store evidence dict by layer name on the fly
                    higher_layer_name = layer.name  # save this layer's name so that next lower layer knows where to send evidence
                    prediction = layer.send_prediction()
                    prunable = layer.send_prunable()
                    lrp = layer.send_long_range_projection()
                    self.set_long_range_projection(lrp)

                    str_intention = str(layer.intention)
                    # highly specialized realizations intention query
                    if layer.name == "Realizations" and layer.intention is not None:
                        if str_intention != realizations_intention:
                            realizations_intention_changed = True
                            realizations_intention = str_intention

                    # collect layer information for visualization
                    if len(layer.hypotheses) > 0:
                        collected_info[layer.name] = {"hypotheses": layer.hypotheses.dpd.tolist(),
                                                      "free_energy": layer.free_energy,
                                                      "precision": layer.K,
                                                      "self": layer.self_estimate,
                                                      "intention": str_intention}

                # VisionLayer update (only receives visual updates and sends evidence to next higher layer)
                self.layer_vision.receive_prediction(prediction)
                self.layer_vision.receive_long_range_projection(long_range_projection.get(self.layer_vision.name, None))
                self.layer_vision.receive_evidence(input_vision)  # input from vision
                self.layer_vision.receive_prunable(prunable)
                self.layer_vision.update()
                # no predictions in this layer and new higher_layer_name for evidence
                self.evidences[higher_layer_name] = self.layer_vision.send_evidence()
                lrp = self.layer_vision.send_long_range_projection()
                self.lower_level_hypotheses[higher_layer_name] = self.layer_vision.send_level_hypos()
                self.set_long_range_projection(lrp)
                # collect layer information for visualization
                if len(self.layer_vision.hypotheses) > 0:
                    collected_info[self.layer_vision.name] = {"hypotheses": self.layer_vision.hypotheses.dpd.tolist(),
                                                              "free_energy": self.layer_vision.free_energy,
                                                              "precision": self.layer_vision.K,
                                                              "self": self.layer_vision.self_estimate,
                                                              "intention": str(self.layer_vision.intention)}

                # motor update

                # IOLayer update (only receives proprioceptive updates and sends evidence to next higher layer, skipping VisionLayer)
                self.layer_io.receive_prediction(prediction)
                self.layer_io.receive_long_range_projection(long_range_projection.get(self.layer_io.name, None))
                self.layer_io.receive_evidence(input_proprioception)  # input from proprioception
                self.layer_io.receive_prunable(prunable)
                self.layer_io.update()
                lrp = self.layer_io.send_long_range_projection()
                self.set_long_range_projection(lrp)
                # no new higher_layer_name for evidence
                # self.evidences[higher_layer_name] = self.layer_io.send_evidence()
                # collect layer information for visualization
                if len(self.layer_io.hypotheses) > 0:
                    collected_info[self.layer_io.name] = {"hypotheses": self.layer_io.hypotheses.dpd.tolist(),
                                                          "free_energy": self.layer_io.free_energy,
                                                          "precision": self.layer_io.K,
                                                          "self": self.layer_io.self_estimate,
                                                          "intention": str(self.layer_io.intention)}
                # output (active inference)
                prediction = self.layer_io.send_prediction()
                if prediction is not None and prediction[0] is not None:
                    only_prediction = prediction[0]
                    self.queue_output.appendleft({"angle": only_prediction})


                # output of meta-communicative signals
                if self.personmodel is not None:
                    comm_channel = self.personmodel.meta_communication.comm_channel
                    if len(comm_channel) > 0:
                        if "out" in comm_channel[-1]:
                            # received a meta-communicative signal
                            signal = comm_channel.pop()["out"]
                            self.queue_output.appendleft({"meta": signal})

                # communicative intention available in Realizations?

                # check time since info updates
                if self.time_since_update > self.update_info_delay or realizations_intention_changed:
                    self.time_since_update = 0.
                    self.last_time_info = time()
                    # Collect Layer info and queue as info
                    # self.log(0, "storing collected information")
                    self.queue_info.appendleft(collected_info)

                # sleep a bit to save processor time
                sleep(self.update_delay)
        #################################

        ################ finalize
        self.finalize()

        self.log(0, "... quit hierarchy update process.")


    def disable_updates(self):
        self.updates_enabled = False



    def enable_updates(self):
        self.updates_enabled = True



    def get_output(self):
        """ Return output from the hierarchy if there is some.
        """
        if len(self.queue_output) > 0:
            return self.queue_output.pop()
        else:
            return None



    def set_input(self, uuid, _input):
        """ Put given input dict into the queue.
        A dict is expected with either a "control", "proprioception", or "vision" key.
        """
        # source agent mustn't be this model, here, if meta-communication is going on
        if "control" in _input or "proprioception" in _input or "vision" in _input or\
                ("meta" in _input and "agent_name" in _input and _input["agent_name"] != self.personmodel.my_id):
            if len(self.queue_input) > 0:
                _prior_input = self.queue_input[0]
                if uuid in _prior_input:
                    new_input = _prior_input[uuid]
                    new_input.update(_input)
                    self.queue_input[0] = {uuid: new_input}
                else:
                    self.queue_input.appendleft({uuid: _input})
                    # print("appended:", uuid)
            else:
                self.queue_input.appendleft({uuid: _input})
                # print("no prior entries:", uuid)

            # print(len(self.queue_input), self.queue_input)
            return True
        else:
            self.error("A dict is expected with at least one of these keys: control, proprioception, vision, or meta")
            return False

        if len(self.queue_input) > 99:
            self.error("Input queue is full!")
            return False



    def set_topdown_influence(self, _input):
        """ Set a specific influence boost in the top-layer.
        """
        if "input" in _input:
            self.queue_top_influence.appendleft(_input)
        else:
            self.error("A dict is expected with an 'input' key.")

        if len(self.queue_top_influence) > 99:
            self.error("Top-down influence queue is full!")



    def set_long_range_projection(self, _input):
        """ Set a specific input to a layer.
        Input expects a list of [layer_name, influence].
        """
        if _input is not None:
            if len(_input) > 0:
                self.queue_long_range_projection.appendleft(_input)
            else:
                self.error("A list is expected with structure [layer_name, influence].")

        if len(self.queue_top_influence) > 99:
            self.error("Long range projection queue is full!")



    def finalize(self):
        """ Call finalize method for each layer.
        """
        self.layer_top.finalize()

        for layer in self.layer_between[::-1]:
            layer.finalize()

        self.layer_io.finalize()
        self.layer_vision.finalize()

        # signal hierarchy processing stopping
        self.is_stopped = True



    def print_report(self):
        """ Display a report on model statistics and content.
        """
        self.log(0, "\n\n")
        self.log(0, "-" * 10, "Hierarchy output:")
        self.log(0, self.layer_top.print_out())

        for layer in self.layer_between[::-1]:
            self.log(0, layer.print_out())

        self.log(0, self.layer_io.print_out())
        self.log(0, self.layer_vision.print_out())



    def hierarchy_sleep(self, seconds):
        sleep(seconds)



    def run(self):
        """ Start the thread updating the hierarchy to have
        a non-blocking instance of the predictive hierarchy.
        """

        update_thread = threading.Thread(target=self.update)
        update_thread.daemon = False
        update_thread.start()

        self.log(1, "Running hierarchy update process ...")



    def hierarchy_is_stopped(self):
        return self.is_stopped

