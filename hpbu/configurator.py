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

""" Hierarchy configuration
Created on 07.02.2017

@author: skahl
"""
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

# modules
import sys
import os

import json


class Config(object):

    def __init__(self, path, filename=None):
        self.layers = []
        self.path = path
        self.config_file = filename
        self.knowledge_file = None
        self.parameters= {}
        self.storage = None
        self.knowledge = None



    def get_config_storage(self):
        if self.config_file is not None:
            try:
                filepath = self.path + os.sep + self.config_file
                print("\t # # # loading config:", filepath)
                filereader = open(filepath, 'r')
                data_json = filereader.read()
                self.storage = json.loads(data_json)
                filereader.close()
                return self.storage
            except IOError as error:
                print(error)
                sys.exit(1)
        else:
            return None

    
    def get_knowledge_storage(self):
        if self.knowledge_file is not None:
            try:
                filepath = self.path + os.sep + self.knowledge_file
                print("\t # # # loading knowledge:", filepath)
                filereader = open(filepath, 'r')
                data_json = filereader.read()
                self.knowledge = json.loads(data_json)
                filereader.close()
                return self.knowledge
            except IOError as error:
                print(error)
                return {}
        else:
            return None



    def config_layer_from_storage(self):
        if "layers" in self.storage:
            layers = self.storage['layers']
            for cfg in layers:
                self.add_layer_from_config(cfg)
            return True
        else:
            return False



    def config_parameters_from_storage(self):
        # config the following model parameters:
        if "parameters" in self.storage:
            params = self.storage["parameters"]
            self.parameters['my_id'] = params['my_id']  # agent id or name, which is to be configured here
            self.parameters['memory_len'] = params['memory_len']  # transient memory length
            self.parameters['update_delay'] = params['update_delay']  # hierarchy update delay
            self.knowledge_file = params['knowledge_file']  # where to read the knowledge from
            self.parameters['store_knowledge'] = params['store_knowledge']  # boolean of whether to store learned hypotheses
            self.parameters['read_knowledge'] = params['read_knowledge']  # boolean of whether to read learned hypotheses
            self.parameters['vis_server'] = params['vis_server']  # read ip of visualization server
            self.parameters['time_step'] = params['time_step']  # time step of model
            self.parameters['self_supervised'] = params['self_supervised']  # allow self-supervised learning of new sequences
            self.parameters['personmodel_file'] = params['personmodel_file']  # filename of the personmodel information
            self.parameters['interaction_influence_mode'] = params['interaction_influence_mode']  # interaction partner influence mode selection
            self.parameters['bias_gain'] = params['bias_gain']  # gain factor for how fast evidence is integrating over time
            return True
        else:
            return False



    def add_layer(self, type, name, color, filename=None):
        conf = {'type': type,
                'name': name,
                'color': color,
                'filename': filename
                }
        self.layers.append(conf)



    def add_layer_from_config(self, config):
        self.layers.append(config)



    def store_knowledge_in_storage(self, hierarchy):   
        if self.knowledge_file is not None:
            self.knowledge = {}
            layers = self.storage["layers"]  # read copy from storage
            for layer in layers:
                if layer["type"] in ['MotorControl']:
                    print(hierarchy.layer_io.name, hierarchy.layer_io.hypotheses.reps.keys())
                    if layer["name"] not in self.knowledge:
                        self.knowledge[layer["name"]] = {}
                    if "knowledge" not in self.knowledge[layer["name"]]:
                        self.knowledge[layer["name"]] = {}
                    
                    self.knowledge[layer["name"]]["type"] = layer["type"]
                    self.knowledge[layer["name"]]["knowledge"] = hierarchy.layer_io.hypotheses.serialize()
                elif layer["type"] == 'Vision':
                    print(hierarchy.layer_vision.name, hierarchy.layer_vision.hypotheses.reps.keys())
                    if layer["name"] not in self.knowledge:
                        self.knowledge[layer["name"]] = {}
                    if "knowledge" not in self.knowledge[layer["name"]]:
                        self.knowledge[layer["name"]] = {}

                    self.knowledge[layer["name"]]["type"] = layer["type"]
                    self.knowledge[layer["name"]]["knowledge"] = hierarchy.layer_vision.hypotheses.serialize()
                elif layer["type"] in ["Top", "Goals"]:
                    print(hierarchy.layer_top.name, hierarchy.layer_top.hypotheses.reps.keys())
                    if layer["name"] not in self.knowledge:
                        self.knowledge[layer["name"]] = {}
                    if "knowledge" not in self.knowledge[layer["name"]]:
                        self.knowledge[layer["name"]] = {}
                        
                    self.knowledge[layer["name"]]["type"] = layer["type"]
                    self.knowledge[layer["name"]]["knowledge"] = hierarchy.layer_top.hypotheses.serialize()
                else:
                    for hierarchy_layer in hierarchy.layer_between:
                        if layer["name"] == hierarchy_layer.name:
                            print(hierarchy_layer.name, hierarchy_layer.hypotheses.reps.keys())
                            if layer["name"] not in self.knowledge:
                                self.knowledge[layer["name"]] = {}
                            if "knowledge" not in self.knowledge[layer["name"]]:
                                self.knowledge[layer["name"]] = {}
                        
                            self.knowledge[layer["name"]]["type"] = layer["type"]
                            self.knowledge[layer["name"]]["knowledge"] = hierarchy_layer.hypotheses.serialize()

            try:
                print("writing:", self.knowledge_file)
                filewriter = open(self.knowledge_file, 'w')
                data_json = json.dumps(self.knowledge, indent=4)  # indent 4 to make json readable
                filewriter.write(data_json)
                filewriter.close()
            except IOError as error:
                print(error)
                sys.exit(1)
        else:
            print("No knowledgefile was configured! NOT SAVING KNOWLEDGE!")



    def restore_knowledge_from_storage(self, top_layer, between_layers, io_layer, vision_layer):
        if self.knowledge is None and self.knowledge_file is not None:
            self.knowledge = self.get_knowledge_storage()

        # look for example knowledge storage in (possibly) old config storage
        if "knowledge" in self.storage["layers"][0]:
            # if there still is knowledge in the config, OVERRIDE and load from that
            for layer in self.storage["layers"]:
                self.knowledge[layer["name"]] = {"type": layer["type"], "knowledge": layer["knowledge"]}
        
        for layer_name, layer in self.knowledge.items():
            if layer["type"] in ['MotorControl']:
                if "knowledge" in layer:
                    print("restoring knowledge for layer", layer_name)
                    io_layer.hypotheses = io_layer.hypotheses.deserialize(layer["knowledge"])
                    print("hypos:", io_layer.hypotheses.reps.keys())
                    io_layer.reset()
                    last_layer = io_layer
            elif layer["type"] == 'Vision':
                if "knowledge" in layer:
                    print("restoring knowledge for layer", layer_name)
                    vision_layer.hypotheses = vision_layer.hypotheses.deserialize(layer["knowledge"])
                    print("hypos:", vision_layer.hypotheses.reps.keys())
                    vision_layer.reset()
                    last_layer = vision_layer
            elif layer["type"] in ["Top"]:
                if "knowledge" in layer:
                    print("restoring knowledge for layer", layer_name)
                    top_layer.hypotheses = top_layer.hypotheses.deserialize(layer["knowledge"], last_layer)
                    print("hypos:", top_layer.hypotheses.reps.keys())
                    top_layer.reset()
                    last_layer = top_layer
            else:
                if "knowledge" in layer:
                    for hierarchy_layer in between_layers:
                        if layer_name == hierarchy_layer.name:
                            print("restoring knowledge for layer", layer_name)
                            hierarchy_layer.hypotheses = hierarchy_layer.hypotheses.deserialize(layer["knowledge"], last_layer)
                            print("hypos:", hierarchy_layer.hypotheses.reps.keys())
                            hierarchy_layer.reset()
                            last_layer = hierarchy_layer


        return top_layer, between_layers, io_layer, vision_layer



    def get_config_for_visualization(self):
        """ Retrieve a reduced version of the configuration for sending to visualization setup.
        """
        conf_layers = []
        for layer in self.layers:
            conf_layers.append({"name": layer["name"], "type": layer["type"]})

        vis_conf = {"layers": conf_layers, "parameters": self.parameters}
        return vis_conf



    def close_storage(self):
        print("closing storage!")
