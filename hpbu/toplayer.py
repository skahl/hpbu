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

""" TopLayer
Created on 16.08.2017

@author: skahl
"""
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from .clusterlayer import *


class TopLayer(ClusterLayer):
    """ TopLayer class for predictive processing hierarchy,
    specialized for clustering layers that collect similar sequences to form clusters
    and learn prototypes for each cluster. Here, also external influence can take effect.
    """


    def __init__(self, name):
        super(TopLayer, self).__init__(name)
        self.type = 'Top'
        self.reset()



    def reset(self):
        super(TopLayer, self).reset()



    def print_out(self):

        _str_ = self.name
        _str_ += "\nhypotheses ("
        _str_ += str(len(self.hypotheses))
        _str_ += "):\n"
        _str_ += str(self.hypotheses)
        return _str_


    # def integrate_evidence(self):
    #   """ Integrate evidence from next lower layer.
    #   Here, just store the evidence as potential extension material.
    #   """
    #   pass


    # def inference(self):
    #   pass


    # def extension(self):
    #   """ Decide on and do hypothesis extension and receive and let possible
    #   external influence take effect.
    #   """
    #   pass



    # def prediction(self):
    #   """ Decide on predicted cluster.
    #   """
    #   pass

    