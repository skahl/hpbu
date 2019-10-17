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

""" VisionLayer
Created on 15.08.2017

@author: skahl
"""
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from .sequencelayer import *

# from .functions import *

from collections import deque


class VisionLayer(SequenceLayer):
    """ Visual input layer class for predictive processing hierarchy,
    specialized for collecting lower level sensory activity in sequence
    of occurrence and for producing same sensorymotor activity through active inference.
    """


    def __init__(self, name):
        super(VisionLayer, self).__init__(name)
        self.type = 'Vision'

        # minimally: r=11, theta=9
        # last: 19, 13

        self.oculocentric_coordinates = define_coordinate_system(r=30, theta=29)
        self.reset()



    def reset(self):
        super(VisionLayer, self).reset()
        # store last temporal delta between divergences
        self.delta_t = 0.
        # isDrawing estimate
        self.isDrawing = True
        # store current coordinate
        self.cur_coordinate = np.array([0., 0.])
        self.last_coordinate = np.array([0., 0.])
        # self.phi_history = deque(maxlen=5)
        # store intentionality information
        self.intention_check = []
        self.last_intention = None
        self.step_counter = 0

        # follow-through necessacities
        self.allow_follow_through = False
        self.prior_distance_to_target = None
        self.will_probably_reach_target = False
        self.follow_through_factor = 0.3

        print("(Re-)initialized layer", self.name)



    def gen_primitives(self):
        # prep angle distribution
        # distribution based on angular primitive resolution of 360/20 = 20 degrees, or ~0,314 radians

        self.hypotheses.dpd = np.zeros((20, 2))
        self.hypotheses.dpd[:10, :] = np.array([[1. / 20, round(-np.pi * (10 - i) / 10, 2)] for i in range(0, 10)])
        self.hypotheses.dpd[10:, :] = np.array([[1. / 20, round(np.pi * i / 10, 2)] for i in range(0, 10)])
        self.hypotheses.reps = {round(self.hypotheses.dpd[i, 1], 2): Representation(round(self.hypotheses.dpd[i, 1], 5)) for i in range(0, 20)}

        for idx, dpd_pair in enumerate(self.hypotheses.dpd):
            self.hypotheses.reps[dpd_pair[1]].dpd_idx = idx
        self.log(2, "Generated primitives:\n", self.hypotheses.reps.keys())



    def print_out(self):

        _str_ = self.name
        _str_ += "\nhypotheses ("
        _str_ += str(len(self.hypotheses))
        _str_ += "):\n"
        _str_ += str(self.hypotheses)
        return _str_



    def integrate_evidence(self):
        """ Integrate evidence from sensory activity.
        """
        if self.lower_layer_evidence is not None:
            self.delta_t += self.lower_layer_evidence[1]

            if self.lower_layer_evidence[0] is not None:

                self.isDrawing = True
                # phi is the vector between last and current coordinate
                self.cur_coordinate += copy(self.lower_layer_evidence[0])
                self.last_coordinate = copy(self.lower_layer_evidence[0])
                phi = self.last_coordinate  # lower layer evidence is also a coordinate

                # estimate if that last movement was actually drawn (very low cost version)
                phi_jump = np_abs(phi[1])
                if phi_jump > 70:
                    self.isDrawing = False
                    self.log(3, "Possible undrawn jump!", phi_jump)

                # calculate the angle (radians) of this vector
                rad = np.round(np.arctan2(phi[1], phi[0]), 3)

                # fit the angle to the hypotheses distribution
                self.likelihood = self.fit_angle_dist(rad)  # lh to be joined with prior distribution

                self.log(3, "movement angle now:", rad, "with fit:", self.likelihood[np_argmax(self.likelihood[:, 0])])

                # count the number of steps perceived
                self.step_counter += 1

                # check if follow-through applies (if over two-thirds of the way to target has already been done)
                # if self.allow_follow_through and self.intention_check is not None and len(self.intention_check) > 0:
                #     cur_distance = np_norm(self.intention_check[0] - phi)
                #     if self.prior_distance_to_target is None:
                #         self.prior_distance_to_target = cur_distance

                #     if cur_distance / self.prior_distance_to_target < self.follow_through_factor:
                #         self.will_probably_reach_target = True
                #         # do as if intention was reached
                #         self.intention_check.append(self.intention_check[0])

            elif self.lower_layer_evidence[1] is not None:
                # good signal for a reset
                self.cur_coordinate = np.array([0., 0.])
                self.last_coordinate = np.array([0., 0.])




    def fit_angle_dist(self, rad):
        """ Fit the given radians to the distribution of radians.
        """
        dpd = self.hypotheses.dpd
        lh = np.array([[gaussian(rad, np.round(i, 3), 0.05), np.round(i, 3)] for i in dpd[:, 1]])  # sigma = 0.17
        return lh



    def td_inference(self):
        """ Integrate influence from higher layers or long range projections.
        """
        # angle possibilities
        if self.cur_coordinate is not None and self.hypotheses.dpd is not None and len(self.hypotheses.dpd) > 0:

            if self.long_range_projection is not None:
                """ These coordinates will get send from the motorcontrol for vision to verify that
                it has indeed correctly moved.
                """
                # self.log(3, "received long-range projection:", self.long_range_projection, self.last_intention)
                if len(self.intention_check) == 0 and ("goal" in self.long_range_projection or "intention" in self.long_range_projection):
                    if "goal" in self.long_range_projection:
                        LRP = self.long_range_projection["goal"]
                        # new target, convert polar to cartesian coordinates
                        r = LRP[0].r
                        theta = LRP[0].theta

                        phi = np.array([r * np.cos(theta), r * np.sin(theta)])
                    elif "intention" in self.long_range_projection:
                        # receive relative coordinates directly
                        phi = self.long_range_projection["intention"]

                    # set intention to check up on in case of LRP from proprioception
                    self.intention_check = [phi]
                    self.log(2, "New primed target coordinate:", phi)

                    rad = round(np.arctan2(phi[1], phi[0]), 3)
                    self.intention = rad

                    # fit the angle to the hypotheses distribution
                    likelihood = self.fit_angle_dist(rad)
                    # self.td_posterior = posterior(self.hypotheses.dpd, likelihood, smooth=True)
                    self.td_posterior = norm_dist(likelihood, smooth=True)

                elif len(self.intention_check) == 1 and "confirm" in self.long_range_projection:
                    LRP = self.long_range_projection["confirm"]
                    coord = LRP[0]

                    # secondary (dismissable) confirmation if last intention and corrdinate match and current and last intention differ
                    # if current and last intention are actually the same we want to know, because this may be voluntarily
                    # if self.last_intention is not None\
                    #          and self.last_intention[0] == coord[0] and self.last_intention[1] == coord[1]\
                    #          and self.last_intention[0] != self.intention[0][0] and self.last_intention[1] != self.intention[0][1]:
                    #     self.log(1, "received secondary confirmation for last intended movement goal")
                    # else:

                    self.delta_t = LRP[1]
                    self.isDrawing = LRP[2]

                    self.intention_check.append(coord)

                    phi = coord  # - self.cur_coordinate  # lower layer evidence is also a coordinate
                    # calculate the angle (radians) of this vector
                    rad = round(np.arctan2(phi[1], phi[0]), 3)

                    # fit the angle to the hypotheses distribution
                    likelihood = self.fit_angle_dist(rad)
                    # self.td_posterior = posterior(self.hypotheses.dpd, likelihood, smooth=True)
                    self.td_posterior = norm_dist(likelihood, smooth=True)
                    self.log(2, "received signal to evaluate target coordinates", coord, "after delay:", self.delta_t)

                    # self.layer_long_range_projection = {}
                    # self.layer_long_range_projection["MC"] = {"done": self.intention}

                elif "done" in self.long_range_projection:
                    if self.long_range_projection["done"] == "Seq":
                        self.log(3, "Received surprise signal from Sequence layer")
                        # self.cur_phi_magnitude = None
                    else:
                        self.intention_check = []
                        self.intention = None
                        self.delta_t = 0.

            elif self.higher_layer_prediction is not None:
                self.log(4, "higher layer projection:", self.higher_layer_prediction)
                higher_layer = copy(self.higher_layer_prediction)

                if self.hypotheses.dpd.shape[0] == higher_layer.shape[0]:
                    # self.td_posterior = posterior(self.hypotheses.dpd, higher_layer, smooth=True)
                    # self.td_posterior = norm_dist(higher_layer, smooth=True)
                    self.td_posterior = joint(self.hypotheses.dpd, higher_layer, smooth=True)
                else:
                    self.log(1, "Incompatible higher layer projection:", higher_layer.shape[0], "to", self.hypotheses.dpd.shape[0])
                    self.log(3, higher_layer)


    def bu_inference(self):
        """ Calculate the posterior for the sequence layer, based on evidence from
        predicted lower level activity.
        """
        # self.bu_posterior = norm_dist(self.likelihood, smooth=True)
        self.bu_posterior = posterior(self.hypotheses.dpd, self.likelihood, smooth=True)



    def extension(self):
        """ No extension in this layer. Primitives are fixed.

        Check if new best hypothesis should be found and if there is surprisal in the currently best_hypo,
        send the current coordinate to next layer as a "waypoint" if input was surprising. (segmentation by Zachs, etc)
        """

        if self.hypotheses is not None and self.hypotheses.dpd is not None:
            max_id = self.hypotheses.max()[1]
            cur_best_hypo = self.hypotheses.reps[max_id]
            # self.log(1, "cur_best_hypo:", cur_best_hypo)
            # self.log(1, "surprise?", self.PE)

            if self.intention_check is not None and len(self.intention_check) > 1:
                # fit coordinate to internal coordinate system
                # diff = self.intention[1] - self.intention[0]

                r, theta = self.fit_coordinate_system(self.intention_check[1])
                # r, theta = self.fit_coordinate_system(self.last_coordinate)
                coord = Coord(r, theta, self.isDrawing)
                # communicate current coordinate to next higher layer
                self.layer_evidence = None
                self.layer_new_hypo = [coord, copy(self.delta_t)]

                # if self.allow_follow_through and self.will_probably_reach_target:
                #     self.log(1, "Follow-through triggered! Will probably reach target:", coord, "in", self.delta_t * (1. + self.follow_through_factor))
                #     self.will_probably_reach_target = False
                #     self.prior_distance_to_target = None
                # else:
                self.log(2, "target:", self.intention_check[0], "evidence:", self.intention_check[1], "reached confirmed after", copy(self.delta_t), "seconds")
                self.intention_check = []
                self.delta_t = 0.

            elif len(self.intention_check) < 1 and\
                    self.lower_layer_evidence is not None and\
                    self.lower_layer_evidence[0] is not None and\
                    ((self.best_hypo is not None and self.PE.some_surprise()) or 
                        not self.isDrawing):

                # change current best hypo to new argmax of hypo dpds
                self.best_hypo = cur_best_hypo
                self.log(3, "best_hypo is:", self.best_hypo)

                # fit movement to internal coordinate system
                r, theta = self.fit_coordinate_system(self.last_coordinate)

                if self.isDrawing and self.step_counter > 1:
                    # stretch the movement over the number of steps perceived in-between
                    r *= self.step_counter
                # reset step-counter, for this step was recorded
                self.step_counter = 0

                coord = Coord(r, theta, self.isDrawing)

                # communicate current movement to next higher layer
                self.layer_evidence = None
                # bu_hypotheses = copy(self.hypotheses)
                # bu_hypotheses.dpd = self.bu_posterior
                self.layer_new_hypo = [coord, copy(self.delta_t)]
                self.log(3, "surprising percept:", self.PE)
                self.log(3, "surprising percept after", self.delta_t, "seconds. Sending movement to next higher layer!")

                self.delta_t = 0.

            elif len(self.intention_check) < 1 and self.best_hypo is None:
                # at the beginning of processing there may be no best hypo yet, so...

                # store current best hypothesis, since there is none at the moment
                self.best_hypo = cur_best_hypo
                self.log(3, "best_hypo is:", self.best_hypo)

                self.layer_evidence = None
            elif self.likelihood is not None:
                # if there is no surprisal we can at least inform about the current best hypo
                r, theta = self.fit_coordinate_system(self.last_coordinate)
                coord = Coord(r, theta, self.isDrawing)

                # bu_hypotheses = copy(self.hypotheses)
                # bu_hypotheses.dpd = self.bu_posterior
                self.layer_evidence = [coord, copy(self.delta_t)]
                self.log(3, "no surprisal:", self.PE)
                
                # NO delta_t RESET HERE!

        # on end of a sequence:
        if len(self.intention_check) < 1 and\
                self.lower_layer_evidence is not None and\
                self.lower_layer_evidence[1] is not None and\
                self.lower_layer_evidence[0] is None:
            # in case of only tempral information receiving, inform the next higher layer also of that
            self.log(2, "end of sequence")
            self.layer_evidence = [None, copy(self.delta_t)]
            self.delta_t = 0  # reset delta_t here, so that a new trajectory can begin



    def fit_coordinate_system(self, rel_coordinate):
        """ Fits the given coordinate to the internal RELATIVE coordinate system.

        Source: Russo and Bruce (1998) - Neurons in the Supplementary Eye Field of Rhesus Monkeys Code Visual Targets
        and Saccadic Eye Movements in an Oculocentric Coordinate System

        The surprisal detected here is probably only movement related, not concerning itself with
        stationary stimuli. Could fix this using an additional mapping to relative coordinate system and
        the stimulus deviations resulting in suprisal in there.

        Returns the fitted RELATIVE coordinate.
        """
        x = rel_coordinate[0]
        y = rel_coordinate[1]
        # convert to polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        # self.log(4, "cartesian:", x, y, "in polar:", r, theta)

        # # find closest polar coordinates in grid
        # grid_r = self.oculocentric_coordinates['r']
        # grid_theta = self.oculocentric_coordinates['theta']

        # r_coord_min = np.argmin(np.abs(grid_r[:] - r))
        # r_fit = round(grid_r[r_coord_min], 2)
        # theta_coord_min = np.argmin(np.abs(grid_theta[:] - theta))
        # theta_fit = round(grid_theta[theta_coord_min], 2)
        # # self.log(4, "polar fit r:", r_fit, "theta:", theta_fit)

        return r, theta


    def prediction(self):
        """ Decide on active inference on sensorimotor activity.
        """
        pass


    def receive_evidence(self, evidence):
        """ Overloaded method to receive only sensory data and no representations.
        """
        self.lower_layer_evidence = evidence

