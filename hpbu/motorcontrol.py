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

""" MotorControl
Created on 13.04.2018

@author: skahl
"""
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from .layer import *


class MotorControl(Layer):
    """ Input-output motor control layer class for predictive processing hierarchy,
    specialized for collecting lower level sensory activity in sequence
    of occurrence and for producing same sensorymotor activity through active inference.
    """


    def __init__(self, name):
        super(MotorControl, self).__init__(name)
        self.type = 'MotorControl'
        self.reset()



    def reset(self):
        super(MotorControl, self).reset()
        # motor control system
        self.alpha = 50  # 25
        self.beta = 50  # self.alpha / 2  # 6.75  # 4

        self.phi_look_ahead = None
        self.joint_vector = np.array([0., 0.])
        self.last_joint_vector = np.array([0., 0.])
        self.joint_velocity = np.array([0., 0.])
        self.rel_movement = None

        # drawing under time pressure
        self.isDrawing = False
        self.sim_step = 1e-4  # 0.001
        self.delay_step = 1e-1
        self.target_precision = None
        self.step_counter = 0
        self.simulation_step_history = deque(maxlen=5)
        # storage
        self.distance = 0.
        self.distances = deque(maxlen=3)
        self.positions = []

        print("(Re-)initialized layer", self.name)



    def gen_primitives(self):
        # prep angle distribution
        # distribution based on angular primitive resolution of 360/18 = 20 degrees, or ~0,349 radians

        self.hypotheses.dpd = np.zeros((20, 2))
        self.hypotheses.dpd[:10, :] = np.array([[1. / 20, round(-np.pi * (10 - i) / 10, 2)] for i in range(0, 10)])
        self.hypotheses.dpd[10:, :] = np.array([[1. / 20, round(np.pi * i / 10, 2)] for i in range(0, 10)])
        self.hypotheses.reps = {round(self.hypotheses.dpd[i, 1], 2): Representation(round(self.hypotheses.dpd[i, 1], 2)) for i in range(0, 20)}

        for idx, dpd_pair in enumerate(self.hypotheses.dpd):
            self.hypotheses.reps[dpd_pair[1]].dpd_idx = idx
        self.log(2, "Generated primitives:\n", self.hypotheses.reps.keys())


    # these primitves are not used atm, as they are for a pole-balancing scenario with only two movement directions
    def gen_lr_primitives(self):
        self.hypotheses.dpd = np.zeros((2, 2))
        self.hypotheses.dpd[:, :] = np.array([[0.5, round(-np.pi, 2)], [0.5, round(0.000, 2)]])
        self.hypotheses.reps = {self.hypotheses.dpd[i, 1]: Representation(round(self.hypotheses.dpd[i, 1], 2)) for i in range(0, 2)}

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

            # if np_norm(self.joint_vector - self.lower_layer_evidence) > 0:
            #     self.log(1, "WTF", self.joint_vector, self.lower_layer_evidence)

            self.joint_vector = copy(self.lower_layer_evidence)  # joint_vector also acts as control mechanism for updates
            # self.log(1, "lower_layer_evidence => joint vector:", self.joint_vector)
            self.last_joint_vector = copy(self.joint_vector)

            phi = None
            if self.intention is not None:
                phi = self.intention - self.joint_vector
                self.log(3, "distance:", self.distance, "to intention", self.intention, "joint", self.joint_vector, "with target-precision:", self.target_precision)
            elif self.rel_movement is not None:
                phi = self.rel_movement

            if phi is not None:
                prop_angle = np.arctan2(phi[1], phi[0])
                self.distance = np.linalg.norm(phi)
                self.distances.append(copy(self.distance))

                # calculate likelihood of angle
                self.likelihood = self.fit_angle_dist(prop_angle)

            # better stop condition
            if self.intention is not None and self.distance <= self.target_precision and self.step_counter == 0:
                # print("stopping condition")
                self.joint_vector = copy(self.intention)
                self.last_joint_vector = copy(self.intention)
                

    def fit_angle_dist(self, rad):
        """ Fit the given radians to the distribution of radians.
        """
        dpd = self.hypotheses.dpd
        lh = np.array([[gaussian(rad, i, 0.1), i] for i in dpd[:, 1]])  # sigma = 0.17
        return lh  # norm_dist(lh)



    def td_inference(self):
        """ Integrate influence from higher layers or long range projections.
        """
        # angle possibilities
        if self.hypotheses.dpd is not None and len(self.hypotheses.dpd) > 0:
            if self.long_range_projection is not None:
                """ Here we receive additional information like higher layer signals for intention to act
                or a signal when a sequence is done.
                The here received coordinates will define the control problem that needs to be solved by the
                dampened spring system.
                """
                self.log(3, "received long-range projection:", self.long_range_projection)

                # for normal writing trajectories, this is the normal path. not the intention path 
                if "goal" in self.long_range_projection:
                    LRP = self.long_range_projection["goal"]

                    if self.intention is None:
                        self.intention = copy(self.joint_vector)

                    r = LRP[0].r
                    theta = LRP[0].theta
                    self.isDrawing = LRP[0].drawing

                    rel_intention = np.array([r * np.cos(theta), r * np.sin(theta)])
                    self.intention += rel_intention
                    self.log(3, "New intention coordinate:", self.intention) # , "polar:", LRP[0], "cartesian:", rel_intention, "subgoal delay:", LRP[1])

                    if self.joint_vector is not None:
                        phi = self.intention - self.joint_vector
                    else:
                        phi = self.intention
                    prop_angle = np.arctan2(phi[1], phi[0])
                    self.distance = np_norm(phi)
                    self.distances.append(copy(self.distance))

                    # recalculate the necessary target precision, based on time pressure
                    self.delay_step = LRP[1]  # step delay precalculated in sequences

                    delay = self.delay_step if self.delay_step > 0.0001 else 0.0001  # total time available
                    delay = delay if delay <= 0.1 else 0.1
                    delay = np_abs(np_log(1 / delay))
                    distance = self.distance if self.distance <= 150 else 150
                    distance = distance if self.distance > 1 else 2
                    precisiontime_factor = np_log(distance) * delay
                    self.target_precision = precisiontime_factor if precisiontime_factor >= 2.0 else 2.0

                    self.step_counter = 0
                    # self.stabilizing = False
                    self.log(3, "step delay:", self.delay_step, "distance:", self.distance)

                    likelihood = self.fit_angle_dist(prop_angle)
                    # self.td_posterior = posterior(self.hypotheses.dpd, likelihood, smooth=True)
                    self.td_posterior = norm_dist(likelihood, smooth=True)
                    self.last_joint_acceleration = None
                    # reset rel movement
                    self.rel_movement = None

                # as of now this path is only used for obstacle simulation in future model implementations
                if "intention" in self.long_range_projection and self.intention is None:
                    abs_look_ahead = self.long_range_projection["intention"]
                    self.intention = copy(self.joint_vector)
                    self.intention += abs_look_ahead
                    self.isDrawing = True

                    if self.joint_vector is not None:
                        self.phi_look_ahead = (self.intention - self.joint_vector) 
                    else:
                        self.phi_look_ahead = self.intention

                    prop_angle = np.arctan2(self.phi_look_ahead[1], self.phi_look_ahead[0])
                    self.distance = np_norm(self.phi_look_ahead)
                    self.distances.append(copy(self.distance))

                    delay = 2  # TODO: default delay for obstacle simulation
                    delay = np_abs(np_log(1 / delay))
                    distance = self.distance if self.distance <= 150 else 150
                    distance = distance if self.distance > 1 else 2
                    precisiontime_factor = np_log(self.distance) * delay
                    self.target_precision = precisiontime_factor if precisiontime_factor >= 2.0 else 2.0

                    self.step_counter = 0
                    # self.stabilizing = False

                    self.log(2, "new intended goal:", abs_look_ahead, "distance:", self.distance)
                    self.log(2, "target precision:", self.target_precision)

                    likelihood = self.fit_angle_dist(prop_angle)
                    # self.td_posterior = posterior(self.hypotheses.dpd, likelihood, smooth=True)
                    self.td_posterior = norm_dist(likelihood, smooth=True)
                    self.last_joint_acceleration = None
                    # reset rel movement
                    self.rel_movement = None

                if "look_ahead_goal" in self.long_range_projection:
                    LRP = self.long_range_projection["look_ahead_goal"]

                    abs_look_ahead = copy(self.intention)
                    if type(LRP) is list:
                        for rel_look_ahead_step in LRP:
                            r = rel_look_ahead_step.r
                            theta = rel_look_ahead_step.theta
                            abs_look_ahead += np.array([r * np.cos(theta), r * np.sin(theta)])
                    elif LRP is not None:
                        r = LRP.r
                        theta = LRP.theta
                        abs_look_ahead += np.array([r * np.cos(theta), r * np.sin(theta)])

                    # set look ahead goal
                    if self.joint_vector is not None:
                        self.phi_look_ahead = abs_look_ahead - self.joint_vector
                    else:
                        self.phi_look_ahead = abs_look_ahead

                    self.log(2, "New look-ahead goal:", abs_look_ahead) # , "distance:", np_norm(self.phi_look_ahead))

                if "done" in self.long_range_projection:
                    self.layer_prediction = ["done", False]
                    self.intention = None
                    # reset position on canvas for new drawing
                    self.joint_vector = np.array([0., 0.])
                    self.last_joint_vector = np.array([0., 0.])
                    self.joint_velocity = np.array([0., 0.])
                    self.log(0, "resetting joint vector")

            elif self.higher_layer_prediction is not None:
                self.log(4, "higher layer projection:", self.higher_layer_prediction)
                higher_layer = copy(self.higher_layer_prediction)

                if self.hypotheses.dpd.shape[0] == higher_layer.shape[0]:
                    self.td_posterior = joint(self.hypotheses.dpd, higher_layer, smooth=True)
                    # self.td_posterior = norm_dist(higher_layer, smooth=True)
                else:
                    self.log(1, "Incompatible higher layer projection:", higher_layer.shape[0], "to", self.hypotheses.dpd.shape[0])
                    self.log(3, higher_layer)




    def bu_inference(self):
        """ Calculate the posterior for the sequence layer, based on evidence from
        predicted lower level activity.
        """
        self.bu_posterior = norm_dist(self.likelihood, smooth=True)
        # self.bu_posterior = posterior(self.hypotheses.dpd, self.likelihood, smooth=True)



    def extension(self):
        pass



    def prediction(self):
        """ Motor execution output from a "driven overdampened harmonic oscillator"
        Active inference using the weighted average error-minimizing motor primitive.
        """
        target_precision = 2  # self.target_precision

        if self.intention is not None and self.joint_vector is not None and target_precision is not None:
            self.log(3, "starting movement simulation to bridge the distance:", self.distance,
                "from", self.joint_vector, "to", self.intention, "drawn:", self.isDrawing)
            last_relevant_pos = copy(self.joint_vector)
            last_joint_vector = copy(self.joint_vector)

            running_avg_sim_steps = np_mean(self.simulation_step_history) if len(self.simulation_step_history) > 4 else 100
            while self.distance > target_precision and self.step_counter < running_avg_sim_steps: # and (self.step_counter < 4 or self.distance <= np_mean(self.distances)):

                if self.isDrawing:
                    # max angle for one-step motor control, not applicable if multiple steps are simulated!
                    # angle_idx = np_argmax(self.bu_posterior[:, 0])
                    # best_angle = self.hypotheses.max()[1]  # self.hypotheses.dpd[angle_idx, 1]
                    # self.log(3, "selected angle is", best_angle, "target area:", self.target_precision)
                    # rel_move = np.array([np.cos(best_angle), np.sin(best_angle)])

                    # if drawing, stepwise approach the intention
                    goal_force = approach_goal(self.joint_vector, self.joint_velocity, self.intention)
                    joint_acceleration = self.alpha * (self.beta * self.phi_look_ahead - self.joint_velocity) + goal_force

                    # integrate acceleration
                    self.joint_velocity += joint_acceleration * self.sim_step
                    # integrate velocity
                    # self.rel_movement += self.joint_velocity * self.sim_step  # remember relative movement only
                    self.joint_vector += self.joint_velocity * self.sim_step
                    self.log(3, "simulated move to:", self.joint_vector)
                    
                else:
                    # not drawing, just jump to the intention
                    # self.rel_movement = self.intention - self.joint_vector
                    self.joint_vector = self.intention
                    self.joint_velocity = 0
                    self.log(0, "non-drawn jump to:", self.joint_vector)

                self.step_counter += 1
                # store only relevant movements >= 1
                # np.linalg.norm(self.joint_vector - last_relevant_pos)
                sampling_frequency = 0.008 # 0.008 # 0.003
                dist_from_start = (self.step_counter * self.sim_step) % sampling_frequency

                if not self.isDrawing or (dist_from_start < 0.0001 and dist_from_start > -0.0001): 
                    # self.log(1, "saving new step after distance of:", dist_from_start)
                    # store new position
                    self.positions.append([copy(self.joint_vector), self.isDrawing])

                    # in case of simulated movement only
                    phi = self.intention - self.joint_vector
                    self.distance = np.linalg.norm(phi)
                    self.distances.append(copy(self.distance))

                    # remember last relevant position
                    last_relevant_pos = copy(self.joint_vector)
                
                running_avg_sim_steps = np_mean(self.simulation_step_history) if len(self.simulation_step_history) > 4 else 100
                # print(self.step_counter, running_avg_sim_steps)
            

            # check if without moving we are close enough
            if self.distance <= target_precision and self.step_counter == 0:
                # just jump to the intention
                # self.rel_movement = self.intention - self.joint_vector
                phi = self.intention - self.joint_vector
                self.joint_vector += phi / 2  # jump only so far, decreasing jumping artifacts...
                self.log(2, "non-moving jump to:", self.joint_vector)
                self.step_counter += 1

            # send joint positions
            if self.step_counter > 0:

                # truly act out the motion
                self.layer_prediction = [copy(self.positions), self.delay_step]

                # print(self.joint_vector, last_joint_vector)
                self.rel_movement = self.joint_vector - last_joint_vector
                
                control_time = self.step_counter * self.sim_step
                # self.log(1, "joint moved by:", self.rel_movement)
                self.log(2, "joint is close enough to intention:", self.intention, "distance:", self.distance, "steps:", self.step_counter, "time:", control_time)

                # not sending intention but only its visually similar counterpart
                self.layer_long_range_projection = {"Vision": {"confirm": [copy(self.rel_movement), self.delay_step, self.isDrawing]}}
                self.intention = None
                self.isDrawing = False
                self.positions = []

                # remember number of necessary simulated steps
                if self.isDrawing:
                    self.simulation_step_history.append(copy(self.step_counter))
                self.step_counter = 0
            # elif self.step_counter > 0:
            #     self.log(1, "joint still not close enough:", self.distance, self.target_precision)




    def receive_evidence(self, evidence):
        """ Overloaded method to receive only sensory data and no representations.
        """
        self.lower_layer_evidence = evidence

