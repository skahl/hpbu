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

""" function storage
Created on 27.01.2016

@author: skahl
"""

# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from datetime import datetime

import simplejson as json

import sys
import numpy as np
import math

import itertools

# from scipy.stats import entropy
from numpy.linalg import norm as np_norm
from numpy import expand_dims as np_expand_dims, atleast_1d as np_atleast_1d, dot as np_dot, sqrt as np_sqrt
from numpy import log as np_log, sum as np_sum, e as np_e, pi as np_pi, exp as np_exp, median as np_median
from numpy import abs as np_abs, mean as np_mean, var as np_var, max as np_max, clip as np_clip, argmax as np_argmax
from numpy import argmin as np_argmin
from scipy.stats import entropy as np_entropy

# riemann distance
# from pyriemann.estimation import Covariances
# from pyriemann.utils.distance import distance_riemann
# dtw
# from fastdtw import fastdtw, dtw
# from scipy.spatial import distance

from copy import copy, deepcopy
from time import time, sleep  # not needed in here, but in other modules
# from sklearn.covariance import oas

from collections import defaultdict, Counter

import traceback


class Logger(object):
    doLogging = 0
    color_dict = {
        'Black': "30",        # Black
        'Red': "31",          # Red
        'Green': "32",        # Green
        'Yellow': "33",       # Yellow
        'Blue': "34",         # Blue
        'Purple': "35",       # Purple
        'Cyan': "36",         # Cyan
        'White': "37"         # White
    }
    # filehandler = open(str(datetime.now().time()) + ".log", "w")

    def __init__(self, color, name):
        self.color = color
        self.name = name

    def log(self, verb, *string):
        if self.doLogging >= verb:
            s = " "
            _clr_b = "\x1b[0;" + self.color_dict[self.color] + ":40m"
            _clr_e = "\x1b[0m"
            # self.filehandler.write(self.name + " " + str(datetime.now().time()) + ": " + s.join(map(str, string)) + "\n")
            print(_clr_b, self.name, datetime.now().time(), s.join(map(str, string)), _clr_e)

    def error(self, *string):
        self.log(0, "ERROR:", *string)



class DataLogger(Logger):
    def __init__(self, name, idx=""):
        super(DataLogger, self).__init__('Black', name)
        try:
            self.filehandler = open(idx + 'precision_' + self.name + ".json", "w")
        except IOError as error:
            print(error)
            traceback.print_exc()

    def log(self, data):
        json_data = json.dumps([time(), data])
        try:
            print(json_data, file=self.filehandler)
        except IOError as error:
            print(error)
            traceback.print_exc()


def joint(A, B, smooth=True):
    """ Joint probability: P(A, B) = P(Ai) + P(Bi) / sum(P(Ai) + P(Bi))
    """
    _A = A[A[:, 1].argsort()]
    _B = B[B[:, 1].argsort()]
    joint = copy(_B)

    if smooth:
        add_one_smoothing = 0.005
        norming_factor = np_sum(_A[:, 0] + _B[:, 0] + add_one_smoothing)
        joint[:, 0] = (_A[:, 0] + _B[:, 0] + add_one_smoothing) / norming_factor

    else:
        joint[:, 0] = _A[:, 0] + _B[:, 0]
        joint[:, 0] = joint[:, 0] / np.sum(joint[:, 0])

    # print("joint probability:\n", A, "\n", B, "\n", joint)

    return joint



def posterior(prior, evidence, smooth=True):
        """ Calculate the posterior given the prior and the given dpd, normalized by a norming factor.
        """
        # P(H|A) = P(H)*P(A|H)/P(A)
        # P(A) = SUM_H(P(H,A)) = SUM_H(P(H)*P(A|H))
        if prior is not None and evidence is not None:

            prior = prior[prior[:, 1].argsort()]
            evidence = evidence[evidence[:, 1].argsort()]

            posterior = copy(prior)
            
            if smooth:
                add_one_smoothing = 0.005  # 1. / posterior.shape[0]
                norming_factor = np_sum(prior[:, 0] * evidence[:, 0] + add_one_smoothing)  # * posterior.shape[0]
                # calculate new posterior
                posterior[:, 0] = (prior[:, 0] * evidence[:, 0] + add_one_smoothing) / norming_factor
            else:
                norming_factor = np_sum(prior[:, 0] * evidence[:, 0])
                # calculate new posterior
                posterior[:, 0] = (prior[:, 0] * evidence[:, 0]) / norming_factor
            
            return posterior
        else:
            return None


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    div = np_norm(vector)
    if div == 0.0:
        return vector
    return vector / div


def dpd_diff(A, B):
    """ Calculate the difference between discrete probability distributions.
    """
    A = A[A[:, 1].argsort()]
    B = B[B[:, 1].argsort()]

    diff = copy(A)
    diff[:, 0] = np.abs(A[:, 0] - B[:, 0])
    # print("A:", A, "\nB:", B, "\ndiff:", diff)
    return diff


def dpd_equalize(A):
    """ Equalize the probability distribution.
    """
    if A.shape[0] > 0:
        one_by_len = 1. / A.shape[0]
        A[:, 0] = one_by_len
    return A


def set_hypothesis_P(dpd, idx, P):
        """ Introduce a specific probability of one of the representations.
        Normalize the distribution afterwards.
        """
        if idx < dpd.shape[0]:
            dpd[idx, 0] = P
        else:
            print("Error in set_hypothesis_P: idx not in dpd!")
        return dpd


def normalized(a, axis=-1, order=2):
    """
    http://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
    """
    l2 = np_atleast_1d(np_norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np_expand_dims(l2, axis)


def norm_dist(distribution, smooth=True):
    """ Normalize distribution, and apply add-one smoothing to leave
    unused probability space.
    """

    if smooth:
        add_one_smoothing = 0.001  # 1. / distribution.shape[0]
        norming_factor = np_sum(distribution[:, 0] + add_one_smoothing)   # * distribution.shape[0])
        distribution[:, 0] = (distribution[:, 0] + add_one_smoothing) / norming_factor
    else:
        distribution[:, 0] = distribution[:, 0] / np_sum(distribution[:, 0])
    return distribution


def soft_max(distribution):
    """ Calculate softmax for given distribution
    """
    sum_exps = np_sum(np_exp(distribution[:, 0]), axis=0)
    distribution[:, 0] = np_exp(distribution[:, 0]) / sum_exps
    return distribution


def discretize_coords_sequence(S, d=3):
    """ Discretize sequence as an alphabet, as described in Mateos et al., 2017
    http://doi.org/10.1063/1.4999613

    Parameters:
    d = embedding dimension (number of bit taken to create a word)
    tau = delay step size, fixed to 1

    Returns: Discretized binary sequence, Counter of word frequencies
    """
    # create binary sequence
    binary_S = np.zeros((len(S) - 1,))
    for idx, s in enumerate(S):
        if idx < binary_S.shape[0]:
            binary_S[idx] = 0 if s <= S[idx + 1] else 1

    if binary_S.shape[0] - (d - 1) > 0:
        # count word frequencies for S
        P_S = Counter()
        for w in range(binary_S.shape[0] - (d - 1)):
            word = tuple(binary_S[w:w + d])
            P_S[word] += 1

        return binary_S, P_S
    else:
        print("Discretize_sequence: Sequence to small for word size d!")
        return binary_S, {}


def kl_div(P, Q):
    # optimized KLD
    kld = np_sum(_p * np_log(_p / _q) for _p, _q in zip(P, Q) if _p != 0)
    return kld if kld != np.inf else 0.


# @profile
def JSD(sequence_P, P_P, sequence_Q, P_Q):
    """ alphabetic Jensen-Shannon Distance, calculated from discretized timeseries
    data in a combined probability space.

    Receives discretized sequences and their word counters using the discretize_sequence method.

    P:=prior, Q:=posterior
    See: https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    Together with: http://doi.org/10.1063/1.4999613
    """

    if len(P_P) > 0 and len(P_Q) > 0:

        # create alphabet dictionary
        # combine words for equal DPD lengths
        P_combined = P_P + P_Q
        sorted_keys = sorted(list(P_combined.keys()))

        # Unknown key returns 0! :)
        P_dpd = np.array([P_P[w] for w in sorted_keys])
        Q_dpd = np.array([P_Q[w] for w in sorted_keys])
        # norm probability distributions
        norm_P = np_dot(P_dpd, P_dpd)
        _P = P_dpd / norm_P
        norm_Q = np_dot(Q_dpd, Q_dpd)
        _Q = Q_dpd / norm_Q
        # calculate Jensen-Shannon Distance
        _M = 0.5 * (_P + _Q)
        js_distance = np_sqrt(0.5 * (kl_div(_P, _M) + kl_div(_Q, _M)))

        return js_distance
    print("JSD: Alphabets must contain at least one word!")
    return 1.0


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np_median(arr)
    return np_median(np_abs(arr - med))


def approach_goal(y, dy, goal):
    # based on Hoffmann (2009) but instead of
    # avoiding obstacles, we approach a goal

    gamma = 10  # 1/5
    beta = 1 / np.pi
    p = np.zeros(2)

    if np_norm(dy) > 1e-5:
        # calculate current heading
        phi_dy = np.arctan2(dy[1], dy[0])
        # calc vector to goal
        goal_vec = goal - y
        phi_goal = np.arctan2(goal_vec[1], goal_vec[0])

        # angle diff
        phi = phi_goal - phi_dy

        # tuned inverse sigmoid to create force towards goal
        dphi = gamma * phi * np_exp(-beta * np_abs(phi))
        pval = goal_vec * dphi
        # print("force vector:", pval, dy)

        p += pval
    return p


def kalman_gain(F, pi, oldK=None, gain_gain=0.5):
    """ Calculate the Kalman gain from free-energy and precision of the layer.

    Examples:
    low pi => steep response in K to small increases in F, max K = 1.0 (strong prediction-error influence)
    high pi => slow response in K to strong increases in F, max K = 0.5 (mostly preserved prior)
    """
    # pi = pi if pi < 5. else 5.  # limit precision variance
    K = F / (F + pi)  # gain factor from free-energy and precision

    if oldK is not None:
        # filter the Kalman Gain over time using a "gain gain" ;)
        K, _ = kalman_filter(oldK, K, gain_gain)

    return K




def kalman_filter(prior, observation, K):
    """ Calculate two iteration kalman filter with given measurement variance.
    The higher the variance the more likely will the prior be preserved.

    Examples:
    low pi => steep response in K to small increases in F, max K = 1.0 (strong prediction-error influence)
    high pi => slow response in K to strong increases in F, max K = 0.5 (mostly preserved prior)

    Returns the posterior estimate.
    """

    # calculate precision from ascending prediction-error
    prediction_error = observation - prior

    xhat = prior + K * (prediction_error)  # posterior estimate

    return xhat, prediction_error


def multisignal_kalman_filter(prior, observation_gain_tuples):
    for gain, obs in observation_gain_tuples:
        xhat = prior + gain * (obs - prior)
        prior = xhat

    return prior


def inhibition_belief_update(P, Q, K, personmodel_influence=None):
    """ Calculate kalman filter with given free_energy and precision for gain factor.
    The higher the precision the more likely will the prior be preserved.

    P = Prior, Q = Posterior, K = Kalman gain

    Returns the update belief estimate.
    """
    # sort first
    # H = H[H[:, 1].argsort()]
    P = P[P[:, 1].argsort()]
    Q = Q[Q[:, 1].argsort()]
    H = copy(P)

    if personmodel_influence is not None and len(personmodel_influence.get_agent_influence_indices()) > 0:
        # prepare observation_gain tuples for multisignal_kalman_filter update
        yous = personmodel_influence.get_agent_influence_indices()
        observation_gain_tuples = []
        if yous is not None and len(yous) > 0:
            knowledge = copy(P[:, 0])
            # knowledge[:] = 0.0001
            # add an observation gain tuple for every prior knowledge
            for agent_id, gain in personmodel_influence.agents.items():
                if agent_id in yous:
                    # create a complete distribution for each agent
                    agent_knowledge = copy(knowledge)
                    P_per_seq = 1 / len(yous[agent_id]) # (1 - K) # limit total influence to 1 
                    agent_knowledge[yous[agent_id]] = P_per_seq
                    agent_knowledge /= np_sum(agent_knowledge)
                    observation_gain_tuples.append((gain * 0.1, agent_knowledge))

            # add observation gain tuple of current posterior to observation_gain_tuples
            observation_gain_tuples.append((K, Q[:, 0]))

            # multisignal kalman update
            H[:, 0] = multisignal_kalman_filter(P[:, 0], observation_gain_tuples)
            H = norm_dist(H, smooth=True)

            # print("\nprior:", P, "\nobservation:", Q)
            # print("resulting updated beliefs:", H)
        else:
            sys.exit(1)

    else:
        # simple kalman update
        H[:, 0], _ = kalman_filter(P[:, 0], Q[:, 0], K)
        H = norm_dist(H, smooth=True)

    return H



def precision(PE):
    """ Calculate precision as the inverse variance of the updated prediction error.

    return updated precision and updated average_free_energy
    """
    with np.errstate(all='raise'):
        try:
            variance = np_var(PE)  # np_var(PE, ddof=1)  # mad(PE)
            variance = variance if variance > 0.00001 else 0.00001 # so log(var) should max at -5
            pi = np_log(1. / variance)  # should max at 5
            new_precision = 1/(1+np.exp(-(pi - 2.5))) # should be about max. 1

            return new_precision  # , variance
        except Exception as e:
            raise Exception("RuntimeWarning in precision(PE):", str(e), "PE:", PE) from e


def prediction_error(P, Q):
    """ Calculate the size of the prediction error and its variance
    """
    pe = Q - P
    # PE = np_sqrt(np_dot(pe, pe))
    # PE = kl_div(P, Q)
    # print("PE:", PE)
    return pe



def free_energy(P, Q):
    """ see Friston (2012) 
    My interpretation in differences between perception and active inference:
    - In perception, the posterior is the new posterior after perception.
    - In active inference, the posterior is the expected/intended distribution, with the
        new posterior after perception as the prior.
    """
    with np.errstate(all='raise'):
        try:
            # PE = Q - P  # prediciton-error
            surprise = np_entropy(P) #  if np_dot(PE, PE) > 0 else 0.
            surprise = surprise if abs(surprise) != np.inf else 0.
            cross_entropy = np_entropy(P, Q)
            cross_entropy = cross_entropy if abs(cross_entropy) != np.inf else 0.
            c_e = 1/(1+np.exp(-4*(cross_entropy - 0.5)))  # should be maxing out at 1
            F = surprise + c_e

            return F, surprise, c_e
        except Exception as e:
            raise Exception("RuntimeWarning in free_energy(P,Q):", str(e), "P:", P, "Q:", Q)


def time_precision(mu, x, pi):
    # fit time 'x' to predicted time 'mu' with precision 'pi'
    # TODO: the scaling of sigma needs to be adapted to SoA empirical evidence
    diff = np_sqrt(np_dot(x - mu, x - mu))
    sig = 2 * (1 - pi) if pi < 1. else 0.1
    return np_e ** (- (diff ** 2) / (sig ** 2)) + 0.001


def gaussian(x, mu, sig):
    """ Not normally distributed!
    """
    diff = x - mu
    return np_exp((-np_sqrt(np_dot(diff, diff)) ** 2) / (2 * sig ** 2))


def distribution(mu, sig, size):
    return [[gaussian(i, mu, sig), i] for i in range(size)]


def extend_sequence(sequence, item):
    """ Simply add a new item to the given sequence.
    Simple version of extend_distribution_sequence, since not using distributions here makes
    the equalization of distributions in sequence irrelevant.
    """
    sequence.append(item)
    return sequence

# @profile
def diff_sequences(seq_a, seq_b):
    """ Calculate the Levenshtein distance and clip the gaussian likelihood of the result.
    By that a similarity measure is returned.
    """
    source = seq_a.seq
    target = seq_b.seq[:len(source)]  # compare only parts of equal size

    s_range = range(len(source) + 1)
    t_range = range(len(target) + 1)
    matrix = np.array([[(i if j == 0 else j) for j in t_range] for i in s_range])
    for i in s_range[1:]:
        for j in t_range[1:]:
            del_dist = matrix[i - 1, j] + 1  # delete distance
            ins_dist = matrix[i, j - 1] + 1  # insertion distance

            # TODO: substitute for actual motor primitive __sub__ function
            dist = np_abs(target[j - 1].theta - source[i - 1].theta)
            sub_trans_cost = dist if dist < 0.5 and target[j - 1].drawing == source[i - 1].drawing else 1
            sub_dist = matrix[i - 1, j - 1] + sub_trans_cost  # substitution distance

            # Choose option that produces smallest distance
            matrix[i, j] = min(del_dist, ins_dist, sub_dist)

    _dist = matrix[len(source), len(target)]
    # print(_dist)
    _gauss = gaussian(_dist, 0., 24)

    return np_clip(_gauss, 0.0001, 1.)


def diff_levenshtein(source, target):
    """ Calculate the Levenshtein distance and clip the gaussian likelihood of the result.
    By that a similarity measure is returned.
    """
    target = target[:len(source)]  # compare only parts of equal size

    s_range = range(len(source) + 1)
    t_range = range(len(target) + 1)
    matrix = np.array([[(i if j == 0 else j) for j in t_range] for i in s_range])
    for i in s_range[1:]:
        for j in t_range[1:]:
            del_dist = matrix[i - 1, j] + 1  # delete distance
            ins_dist = matrix[i, j - 1] + 1  # insertion distance

            sub_trans_cost = 0 if source[i - 1] == target[j - 1] else 1
            sub_dist = matrix[i - 1, j - 1] + sub_trans_cost  # substitution distance

            # Choose option that produces smallest distance
            matrix[i, j] = min(del_dist, ins_dist, sub_dist)

    _dist = matrix[len(source), len(target)]
    _gauss = gaussian(_dist, 0., 4)

    return np_clip(_gauss, 0.0001, 1.)


def mixture_experts(S, C, matrix, smooth=True):
    """ From a mixture of experts C calculate the S probabilities.
    """
    post = copy(S)

    post[:, 0] = np.array([np_sum([s_P * c_P * len(matrix[c_id]) for c_P, c_id in C if s_id in matrix[c_id]]) for s_P, s_id in S[:]])
    post = norm_dist(post, smooth=smooth)
    # sum_bu = np_sum(post[:, 0])
    # if sum_bu > 1.1 or sum_bu < 0.9:
    #     print("mixture_experts not normalized:", sum_bu)
    #     print("S:\n", S, "C:\n", C, "matrix:\n", matrix)

    return post


def soft_evidence(prior_dpd, evidence, LH_C, smooth=True):
    """ Calculate soft evidence (new posterior) for prior, using the 'all things considered' method.
    
    See: Darwiche "Modeling and Reasoning with Bayes" - Chapter 3.6

    Requires updated P(C|S) likelihood matrix from cluster layer.
    """
    # posterior = copy(prior_dpd)

    # if posterior is not None:
    #     # iterate over cluster probabilities p:
    #     # for c_p in posterior:
    #     #     # all sequences in cluster
    #     #     seq_ids = LH_C[c_p[1]]

    #     #     # until 26.07.2018
    #     #     # p[0] = np.sum([evidence.dpd[evidence.reps[ID].dpd_idx, 0] for ID in seq_ids])

    #     #     # after 04.08.2018
    #     #     # \sum_si p(C|si) * p(C) with si \in clustered_sequences

    #     #     c_p[0] = np_sum([c_p[0] * evidence.dpd[evidence.reps[ID].dpd_idx, 0] for ID in seq_ids])

    #     posterior[:, 0] = np.array([np_sum([c_p * evidence.dpd[evidence.reps[s_id].dpd_idx, 0] for s_id in LH_C[c_id]]) for c_p, c_id in posterior])
    #     # posterior = norm_dist(posterior, smooth=True)
    #     posterior = norm_dist(posterior, smooth=False)

    # soft-evidence with external storage of previous distribution

    if prior_dpd is None:
        prior_dpd = np.zeros((len(LH_C), 2))
        prior_dpd[:, 1] = list(LH_C.keys())
        prior_dpd[:, 0] = 1/len(LH_C)
        print("soft_evidence: prior_dpd was None")

    P_E = {s_id: P for P, s_id in evidence}

    posterior = copy(prior_dpd)
    
    posterior[:, 0] = np.array([np_sum([P_E[s_id] for s_id in LH_C[c_id]]) * c_p for c_p, c_id in prior_dpd])
    posterior = norm_dist(posterior, smooth=smooth)
    # print("posterior:\n", posterior)

    return posterior



def define_coordinate_system(r=20, theta=25, show=False):
    """ Calcuate a mesh of a polar coordinate system with theta angular resolution
    and r radial resolution.
    Polar coordinates can be described by radius and theta-angle values.
    Radial resolution follows a logarithmic scale, decreasing resolution to the edges.
    """
    _r = np.geomspace(1, 400, r)
    _theta = np.linspace(-np_pi, np_pi, theta)

    return {'r': _r, 'theta': _theta}


def rolling_window(a, size):
    """ Rolling window function for subsequence matching
    """
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)



def subsequence_matching(ar, sub_ar):
    """ Subsequence matching using a rolling window function to find
    subsequence within sequences, looking at all possible windows.
    """
    return np.all(rolling_window(ar, len(sub_ar)) == sub_ar, axis=1)



def find_signaling_candidate(cluster, contrastor_candidate):
    """ Find a candidate in cluster that is most different
    from the contrastor_candidate.
    Iterate over all sequences in cluster.

    Return the found candidate.
    """
    candidate = None
    len_cluster = len(cluster.seqs)
    if len_cluster > 1:
        for seq in cluster.seqs:
            sim = diff_sequences(seq, contrastor_candidate)
            print("signaling similarity:", sim, seq, "to", contrastor_candidate)
            # remember new potential candidate if the newly calculated similarity is lower than before
            if candidate is None:
                candidate = [sim, seq]
            elif candidate is not None and candidate[0] > sim:
                candidate = [sim, seq]
        return candidate
    elif len_cluster > 0:
        return cluster.seqs[0]
    else:
        return None


def within_cluster_similarity_statistics(representations):
    """ Calculate the sequence similarities within a cluster.

    Return the similarity matrix.
    """
    lenrep = len(representations)

    similarities = np.ones((lenrep, lenrep, 3))
    for j in range(lenrep):
        for k in range(j + 1, lenrep):
            # calculate once
            sim = diff_sequences(representations[j], representations[k])
            # but fill both triangles of the matrix
            similarities[j, k, :] = [representations[j].id, representations[k].id, sim]
            similarities[k, j, :] = [representations[k].id, representations[j].id, sim]

    average_rep_sim = np_mean(similarities[:, :, 2])
    var_rep_sim = np_var(similarities[:, :, 2])

    return similarities, average_rep_sim, var_rep_sim



def inter_cluster_similarity_statistics(cluster_a, cluster_b):
    """ Calculate the similarities between the two cluster's sequences.

    Return the matrix, mean distance and variance.
    """
    seqs_a = cluster_a.seqs
    seqs_b = cluster_b.seqs
    lenrep_a = len(cluster_a.seqs)
    lenrep_b = len(cluster_b.seqs)

    similarities = np.ones((lenrep_a, lenrep_b, 3))
    for j in range(lenrep_a):
        for k in range(lenrep_b):  # sadly have to compare all of them
            # calculate free-energy between sequences
            sim = diff_sequences(seqs_a[j], seqs_b[k])
            similarities[j, k, :] = [seqs_a[j].id, seqs_b[k].id, sim]

    average_cluster_sim = np_mean(similarities[:, :, 2])
    var_cluster_sim = np_var(similarities[:, :, 2])

    return similarities, average_cluster_sim, var_cluster_sim


