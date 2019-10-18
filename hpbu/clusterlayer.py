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

""" ClusterLayer
Created on 15.08.2017

@author: skahl
"""
# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

from .layer import *

from random import choice

from sklearn.cluster import affinity_propagation


class ClusterLayer(Layer):
    """ Layer class for predictive processing hierarchy,
    specialized for clustering layers that collect similar sequences to form clusters
    and learn prototypes for each cluster.
    """


    def __init__(self, name):
        super(ClusterLayer, self).__init__(name)
        self.type = 'Cluster'
        self.reset()



    def reset(self):
        super(ClusterLayer, self).reset()
        # cluster layer likelihood P(C|S)
        self.layer_LH = None

        self.seq_intention = None  # in case of very specific selection of sequence production

        self.production_candidate = None  # store identified candidate for production
        self.signaling_distractor = None

        # store last surprisal time
        self.surprise_received = False
        self.last_surprise_time = 0.
        self.last_production_time = 0.
        # store maximum cluster from likelihood
        self.best_hypo = None
        self.avg_inter_cluster_sim = 0.
        self.var_inter_cluster_sim = 0.
        # store special sequences
        self.lost_sequences = []
        # store system sense of agency estimate
        self.self_estimate = 0.
        # remember if during this run a new sequence was learned, making clustering necessary
        self.new_sequences_learned = False

        for cluster in self.hypotheses.reps.values():
            # recalculate the cluster prototype
            cluster.find_prototype()

        print("(Re-)initialized layer", self.name)



    def __str__(self):
        return "Layer '" + self.name + "', type:" + self.type



    def finalize(self):
        """ finalize gets called upon the end of the hierarchy main loop
        So for clustering during training, re-cluster at the end of the training session.
        """
        if self.params["self_supervised"] and self.new_sequences_learned:
            self.log(0, "performing expensive '''night cycle''' update, reclustering...")
            self.recalculate_clusters_with_new_hypo()
            self.cluster_statistics()



    def print_out(self):

        _str_ = self.name
        _str_ += "\nhypotheses ("
        _str_ += str(len(self.hypotheses.reps))
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
                for seq in rep.seqs:
                    if seq.id not in self.lost_sequences:
                        self.layer_LH[cl_id].append(seq.id)

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

                # for external selection of sequences for production, look for intention
                if "seq_intention" in self.long_range_projection:
                    self.seq_intention = self.long_range_projection["seq_intention"]

                # get idx of intended hypothesis:
                lrp_idx = self.hypotheses.reps[lrp].dpd_idx

                # create motor intention
                # self.hypotheses.equalize()
                avg_P = np_mean(self.hypotheses.dpd[:, 0])
                var_P = 0.2
                critical_intention = avg_P + var_P  # only +1 var
                self.log(1, "setting intention", lrp, "P at critical:", critical_intention)

                # idx from rep id
                self.td_posterior = dpd_equalize(self.hypotheses.dpd)
                self.td_posterior = set_hypothesis_P(self.td_posterior, lrp_idx, critical_intention)
                self.td_posterior = norm_dist(self.td_posterior, smooth=True)
                self.intention = lrp
                # print("projected id:", lrp, self.td_posterior[lrp_idx])

            if "signaling_distractor" in self.long_range_projection:
                self.signaling_distractor = self.long_range_projection["signaling_distractor"]

            if "surprise" in self.long_range_projection:
                self.log(1, "Received delay-surprise signal from level:", self.long_range_projection["surprise"])
                self.surprise_received = True

            # if successful intention production was communicated via long range projection:
            if "done" in self.long_range_projection:
                if self.production_candidate == self.long_range_projection["done"]:
                    self.log(1, "intention production of cluster", self.intention, "was finished!")
                    self.long_range_projection = {}
                    self.long_range_projection["Realizations"] = {"done", self.intention}
                    # Realizations level needs important final information
                    self.intention = None
                    self.production_candidate = None
                    self.surprise_received = True
                    self.seq_intention = None

        # elif self.higher_layer_prediction is not None:
        #     self.log(4, "higher layer projection:", self.higher_layer_prediction)
        #     higher_layer = copy(self.higher_layer_prediction)
        #     # P = higher_layer[0]
        #     # matrix = higher_layer[1]

        #     # self.td_posterior = higher_layer
        #     # self.td_posterior = norm_dist(higher_layer)
        #     # self.td_posterior = posterior(self.hypotheses.dpd, higher_layer, smooth=True)
        #     self.td_posterior = joint(self.hypotheses.dpd, higher_layer, smooth=True)

        #     self.log(4, "updated top-down posterior from higher layer:\n", self.td_posterior)



    def bu_inference(self):
        """ Calculate the new posterior for the cluster layer, based on evidence from
        predicted sequences per cluster, calculating marginal P(S) using soft-evidence.
        """
        if self.layer_LH is not None:
            # normalization is included in soft_evidence function. 
            # BE CAREFUL: this is where things break!

            # properly calculate new P'(C)
            self.bu_posterior = soft_evidence(self.hypotheses.dpd, self.lower_layer_evidence[0].dpd, self.layer_LH, smooth=True)
            # self.bu_posterior = norm_dist(self.bu_posterior, smooth=True)
            # self.log(1, "posterior sum:", np_sum(self.bu_posterior[:, 0]))

            self.log(4, "bu_update:", self.bu_posterior)



    def extension(self):
        """ Decide on and do hypothesis extension and
        decide on evidence for next higher level.
        The decision for when to send updates to the next higher level is here one made to filter for
        the stable hypothesis, not the most surprising one.
        Thus, surprising signals here, are important only for extension, not for
        influencing higher level processes.
        """

        # decide on extension and of unseen evidence sequence
        if self.lower_layer_new_hypo is not None and self.params["self_supervised"]:
            self.new_sequences_learned = True
            if len(self.hypotheses.reps) == 0:
                hypo = self.hypotheses.add_hypothesis(Cluster, 0.1)
            else:
                hypo = self.hypotheses.reps[self.hypotheses.max()[1]]
            self.collect_in_cluster(hypo, self.lower_layer_new_hypo)

        # decide on updates to communicate for higher level layer
        # decision should be based on stability of hypothesis, not surprise
        # TODO: does this work to just look for a surprising PE?
        if not self.PE.some_surprise() and len(self.hypotheses.reps) > 0:
            # update best hypothesis if there is enough stability in current hypothesis space
            self.best_hypo = self.hypotheses.reps[self.hypotheses.max()[1]]
            self.log(4, "new best hypo:", self.best_hypo.id)

            # stable hypothesis: current free energy is smaller than average, including variance margin
            # if surprise in the hierarchy, inform!
            if self.surprise_received:
                self.layer_evidence = [self.hypotheses,
                                       copy(self.last_surprise_time),
                                       self.PE,
                                       self.self_estimate,
                                       self.intention]
                self.last_surprise_time = 0.
                self.surprise_received = False  # reset signal
            



    def recalculate_clusters_with_new_hypo(self):
        """ Recalculate clusters using Affinity Propagation.
        """

        seqs = list(self.lower_layer_evidence[0].reps.values())
        lenseqs = len(seqs)

        # calculate the whole similarity matrix
        Similarity = np.zeros((lenseqs, lenseqs))
        for j in range(lenseqs):
            for k in range(lenseqs):
                Similarity[j, k] = diff_sequences(seqs[j], seqs[k])

        # clustering until convergence

        # preference=0.9, convergence_iter=50, damping=0.9
        # preference=None, convergence_iter=15, damping=0.5
        cluster_prototypes, labels = affinity_propagation(Similarity, copy=False, verbose=True) #, preference=None, convergence_iter=15, damping=0.5)

        self.log(0, "Estimating number of clusters:", len(cluster_prototypes))

        idx_cluster = defaultdict(list)
        # prep cluster dict
        for seq_idx, center_idx in enumerate(labels):
            idx_cluster[center_idx].append(seqs[seq_idx])

        # wipe and add cluster hypotheses and members
        self.hypotheses.reps = {}
        self.hypotheses.dpd = []
        self.hypotheses.id_counter = 0
        for idx, members in idx_cluster.items():
            new_cluster = self.hypotheses.add_hypothesis(Cluster, P=0.1)
            new_cluster['seqs'] = members
            new_cluster['prototype'] = seqs[idx]

        # construct sparse LH matrix from part-of relationships with space for new sequence hypothesis
        self.layer_LH = defaultdict(list)
        for cl_id, rep in self.hypotheses.reps.items():
            for seq in rep.seqs:
                if seq.id not in self.lost_sequences:
                    self.layer_LH[cl_id].append(seq.id)



    def collect_in_cluster(self, cluster, new_hypo):
        """ Add new cluster if best_hypo precision is lower than average
            and its free-energy is higher than average layer free-energy
        """

        cluster['seqs'].append(new_hypo)

        # # recalculate the cluster prototype
        cluster.find_prototype()

        if self.best_hypo is not None:
            self.log(1, "Added to cluster", cluster['id'], "from sequence", new_hypo['id'])
            # if len(self.best_hypo.seqs) < 2:
            #     self.log(1, "best hypo", self.best_hypo['id'], "inter-cluster:", self.best_hypo.fe, '<', self.avg_inter_cluster_fe - self.var_inter_cluster_fe)
            # else:
            #     self.log(1, "best hypo", self.best_hypo['id'], "within-cluster:", self.best_hypo.fe, '<', self.best_hypo.avg_fe + self.best_hypo.var_fe)

        else:
            self.log(1, "No best cluster! Added to cluster", cluster['id'], "from sequence", new_hypo['id'])

        # construct sparse LH matrix from part-of relationships with space for new sequence hypothesis
        self.layer_LH = defaultdict(list)
        for cl_id, rep in self.hypotheses.reps.items():
            for seq in rep.seqs:
                if seq.id not in self.lost_sequences:
                    self.layer_LH[cl_id].append(seq.id)



    def prediction(self):
        """ Prepare prediction.
        """

        """ Produce a lower layer influencing prediction if there is at least one high precision hypothesis.
        """
        # print(self.best_hypo.id, self.best_hypo.precision, self.layer_average_precision)
        if len(self.hypotheses.reps) > 1 and self.layer_LH is not None:

            # no special mapping here
            if self.intention is not None and self.intention in self.layer_LH:
                # chose a production candidate if there currently is none
                if self.production_candidate is None:
                    # chose signaling if another hypothesis is maxed than the intended
                    if self.signaling_distractor is not None and self.signaling_distractor != self.intention:
                        best_lower_level_hypo = self.lower_layer_evidence[0].reps[self.lower_layer_evidence[0].max()[1]]
                        print("is", best_lower_level_hypo.id, "in", self.layer_LH[self.signaling_distractor], "?")
                        if best_lower_level_hypo.id in self.layer_LH[self.signaling_distractor]:
                            contrastor = best_lower_level_hypo
                        else:
                            self.log(1, "best Seq hypo:", best_lower_level_hypo, "is not in signaling distractor class:", self.signaling_distractor)
                            contrastor = self.hypotheses.reps[self.signaling_distractor]['prototype']
                        # find a signaling candidate from a cluster that is most different than a contrastor_candidate
                        candidate = find_signaling_candidate(cluster=self.hypotheses.reps[self.intention], contrastor_candidate=contrastor)
                        
                        # reinforce motor intention
                        # self.hypotheses.equalize()
                        avg_P = np_mean(self.hypotheses.dpd[:, 0])
                        var_P = 0.2
                        critical_intention = avg_P + var_P  # only +1 var
                        # idx from rep id
                        lrp_idx = self.hypotheses.reps[self.intention].dpd_idx
                        self.hypotheses.dpd = set_hypothesis_P(self.hypotheses.dpd, lrp_idx, critical_intention)
                        self.hypotheses.dpd = norm_dist(self.hypotheses.dpd, smooth=True)

                        self.production_candidate = candidate[1]['id']
                        self.log(1, "Found signaling candidate contrasting from cluster", self.signaling_distractor, ":", candidate)
                    else:
                        
                        if self.seq_intention is None:
                            # chose random production candidate
                            candidate = self.hypotheses.reps[self.intention]['prototype']
                            self.production_candidate = candidate['id']

                            # self.hypotheses.equalize()
                            avg_P = np_mean(self.hypotheses.dpd[:, 0])
                            var_P = 0.2
                            critical_intention = avg_P + var_P  # only +1 var
                            # idx from rep id
                            lrp_idx = self.hypotheses.reps[self.intention].dpd_idx
                            self.hypotheses.dpd = set_hypothesis_P(self.hypotheses.dpd, lrp_idx, critical_intention)
                            self.hypotheses.dpd = norm_dist(self.hypotheses.dpd, smooth=True)
                        else:
                            # we have a seq_intention for specific selection of sequence production
                            self.production_candidate = self.seq_intention

                        self.log(1, "Found intended production candidate", self.production_candidate, "from cluster", self.intention)

                    # in addition and to let intention percolate down the hierarchy, send a LRP
                    self.layer_long_range_projection = {"Seq": {"intention": self.production_candidate}}

            self.layer_prediction = [self.hypotheses.dpd, self.layer_LH]
            # self.log(4, "prediction:", mapping)
            # reset production delay timer
            self.last_production_time = 0.



    def cluster_statistics(self):
        """ Calculate cluster fe matrix and cluster statistics.

        Return cluster similarity matrix.
        """
        clusters = list(self.hypotheses.reps.values())
        lenrep = len(clusters)

        cluster_similarities = np.ones((lenrep, lenrep, 3))
        for j in range(lenrep):
            for k in range(j + 1, lenrep):
                # calculate free-energy between sequences
                matrix, avg_sim, var_sim = inter_cluster_similarity_statistics(clusters[j], clusters[k])
                cluster_similarities[j, k, :] = [clusters[j].id, clusters[k].id, avg_sim]
                cluster_similarities[k, j, :] = [clusters[k].id, clusters[j].id, avg_sim]

        # store for later use
        average_cluster_fe = np_mean(cluster_similarities[:, :, 2])
        var_cluster_fe = np_var(cluster_similarities[:, :, 2])
        self.avg_inter_cluster_sim = average_cluster_fe
        self.var_inter_cluster_sim = var_cluster_fe

        # self.log(3, "inter-cluster distances:", cluster_distances)
        self.log(0, "average inter-cluster similarity:", average_cluster_fe, "variance:", var_cluster_fe)

        return cluster_similarities



