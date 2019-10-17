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

""" Hierarchy script to produce all Sequences from a learned knowledge serialization
    Created on 23.05.2019

    @author: skahl
"""


# Imports from __future__ in case we're running Python 2
from __future__ import division, print_function
from __future__ import absolute_import, unicode_literals

import uuid

import os
import time
# import signal
import sys, getopt
import hpbu

from collections import deque

from copy import copy

global quit

# def signal_handler(signal, frame):
#     global quit
#     print('You pressed Ctrl+C!')
#     quit = True
# signal.signal(signal.SIGINT, signal_handler)



def main():
    global quit

    # logging and scenario setup
    hpbu.Logger.doLogging = 1

    """ Very important variables
    """
    _input = None
    _goals_input = None
    _received_interaction_start_signal = False

    # trigger active inference of schema:
    # config_filename = "full_hierarchy_conf.json"
    # _input = [174, 179, 175, 177, 178, 180, 181, 176]
    # _meaning = [1, 2, 3, 5, 6, 7, 8, 9]

    # config_filename = "single_user_full_corpus_no_filter_evaluation_conf.json"
    # _input =   [2, 3, 7, 5, 4, 6, 8, 11, 9, 10]
    # config_filename = "full_single_user_corpus_conf.json"
    # _input =  [11, 8, 9, 5, 2, 3, 7, 6, 4, 10]  # 9-0
    # _meaning = [ 5, 4, 1, 6, 2, 3, 8, 7,  0,  9]
    config_filename = "produce_all_digits_conf.json"
    _input =   [5, 10, 3, 2, 4, 11, 9, 8, 7, 6]
    _meaning = [0,  1, 2, 3, 4,  5, 6, 7, 8, 9]
    # _input = [4, 8]
    # _meaning = [4, 7]

    

    _schema_id_dict = dict(zip(_input, _meaning))
    # _input = [177]  # test only digits with jumps
    
    # config setup
    if config_filename is not None:
        my_path = os.path.dirname(os.path.abspath(__file__))
        hierarchy_config = hpbu.Config(my_path, config_filename)
        storage = hierarchy_config.get_config_storage()
        if storage is not None:
            if not hierarchy_config.config_layer_from_storage():
                print("Layer config not available in DB")
                sys.exit(1)
            if not hierarchy_config.config_parameters_from_storage():
                print("Parameter config not available in DB")
                sys.exit(1)
        else:
            print("Data storage could not be loaded:", hierarchy_config.data_file)
    else:
        print("no config_filename set!")
        sys.exit(1)

    # timing setup
    time_step = hierarchy_config.parameters["time_step"]  # internal model time
    time_diff = 0.0001 # time for this update loop

    # websocket communication setup
    # vis_server_ip = hierarchy_config.parameters["vis_server"]
    # ws_handler = WebsocketHandler(server=vis_server_ip, hierarchy_config=hierarchy_config.get_config_for_visualization(),
    #                               agent_name=hierarchy_config.parameters["my_id"])

    # hierarchy setup
    my_hierarchy = hpbu.Hierarchy(hierarchy_config)
    my_hierarchy.layer_io.gen_primitives()
    my_hierarchy.layer_vision.gen_primitives()

    # disable processing from layer and above
    schema_layer = None
    l = my_hierarchy.layer_top
    if l.name == "Schm":
        schema_layer = l 
    _received_interaction_start_signal = True

    # start update thread
    my_hierarchy.run()

    # model output handling
    prod_idx = deque()
    output_coords = None  # coordinates to send to visualization for drawing
    _last_coord = hpbu.np.array([0., 0.])
    new_coord = hpbu.np.array([0., 0.])
    new_coords = None
    new_timing = None
    motor_step = None
    remember_first_coord = None
    # step counter for tests
    step_counter = 0

    # web input handling
    last_t = None
    _sequence_done = True  # needs to be true to start first production
    _available_sequences = []  # from currently produced schema

    resting_pos = hpbu.np.array([350., 250.])  # on ipad coordinate system source is in the top left corner

    # clean output before starting
    # ws_handler.send_to_web("data", "clear")

    quit = False
    while not quit:

        try:
            unique_loop_id = uuid.uuid4().hex
            """ Check if the start signal for the interaction was received.
            If not, put the model to sleep.
            """
            if not _received_interaction_start_signal:
                my_hierarchy.disable_updates()
            else:
                my_hierarchy.enable_updates()

            # start test_production if that role is set
            if _sequence_done and _received_interaction_start_signal:
                """ general scripted test input
                """
                if len(_available_sequences) == 0 and type(_input) == list and len(_input) > 0:
                    lrp = _input.pop()
                    schm_hypo = schema_layer.hypotheses.reps[lrp]
                    # which sequences are in schema?
                    _available_sequences = copy(schm_hypo.seqs)
                    print("Selecting new schema", lrp, "with sequences:", _available_sequences)
                if len(_available_sequences) > 0:
                    # produce one sequence and prime fitting schm and seq
                    seq_lrp = _available_sequences.pop()
                    my_hierarchy.set_long_range_projection({"Schm": {"intention": lrp, "seq_intention": seq_lrp.id}})
                    print("\nSelecting sequence", seq_lrp, "from schema", lrp, "for production\n")
                    _sequence_done = False

            """ model output
            """
            _model_output = my_hierarchy.get_output()  # receive hierarchy output
            if _model_output is not None and "angle" in _model_output:
                output = _model_output["angle"]  # as movement angle in radians
            else:
                output = None
            # if _model_output is not None and "meta" in _model_output:
            #     # receive meta-communicative signals from the model to send to another agent
            #     ws_handler.send_to_web("meta", _model_output["meta"])

            if output is not None:
                if output[0] is not "done":
                    if new_coords is not None and new_timing is not None:
                        new_coords.extend(output[0])
                        new_timing += output[1]
                    else:
                        # first positions
                        new_coords = output[0]
                        new_timing = output[1]
                        if len(new_coords) > 0:
                            print("remembering first position:", output[0], end=": ")
                            remember_first_coord = output[0][0][0]
                            print(remember_first_coord)

                    motor_step = int(new_timing / time_step)
                else:
                    print("done with drawing after steps:", step_counter)
                    step_counter = 0

                    # print("sending signal to clear webinterface drawing canvas...")
                    _last_coord = hpbu.np.array([0., 0.])
                    new_coord = hpbu.np.array([0., 0.])
                    cur_coord = None
                    new_coords = None
                    new_timing = None
                    motor_step = None
                    remember_first_coord = None

                    _sequence_done = True

                    # ws_handler.send_to_web("data", "clear")

                    # clear output queue
                    # wait_until_queue_empty(ws_queue_output)

            if new_coords is not None and len(new_coords) > 0 and motor_step is not None and motor_step > 0:
                if len(new_coords) > motor_step:
                    prod_idx.extend(hpbu.copy(new_coords[:motor_step]))  # in case of simulation time bigger than production time
                    del new_coords[:motor_step]  # del already produced steps
                else:
                    prod_idx.extend(hpbu.copy(new_coords))  # in case of simulation time smaller than production time
                    del new_coords[:]  # del already produced steps
                # print("added new coords, resulting in queue:", prod_idx)

            # apply new list of movements to environment
            while len(prod_idx) > 0:
                # count steps for test
                step_counter += 1

                new_coord = prod_idx.popleft()
                # subtract first coord to center drawing
                if remember_first_coord is not None:
                    cur_coord = resting_pos + new_coord[0] - remember_first_coord
                else:
                    # skip centration when jumping not at beginning of a sequence
                    cur_coord = resting_pos + new_coord[0]
                # print("motor control moves to: +", new_coord[0], "=", cur_coord)

                # these coordinates simulate visual joint position movement feedback
                my_hierarchy.set_input(unique_loop_id, {"vision": [new_coord[0] - _last_coord, time_step]})
                _last_coord = hpbu.copy(new_coord[0])

                # prep message for visualization
                # output_coords = {"idx": _schema_id_dict[lrp], "timing": new_timing, "isDrawing": new_coord[1], "x": (cur_coord[0]), "y": cur_coord[1], "pressure": 1.0}
                # ws_handler.send_to_web("data", {"coordinates": output_coords})

                # inform proprioception if the last element has just been taken from prod_idx
                if len(prod_idx) == 0:
                    # these coordinates simulate spindle fiber movement feedback and
                    # enables automatic reaction to movement deviations
                    my_hierarchy.set_input(unique_loop_id, {"proprioception": _last_coord})


            # """ send model info to visualization
            # """
            # while len(my_hierarchy.queue_info) > 0:
            #     layer_infos = my_hierarchy.queue_info.pop()
            #     ws_handler.send_to_web("data", layer_infos)
            #     # print(len(my_hierarchy.queue_info), "new layer info")

            hpbu.sleep(time_diff)

        except KeyboardInterrupt:
            quit = True
            my_hierarchy.set_input(unique_loop_id, {"control": "quit"})


    # end while loop

    # stop websocket connection
    # ws_handler.stop()
    
    print("wait to end hierarchy updates")
    hpbu.sleep(1.)

    # print hierarchy report
    my_hierarchy.print_report()

    # store knowledge to storage
    if hierarchy_config.parameters["store_knowledge"]:
        print("Saving layer knowledge to data storage...")
        hierarchy_config.store_knowledge_in_storage(my_hierarchy)

    # close and write back data to storage
    hierarchy_config.close_storage()



    sys.exit(1)




if __name__ == "__main__":
    main()
