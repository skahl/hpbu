import uuid
import os
import time
import sys, getopt
import hpbu
from copy import copy

global quit

config_filename = "minimal_config.json"

def main():
    global quit

    # logging and scenario setup
    hpbu.Logger.doLogging = 1

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

    # hierarchy setup
    my_hierarchy = hpbu.Hierarchy(hierarchy_config)
    my_hierarchy.layer_io.gen_primitives()
    my_hierarchy.layer_vision.gen_primitives()

    # timing setup
    time_step = hierarchy_config.parameters["time_step"]  # internal model time
    time_diff = 0.0001 # time for this update loop

    # start update thread
    my_hierarchy.run()
    my_hierarchy.enable_updates()
    # my_hierarchy.disable_updates()

    quit = False
    while not quit:
        try:
            unique_loop_id = uuid.uuid4().hex

            """ model output
            """
            _model_output = my_hierarchy.get_output()  # receive hierarchy output

            """ model input
            """
            my_hierarchy.set_input(unique_loop_id, {"vision": [_model_output, time_step]})
            my_hierarchy.set_input(unique_loop_id, {"proprioception": _model_output})

            hpbu.sleep(time_diff)

        except KeyboardInterrupt:
            quit = True
            my_hierarchy.set_input(unique_loop_id, {"control": "quit"})

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

    sys.exit()

if __name__ == "__main__":
    main()