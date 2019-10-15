# HPBU

Here I provide the core algorithm for the hierarchical predictive belief update (HPBU),

as it was described in:
Kahl, S., & Kopp, S. (2018). A Predictive Processing Model of Perception and Action for Self-Other Distinction. Frontiers in Psychology, 9, 47–14. http://doi.org/10.3389/fpsyg.2018.02421

and in:
Kahl (in work). Social Motorics: A predictive processing model for efficient embodied communication

### Build & Install

To build HPBU, using conda-build:

    conda-build conda.recipe

To install HPBU into your path (Conda is highly recommended!):

    pip install .

### RUN

For simple access to the hpbu core module,

    import hpbu

You will need to have a script ready that loads a configuration for the model hierarchy.
An example hierarchy configuration can be found under hpbu/examples, but here is a minimal example:

    {
        "layers": [
            {
                "type": "MotorControl",
                "name": "MC",
                "color": "Red",
            },
            {
                "type": "Vision",
                "name": "Vision",
                "color": "Blue",
            },
            {
                "type": "Sequence",
                "name": "Seq",
                "color": "Yellow",
            },
            {
                "type": "Top",
                "name": "Schm",
                "color": "Green",
            }
        ],
        "parameters": {
            "my_id": "Agent",
            "personmodel_file": "",
            "interaction_influence_mode": "focus_only",
            "self_supervised": true,
            "update_delay": 0.001,
            "memory_len": 100,
            "store_knowledge": false,
            "read_knowledge": true,
            "vis_server": "127.0.0.1",
            "time_step": 0.01,
            "bias_gain": 0.1
        }
    }

Coming back to the script that loads the configuration.
Again, you can find an example under hpbu/examples.
But also, here is a minimal example:

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
            hierarchy_config = hpbu.Config(my_path + os.sep + config_filename)
            storage = hierarchy_config.get_data_storage()
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
                my_hierarchy.set_input(unique_loop_id, {"vision": "test", time_step]})
                my_hierarchy.set_input(unique_loop_id, {"proprioception": "test"})

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


These examples just show the general setup. By itself it won't do anything interesting.

---
Copyright 2017, 2018, 2019 by Sebastian Kahl
