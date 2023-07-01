import os


class Dir:

    def __init__(self, resize):
        self.resize = resize

    def dir(self):
        # The database file for training data. Created by data/text/create_data.sh
        train_data = "/home/sweekar/SDN_main/data/lmdb/lmdb/text_train_lmdb"
        # The database file for testing data. Created by data/text_10x/create_data.sh
        test_data = "/home/sweekar/SDN_main/data/lmdb/lmdb/text_test_lmdb"

        # Modify the task name if you want.
        task_name = "{}".format(self.resize)
        # The name of the model. Modify it if you want.
        model_name = "VGG_{}".format(task_name)

        # Directory which stores the model .prototxt file.
        save_dir = "/home/sweekar/SDN_main/models/VGG/{}".format(task_name)
        # Directory which stores the snapshot of models.
        snapshot_dir = "/home/sweekar/SDN_main/models/VGG/{}".format(task_name)
        # Directory which stores the task script and log file.
        task_dir = "/home/sweekar/SDN_main/tasks/VGG/{}".format(task_name)
        # Directory which stores the detection results.
        output_result_dir = "{}/data/text/results/{}/Main".format(os.environ['HOME'], task_name)

        # model definition files.
        train_net_file = "{}/train.prototxt".format(save_dir)
        test_net_file = "{}/test.prototxt".format(save_dir)
        deploy_net_file = "{}/deploy.prototxt".format(save_dir)
        solver_file = "{}/solver.prototxt".format(save_dir)
        # snapshot prefix.
        snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
        # task script path.
        task_file = "{}/{}.sh".format(task_dir, model_name)

        return train_data, test_data, model_name, output_result_dir, train_net_file, test_net_file, deploy_net_file, solver_file, snapshot_prefix, task_file, save_dir,task_dir,snapshot_dir
