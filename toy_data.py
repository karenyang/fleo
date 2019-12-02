import pandas as pd
import tensorflow as tf
import collections
import random
import numpy as np

TOY_IMG_DIM = 1
NUM_TASKS = 100   # Number of tasks in the dataset
DATASET_PATH = "toy_dataset.csv"

ProblemInstance = collections.namedtuple(
    "ProblemInstance",
    ["tr_input", "tr_output", "tr_info", "val_input", "val_output", "val_info"])


class DataProvider(object):
    def __init__(self, dataset_split):
        if dataset_split not in ["train", "val", "test"]:
            raise Exception("invalid dataset split")

        self.df_dataset = pd.read_csv(DATASET_PATH)
        self._verbose = False
        # todo: get dataset split
        # todo: ensure meta_split vs inner_loop_split

    """
    @returns 
        tr_input   tensor (batch_size, tr_size)  
        tr_output  tensor (batch_size, val_size) 
        tr_functions list (batch_size,) 
        val_input  tensor (batch_size, tr_size) 
        val_output tensor (batch_size, val_size) 
        val_functions list (batch_size,) 
    """
    def get_instance(self, num_tasks, tr_size, val_size):
        sample_count = tr_size + val_size
        all_task_ids = list(range(NUM_TASKS))
        sampled_task_ids = random.sample(all_task_ids, num_tasks)
        random.shuffle(sampled_task_ids)

        X_list, Y_list, functions_list = [], [], []
        for task_id in sampled_task_ids:
            df_task = self.df_dataset[self.df_dataset.task_id == task_id]
            df_sample = df_task.sample(sample_count)
            X_list.append(df_sample.X.tolist())
            Y_list.append(df_sample.Y.tolist())
            functions_list.append(df_sample.function.tolist()[0])

        X, Y = np.array(X_list), np.array(Y_list)
        tr_input, val_input = tf.Variable(X[:, :tr_size], tf.float32), tf.Variable(X[:, tr_size:], tf.float32)
        tr_output, val_output = tf.Variable(Y[:, :tr_size], tf.int32), tf.Variable(Y[:, tr_size:], tf.int32)
        tr_input, val_input = tf.expand_dims(tr_input, axis=-1), tf.expand_dims(val_input, axis=-1)
        tr_output, val_output = tf.expand_dims(tr_output, axis=-1), tf.expand_dims(val_output, axis=-1) 
        #return tr_input, tr_output, functions_list[:tr_size], val_input, val_output, functions_list[tr_size:]
        return tr_input, tr_output, tf.zeros((num_tasks, tr_size), tf.float32), val_input, val_output, tf.zeros((num_tasks, val_size), tf.float32)

    def get_batch(self, num_tasks, tr_size, val_size):
        tasks_batch = self.get_instance(num_tasks, tr_size, val_size)
        correct_task_batch = []
        for task in tasks_batch:
          if task.dtype == tf.float64:
            correct_task_batch.append(tf.cast(task, tf.float32))
          elif task.dtype == tf.int64:
            correct_task_batch.append(tf.cast(task, tf.int32))
          else:
           correct_task_batch.append(task)
        return ProblemInstance(*correct_task_batch)
        #return ProblemInstance(*tasks_batch)

    # NUM_CLASSES = 100  # total number of classes in dataset
    # def get_instance(self, num_classes, tr_size, val_size):
    #     """Samples a random N-way K-shot classification problem instance.
    #     Args:
    #       num_classes: N in N-way classification.
    #       tr_size: K in K-shot; number of training examples per class.
    #       val_size: number of validation examples per class.
    #     Returns:
    #       A tuple with 6 Tensors with the following shapes:
    #       - tr_input: (num_classes, tr_size, NDIM): training image embeddings.
    #       - tr_output: (num_classes, tr_size, 1): training image labels.
    #       - tr_info: (num_classes, tr_size): training image file names.
    #       - val_input: (num_classes, val_size, NDIM): validation image embeddings.
    #       - val_output: (num_classes, val_size, 1): validation image labels.
    #       - val_info: (num_classes, val_size): validation image file names.
    #
    #       tr_input (batch size, 5, 1)  batch_size = 1
    #       tr_output (batch size, 5, 1)    batch_size = 1
    #
    #
    #     """
    #     #import pdb; pdb.set_trace()
    #     sample_count = tr_size + val_size
    #     all_class_ids = list(range(NUM_CLASSES))
    #     sampled_class_ids = random.sample(all_class_ids, num_classes)
    #     random.shuffle(sampled_class_ids)
    #
    #     images, image_labels, image_paths = [], [], []
    #     for class_label, class_id in enumerate(sampled_class_ids):
    #         # get all images for the given class id
    #         df_class = self.df_dataset[self.df_dataset.class_id == class_id]
    #         # save sample_count sample of images with label class_label
    #         df_sample = df_class.sample(sample_count)
    #         images.append(list(map(self._str_to_list, df_sample.image_embedding)))
    #         image_labels.append([class_label for _ in range(sample_count)])
    #
    #     images, image_labels = np.array(images), np.array(image_labels)
    #     tr_input, val_input = tf.Variable(images[:, :tr_size], tf.float32), tf.Variable(images[:, tr_size:], tf.float32)
    #     tr_output, val_output = tf.Variable(image_labels[:, :tr_size], tf.int32), tf.Variable(image_labels[:, tr_size:],
    #                                                                                      tf.int32)
    #     tr_output, val_output = tf.expand_dims(tr_output, axis=-1), tf.expand_dims(val_output, axis=-1)
    #     #import pdb; pdb.set_trace()
    #     return tr_input, tr_output, tf.zeros((num_classes, tr_size), tf.float32), val_input, val_output, tf.zeros((num_classes, val_size), tf.float32)
    #
    # def get_batch(self, batch_size, num_classes, tr_size, val_size,
    #         num_threads=10):
    #
    #     one_instance = self.get_instance(num_classes, tr_size, val_size)
    #     tr_data_size = (num_classes, tr_size)
    #     val_data_size = (num_classes, val_size)
    #     #import pdb; pdb.set_trace()
    #
    #     # todo: do not shuffle if is test
    #     task_batch = tf.train.shuffle_batch(one_instance, batch_size=batch_size,
    #                                         capacity=1000, min_after_dequeue=0,
    #                                         enqueue_many=False,
    #                                         shapes=[tr_data_size + (TOY_IMG_DIM,),
    #                                                 tr_data_size + (1,),
    #                                                 tr_data_size,
    #                                                 val_data_size + (TOY_IMG_DIM,),
    #                                                 val_data_size + (1,),
    #                                                 val_data_size],
    #                                         num_threads=num_threads)
    #
    #     if self._verbose:
    #         tf.logging.info(task_batch)
    #     #import pdb; pdb.set_trace()
    #     correct_task_batch = []
    #     for task in task_batch:
    #         if task.dtype == tf.float64:
    #             correct_task_batch.append(tf.cast(task, tf.float32))
    #         elif task.dtype == tf.int64:
    #             correct_task_batch.append(tf.cast(task, tf.int32))
    #         else:
    #             correct_task_batch.append(task)
    #     #import pdb; pdb.set_trace()
    #     return ProblemInstance(*correct_task_batch)
    #
    # def _str_to_list(self, list_as_str):
    #     items = list_as_str[1:-1].split(",")
    #     return [float(item) for item in items]
