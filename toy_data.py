import pandas as pd
import tf
import collections
import random

TOY_IMG_DIM = 2
NUM_CLASSES = 100  # total number of classes in dataset
DATASET_PATH = "toy_dataset.csv"

ProblemInstance = collections.namedtuple(
    "ProblemInstance",
    ["tr_input", "tr_output", "tr_info", "val_input", "val_output", "val_info"])


class DataProvider(object):
    def __init__(self, dataset_split):
        if dataset_split not in ["train", "val", "test"]:
            raise Exception("invalid dataset split")

        self.df_dataset = pd.read_csv(DATASET_PATH)

        # todo: get dataset split
        # todo: ensure meta_split vs inner_loop_split

        def get_instance(self, num_classes, tr_size, val_size):
            """Samples a random N-way K-shot classification problem instance.
            Args:
              num_classes: N in N-way classification.
              tr_size: K in K-shot; number of training examples per class.
              val_size: number of validation examples per class.
            Returns:
              A tuple with 6 Tensors with the following shapes:
              - tr_input: (num_classes, tr_size, NDIM): training image embeddings.
              - tr_output: (num_classes, tr_size, 1): training image labels.
              - tr_info: (num_classes, tr_size): training image file names.
              - val_input: (num_classes, val_size, NDIM): validation image embeddings.
              - val_output: (num_classes, val_size, 1): validation image labels.
              - val_info: (num_classes, val_size): validation image file names.
            """
            sample_count = tr_size + val_size
            all_class_ids = list(range(NUM_CLASSES))
            sampled_class_ids = random.sample(all_class_ids, num_classes)
            random.shuffle(sampled_class_ids)

            images, image_labels, image_paths = [], [], []
            for class_label, class_id in enumerate(sampled_class_ids):
                # get all images for the given class id
                df_class = self.df_dataset[self.df_dataset.class_id == class_id]
                # save sample_count sample of images with label class_label
                df_sample = df_class.sample(sample_count)
                images += df_sample.image_embedding.tolist()
                image_labels += [class_label for _ in range(sample_count)]

                #todo: see if needed
                #image_paths += ["" for _ in range(sample_count)]    # vestigial list that is not used

            tr_input, val_input = tf.Variable(images[:tr_size], tf.float), tf.Variable(images[tr_size:], tf.float)
            tr_output, val_output = tf.Variable(image_labels[:tr_size], tf.int), tf.Variable(image_labels[tr_size:],
                                                                                             tf.int)
            return tr_input, tr_output, tf.Variable([], tf.float), val_input, val_output, tf.Variable([], tf.float)

        def get_batch(self, batch_size, num_classes, tr_size, val_size,
                      num_threads=10):

            one_instance = self.get_instance(num_classes, tr_size, val_size)
            tr_data_size = (num_classes, tr_size)
            val_data_size = (num_classes, val_size)
            task_batch = tf.train.shuffle_batch(one_instance, batch_size=batch_size,
                                                capacity=1000, min_after_dequeue=0,
                                                enqueue_many=False,
                                                shapes=[tr_data_size + (TOY_IMG_DIM,),
                                                        tr_data_size + (1,),
                                                        tr_data_size,
                                                        val_data_size + (TOY_IMG_DIM,),
                                                        val_data_size + (1,),
                                                        val_data_size],
                                                num_threads=num_threads)

            if self._verbose:
                tf.logging.info(task_batch)

            return ProblemInstance(*task_batch)