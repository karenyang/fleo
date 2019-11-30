"""
Create a CSV file with two columns: class_id and image_embedding
"""
import pandas as pd
import numpy as np

STDEV_NOISE = 0.3  # Standard deviation of gaussian noise
NUM_CLASSES = 100
SAMPLES_PER_CLASS = 5000


class RandomLine:
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def sample_batch(self, sample_size):
        points = []
        for _ in range(sample_size):
            x = np.random.uniform(-5, 5)
            y = self.m * x + self.b
            x += np.random.normal(scale=STDEV_NOISE)
            y += np.random.normal(scale=STDEV_NOISE)
            points.append([x, y])
        return points


class RandomSine:
    def __init__(self, amplitude, phase):
        self.amplitude = amplitude
        self.phase = phase

    def sample_batch(self, sample_size):
        points = []
        for _ in range(sample_size):
            x = np.random.uniform(-5, 5)
            y = self.amplitude * np.sin(x + self.phase)
            x += np.random.normal(scale=STDEV_NOISE)
            y += np.random.normal(scale=STDEV_NOISE)
            points.append([x, y])
        return points


def create_dataset(save_path):
    image_embeddings, labels = [], []
    for i in range(NUM_CLASSES):
        use_sine = i % 2 == 0
        if use_sine:  # Generate samples from sine function
            amplitude, phase = np.random.uniform(0.1, 5), np.random.uniform(0, np.pi)
            sampler = RandomSine(amplitude, phase)
        else:  # Generate samples from linear function
            slope, intercept = np.random.randint(-3, 3), np.random.randint(-3, 3)
            sampler = RandomLine(slope, intercept)
        image_embeddings += sampler.sample_batch(SAMPLES_PER_CLASS)
        labels += [i] * SAMPLES_PER_CLASS

    df = pd.DataFrame(list(zip(image_embeddings, labels)), columns=["image_embedding", "class_id"])
    df.to_csv(save_path)


create_dataset("toy_dataset.csv")




