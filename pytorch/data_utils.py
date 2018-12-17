from torch.utils.data.sampler import BatchSampler
from torch.utils.data import Dataset
import numpy as np

class SiameseTensorset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset, neighbors = None):
        self.mnist_dataset = mnist_dataset

        self.train = True
#         self.transform = self.mnist_dataset.transform

        if self.train:
            self.neighbors = neighbors
            self.train_labels = self.mnist_dataset.tensors[1]
            self.train_data = self.mnist_dataset.tensors[0]
#             self.labels_set = set(self.train_labels.numpy())
#             self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
#                                      for label in self.labels_set}
#         else:
#             # generate fixed pairs for testing
#             self.test_labels = self.mnist_dataset.test_labels
#             self.test_data = self.mnist_dataset.test_data
#             self.labels_set = set(self.test_labels.numpy())
#             self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
#                                      for label in self.labels_set}

#             random_state = np.random.RandomState(29)

#             positive_pairs = [[i,
#                                random_state.choice(self.label_to_indices[self.test_labels[i]]),
#                                1]
#                               for i in range(0, len(self.test_data), 2)]

#             negative_pairs = [[i,
#                                random_state.choice(self.label_to_indices[
#                                                        np.random.choice(
#                                                            list(self.labels_set - set([self.test_labels[i]]))
#                                                        )
#                                                    ]),
#                                0]
#                               for i in range(1, len(self.test_data), 2)]
#             self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
#         if self.train:
        target = np.random.choice([0, 1], p = [.8, .2])
        img1 = self.train_data[index]#, self.train_labels[index]
        if target == 1:
            siamese_index = np.random.choice(self.neighbors[index])
        else:
            siamese_index = np.random.choice(np.arange(self.train_data.shape[0]))
            while siamese_index in self.neighbors[index]:
                siamese_index = np.random.choice(np.arange(self.train_data.shape[0]))
        img2 = self.train_data[siamese_index]
#         else:
#             img1 = self.test_data[self.test_pairs[index][0]]
#             img2 = self.test_data[self.test_pairs[index][1]]
#             target = self.test_pairs[index][2]

#         if self.transform is not None:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
        return img1, img2, target

    def __len__(self):
        return len(self.mnist_dataset)