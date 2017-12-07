import numpy as np
from torch.utils.data import DataLoader


class MHDataLoader(DataLoader):
    """
    DataLoader class for raw audio music performance assessment data
    """

    def __init__(self, dataset, num_data_points=None,):
        """
        Initializes the class, defines the number of batches and other parameters
        Args:
                dataset:  	object of the RawAudioDataset class, should be properly initialized
                num_data_pts:	int, number of data points to be considered while loading the data
        """
        if num_data_points is None:
            num_data_points = len(dataset)

        # check if input parameters are accurate
        assert num_data_points <= len(dataset)
        self.dataset = dataset[:num_data_points]
        self.num_data_points = num_data_points

    def get_data(self):
        return self.dataset

    def get_batched_data(self, batch_size):
        """
        Returns batched data
        """
        assert batch_size <= self.num_data_points
        num_batches = int(np.floor(self.num_data_points / batch_size))
        return np.split(self.dataset, num_batches, axis=0)
