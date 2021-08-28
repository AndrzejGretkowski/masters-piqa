from torch.utils.data import Dataset
from piqa.data.loader import Loader
from piqa.data.downloader import Downloader

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class PiqaDataset(Dataset):
    def __init__(self, name='train', fix=True):
        super().__init__()
        downloader = Downloader()
        status = downloader.download_set(name)
        if not status:
            raise RuntimeError('Wrong name {name} of piqa set. Try train, test or valid.')

        # This merges validation data set into train and takes a bit from train to valid
        if fix is True and name in {'valid', 'train'}:
            _ = downloader.download_set('train')
            train_loader = Loader('train')
            _ = downloader.download_set('valid')
            valid_loader = Loader('valid')

            train_data = train_loader.load()
            valid_data = valid_loader.load()

            train_part, valid_fixed = train_test_split(train_data, test_size=len(valid_data), random_state=42, shuffle=False)

            if name == 'valid':
                self._data = valid_fixed
            else:
                full_train = train_part + valid_data
                self._data = shuffle(full_train, random_state=42)

        # If not fix, just load the given sets
        else:
            loader = Loader(name)
            self._data = loader.load()

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)
