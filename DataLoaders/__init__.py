from torch.utils.data import Dataset, DataLoader


# Data loader that loads inputs and associated labels:
class DataLoaderLabels(Dataset):

    def __init__(self, data, labels):

        self.data = data
        self.labels = labels

    def __getitem__(self, index):

        sample = self.data[index]
        label = self.labels[index]

        return sample, label

    def __len__(self):

        return len(self.labels)


# Data loader that loads only inputs (this was used in val and test sets where unshuffled samples were passed through
# the model in batches and then their embeddings were combined into a single object. The labels were accessed via
# labels_va and labels_te objects (see train_val funcs in TrainEval package).)
class DataLoaderNoLabels(Dataset):

    def __init__(self, data):

        self.data = data

    def __getitem__(self, index):

        sample = self.data[index]

        return sample

    def __len__(self):

        return self.data.size(0)


def init_data_loaders(data_loader_tr_class, data_loader_va_class, dataset_tr, labels_tr, dataset_va, dataset_te,
                      batch_size, sampler_tr):

    data_loader_tr = DataLoader(data_loader_tr_class(dataset_tr, labels_tr), batch_size=batch_size, sampler=sampler_tr,
                                num_workers=0)
    data_loader_va = DataLoader(data_loader_va_class(dataset_va), batch_size=batch_size, shuffle=False, num_workers=0)
    data_loader_te = DataLoader(data_loader_va_class(dataset_te), batch_size=batch_size, shuffle=False, num_workers=0)

    return data_loader_tr, data_loader_va, data_loader_te
