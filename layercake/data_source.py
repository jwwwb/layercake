import layercake as lc
import numpy as np


def get_len(data_dict):
    lens = np.unique(np.asarray([len(v) for k,v in data_dict.items()]))
    if len(lens) != 1:
        raise ValueError("All variables in data_dict must be same length!")
    else:
        return lens[0]


def split_data(data_dict, split_ratio):
    size = get_len(data_dict)
    subsizes = [int(s*size) for s in np.cumsum(split_ratio)]
    data_dicts, p = [], 0
    for s in subsizes:
        data_dicts.append({k: v[p:s] for k,v in data_dict.items()})
        p = s
    return data_dicts


class DataSource:
    def __init__(self, data_dict, batch_size=32):
        """
        The data_dict contains all variables that should be iterated together,
        including inputs, targets, and other information not needed for the
        execution of a network (such a human readable labels).
        """
        self.data_dict = data_dict
        self.batch_size = batch_size
        self.len = get_len(self.data_dict)

    def shuffle_order(self):
        self.order = np.random.permutation(self.len).reshape(-1, self.batch_size)
        #  self.order = np.arange(self.len).reshape(-1, self.batch_size)

    def __len__(self):
        return self.len // self.batch_size

    def __iter__(self):
        self.shuffle_order()
        for o in self.order:
            yield {k: v[o] for k, v in self.data_dict.items()}


if __name__ == '__main__':
    ds = DataSource({'x': np.arange(10), 'y': np.arange(10)}, 2)
    for d in ds:
        print(d)

    dd = ds.data_dict
    a, b = split_data(dd, [.5, .5])
    print(a)
    print(b)

