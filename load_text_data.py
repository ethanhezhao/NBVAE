
import scipy.io as sio


def load_data(mat_file_name):

    data = sio.loadmat(mat_file_name)
    train_data = data['wordsTrain'].transpose()
    test_tr = data['wordsHeldout'].transpose()
    test_te = data['wordsTest'].transpose()

    return train_data, test_tr, test_te

