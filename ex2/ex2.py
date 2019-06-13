import numpy as np
import os
import glob
import librosa
import scipy.stats as stats


def euclidean_dist(x, y):
    return np.linalg.norm(x - y)


def load_data(path):
    files = [f for f in glob.glob(os.path.join(path, '*.wav'))]
    waves = []
    for f in files:
        y, sr = librosa.load(f, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        # normalized

        waves.append(stats.zscore(mfcc))
    return waves


def load_train_data():
    """
    load all train files- 5 labeled as 1, 5 as 2 and so on.. till 5
    :return: list of all train files
    """
    return load_data('data/train_data/one') + load_data('data/train_data/two') + \
           load_data('data/train_data/three') + \
           load_data('data/train_data/four') + load_data('data/train_data/five')


def get_predictions(train_data, test_file_path):
    """
    load test data, and for each test sample check 1-Nearest neighbor by euclidean distance
    and dtw distance.
    :param train_data: all 25 traning set
    :param test_file_path: path to test files
    :return: list of prediction in the format: <name> - <euclidean dist> - <dtw dist>
    """
    pred = []
    # run over test
    files = [f for f in glob.glob(os.path.join(test_file_path, '*.wav'))]
    # read all test files
    for f in files:
        y, sr = librosa.load(f, sr=None)
        test_sample = librosa.feature.mfcc(y=y, sr=sr)
        test_name = os.path.basename(f)

        # normalized
        test_sample = stats.zscore(test_sample)

        min_dtw = np.inf
        min_euc = np.inf
        min_indx_dtw = -1
        min_indx_euc = -1
        for i, train_sample in enumerate(train_data):
            # find DTW prediction on test sample
            dist_mat = build_dist_matrix(train_sample, test_sample)
            DTW = build_DTW_matrix(dist_mat)

            if DTW[-1, -1] < min_dtw:
                min_dtw = DTW[-1, -1]
                min_indx_dtw = i

            # find euclidean prediction
            euc_dist = euclidean_dist(train_sample.flatten(), test_sample.flatten())
            if euc_dist < min_euc:
                min_euc = euc_dist
                min_indx_euc = i

        dtw_pred = int(min_indx_dtw / 5) + 1  # for example index 0-> label 1, index 5 -> label 2 and so on
        euclidean_pred = int(min_indx_euc / 5) + 1
        pred.append((test_name, euclidean_pred, dtw_pred))
    return pred


def build_dist_matrix(m1, m2):
    """
    build distance matrix (euclidean dist) between m1,m2
    :param m1: matrix
    :param m2: matrix
    :return: distance matrix
    """
    rows = np.shape(m1)[1]
    cols = np.shape(m2)[1]

    dist_mat = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            # get the distance between the i column of m1, and j column of m2
            dist_mat[i][j] = euclidean_dist(m1[:, i], m2[:, j])

    return dist_mat


def build_DTW_matrix(d):
    rows, cols = np.shape(d)
    DTW = np.zeros(np.shape(d))

    # build stopping condition
    DTW[0][0] = d[0][0]
    for j in range(1, cols):
        DTW[0][j] = d[0][j] + DTW[0][j - 1]
    for i in range(1, rows):
        DTW[i][0] = d[i][0] + DTW[i - 1][0]
    # build the inside matrix
    for i in range(1, rows):
        for j in range(1, cols):
            min_val = np.amin([DTW[i][j - 1], DTW[i - 1][j], DTW[i - 1][j - 1]])
            DTW[i][j] = d[i][j] + min_val
    return DTW


def print_prd(prediction, filename):
    with open(filename, 'w') as f:
        for pred in prediction:
            line = str(pred[0]) + ' - ' + str(pred[1]) + ' - ' + str(pred[2]) + '\n'
            f.writelines(line)


def main():
    train_data = load_train_data()
    predictions = get_predictions(train_data, 'data/test_files')
    print_prd(predictions, 'output.txt')


if __name__ == '__main__':
    main()
