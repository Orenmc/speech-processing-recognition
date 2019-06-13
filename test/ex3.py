import numpy as np
import sys

# global
blank = '#'


def letter_to_index(alphabet):
    """
    dict of letter:index
    :param alphabet: letters
    :return: dictionary(l:i)
    """
    res = {l: i for i, l in enumerate(alphabet)}
    return res


def padding_with_blanks(trans):
    """
    padding transcript with blanks ( 'aa' -> '#a#a#' )
    :param trans: transcript to pad
    :return: padded transcript
    """
    global blank
    res = [blank]
    for s in trans:
        res.append(s)
        res.append(blank)
    return res


def empty_cache(row, col):
    """
    create empty matrix for fill the CTC in the right size
    :param row: number of rows
    :param col: number of cols
    :return:
    """
    return np.zeros((row, col))


def ctc(mat, transcript, alphabet):
    # rows is in size of padded transcription, cols is in size of time size
    cache = empty_cache(len(transcript), mat.shape[0])
    L2I = letter_to_index(alphabet)  # dict letter to index (in alphabet

    # start condition of filling matrix
    cache[0, 0] = mat[0, L2I[transcript[0]]]
    cache[1, 0] = mat[0, L2I[transcript[1]]]

    # build cache column by column
    for col in range(1, cache.shape[1]):
        for row in range(cache.shape[0]):
            if row == 0:  # only 1 way to get to this cell
                path_sum = cache[row, col - 1]
            elif row == 1:
                path_sum = cache[row - 1, col - 1] + cache[row, col - 1]
            elif transcript[row] == '#' or transcript[row] == transcript[row - 2]:
                path_sum = cache[row - 1, col - 1] + cache[row, col - 1]
            else:
                path_sum = cache[row - 2, col - 1] + cache[row - 1, col - 1] + cache[row, col - 1]
            # sum of paths multiple in the probability to get this letter in this time
            cache[row, col] = path_sum * mat[col, L2I[transcript[row]]]
    return cache[-1, -1] + cache[-2, -1]  # return sum of last to cells (blank or last letter in transcript)


def main():
    global blank
    # get args from user
    args = sys.argv
    mat = np.load(str(args[1]))
    transcript = list(args[2])
    alphabet = list(args[3])

    assert blank not in alphabet, '# should not be in alphabet'

    # padded transcript
    pad_transcript = padding_with_blanks(transcript)
    # add blank word to alphabet
    alphabet.append(blank)

    print(round(ctc(mat, pad_transcript, alphabet), 2))


if __name__ == '__main__':
    main()
