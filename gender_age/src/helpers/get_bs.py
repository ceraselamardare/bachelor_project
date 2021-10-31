# determinarea batch_size

def get_bs(paths, b_max):
    length = len(paths)
    batch_size = \
    sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= b_max], reverse=True)[0]
    return batch_size, int(length / batch_size)
