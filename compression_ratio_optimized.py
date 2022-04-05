from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
from scipy.sparse import csr_matrix
import glzip as gl

def normal_csr(edge_index):
    src = edge_index[0]
    dst = edge_index[1]
    scipy_csr = csr_matrix((np.zeros(dst.shape, dtype=np.int32), (src, dst)))
    scipy_nbytes = scipy_csr.indptr.astype(np.uint64).nbytes + scipy_csr.indices.astype(np.uint32).nbytes
    return scipy_nbytes

def compare_csrs(name, root):
    dataset = PygNodePropPredDataset(name = args.name, root = args.root)
    train_idx = dataset.get_idx_split()['train'].numpy()
    edge_index = dataset.data.edge_index.numpy()
    (glzip_csr, _) = gl.CSR(edge_index=edge_index).optimize(train_idx, [15, 10, 5])
    normal_csr_nbytes = normal_csr(edge_index)
    
    edge_index_ratio_str = "{:2.2%}".format(glzip_csr.nbytes / edge_index.nbytes)
    normal_csr_ratio_str = "{:2.2%}".format(glzip_csr.nbytes / normal_csr_nbytes)

    edge_index_nbytes_str = str(edge_index.nbytes)
    normal_csr_nbytes_str = str(normal_csr_nbytes)
    compressed_csr_nbytes_str = str(glzip_csr.nbytes)

    nbytes_len = max(len(edge_index_nbytes_str), len(normal_csr_nbytes_str)) + 2
    ratio_len = max(len(edge_index_ratio_str), len(normal_csr_nbytes_str)) + 2

    seperator = ('-' * 16) + '+' + ('-' * nbytes_len) + '+' + ('-' * ratio_len)

    edge_index_nbytes_str = edge_index_nbytes_str.center(nbytes_len, ' ')
    normal_csr_nbytes_str = normal_csr_nbytes_str.center(nbytes_len, ' ')
    compressed_csr_nbytes_str = compressed_csr_nbytes_str.center(nbytes_len, ' ')

    edge_index_ratio_str = edge_index_ratio_str.center(ratio_len, ' ')
    normal_csr_ratio_str = normal_csr_ratio_str.center(ratio_len, ' ')

    print('')
    print('representation'.center(16, ' ') + '|' + 'nbytes'.center(nbytes_len, ' ') + '|' + 'ratio'.center(ratio_len, ' '))
    print(seperator)
    print('edge index'.center(16, ' ') + '|' + edge_index_nbytes_str + '|' + edge_index_ratio_str)
    print(seperator)
    print('normal csr'.center(16, ' ') + '|' + normal_csr_nbytes_str + '|' + normal_csr_ratio_str)
    print(seperator)
    print('compressed csr'.center(16, ' ') + '|' + compressed_csr_nbytes_str + '|' + '~'.center(ratio_len, ' '))
    print('')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare glzip to scipy')
    parser.add_argument('name', metavar='NAME', type=str)
    parser.add_argument('root', metavar='ROOT', type=str)
    args = parser.parse_args()
    compare_csrs(args.name, args.root)
