from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'avg_pool_3x3',
    'skip_connect',
    'nor_conv_3x3',
    'nor_conv_5x5',
    'nor_conv_7x7',
]

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

#################### Best Genotypes Found #############################
DARTS_CIFAR10 = Genotype(normal=[('nor_conv_3x3', 1), ('nor_conv_7x7', 0), ('nor_conv_3x3', 2), ('nor_conv_3x3', 1), ('nor_conv_3x3', 3), ('nor_conv_3x3', 2), ('skip_connect', 0), ('nor_conv_7x7', 4)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('nor_conv_7x7', 2), ('nor_conv_7x7', 0), ('skip_connect', 0), ('nor_conv_7x7', 2), ('skip_connect', 1), ('skip_connect', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_DEVANAGARI = Genotype(normal=[('nor_conv_7x7', 0), ('nor_conv_7x7', 1), ('nor_conv_7x7', 2), ('skip_connect', 0), ('avg_pool_3x3', 3), ('skip_connect', 1), ('avg_pool_3x3', 3), ('avg_pool_3x3', 4)], normal_concat=[2, 3, 4, 5], reduce=[('nor_conv_7x7', 1), ('nor_conv_7x7', 0), ('nor_conv_7x7', 2), ('skip_connect', 1), ('nor_conv_7x7', 2), ('nor_conv_7x7', 3), ('nor_conv_7x7', 4), ('nor_conv_7x7', 3)], reduce_concat=[2, 3, 4, 5])
DARTS_FASHION = Genotype(normal=[('nor_conv_7x7', 1), ('nor_conv_7x7', 0), ('nor_conv_3x3', 2), ('nor_conv_7x7', 1), ('nor_conv_3x3', 3), ('nor_conv_3x3', 2), ('nor_conv_3x3', 3), ('nor_conv_5x5', 2)], normal_concat=[2, 3, 4, 5], reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('nor_conv_7x7', 2), ('skip_connect', 0), ('nor_conv_5x5', 1), ('skip_connect', 0), ('skip_connect', 1)], reduce_concat=[2, 3, 4, 5])
DARTS_MNIST = Genotype(normal=[('nor_conv_5x5', 1), ('skip_connect', 0), ('nor_conv_7x7', 2), ('nor_conv_7x7', 1), ('nor_conv_7x7', 1), ('skip_connect', 0), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5], reduce=[('nor_conv_7x7', 0), ('avg_pool_3x3', 1), ('nor_conv_3x3', 2), ('nor_conv_7x7', 1), ('skip_connect', 1), ('nor_conv_7x7', 2), ('nor_conv_7x7', 4), ('skip_connect', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_STL10 = Genotype(normal=[('nor_conv_3x3', 1), ('nor_conv_3x3', 0), ('nor_conv_3x3', 2), ('nor_conv_3x3', 0), ('nor_conv_3x3', 3), ('skip_connect', 0), ('nor_conv_3x3', 4), ('skip_connect', 0)], normal_concat=[2, 3, 4, 5], reduce=[('nor_conv_5x5', 1), ('nor_conv_5x5', 0), ('nor_conv_5x5', 2), ('nor_conv_3x3', 1), ('skip_connect', 1), ('nor_conv_5x5', 3), ('nor_conv_3x3', 4), ('nor_conv_7x7', 1)], reduce_concat=[2, 3, 4, 5])
DARTS_SVHN = Genotype(normal=[('nor_conv_7x7', 1), ('nor_conv_7x7', 0), ('nor_conv_7x7', 2), ('skip_connect', 1), ('skip_connect', 0), ('nor_conv_3x3', 3), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=[2, 3, 4, 5], reduce=[('nor_conv_7x7', 1), ('skip_connect', 0), ('nor_conv_7x7', 2), ('skip_connect', 1), ('nor_conv_5x5', 3), ('nor_conv_5x5', 2), ('nor_conv_3x3', 3), ('nor_conv_3x3', 4)], reduce_concat=[2, 3, 4, 5])
DARTS = DARTS_V2

