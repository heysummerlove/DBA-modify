import torch
device=torch.device('cpu')

AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'
AGGR_FOOLSGOLD='foolsgold'
AGGR_GRAD_MEAN='grad_mean'
MAX_UPDATE_NORM = 1000  # reject all updates larger than this amount
patience_iter=20

TYPE_LOAN='loan'
TYPE_CIFAR='cifar'
TYPE_MNIST='mnist'
TYPE_TINYIMAGENET='tiny-imagenet-200'