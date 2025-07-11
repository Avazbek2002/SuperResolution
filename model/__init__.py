import logging
logger = logging.getLogger('base')


def create_model(opt, rank):
    from .model import DDPM as M
    m = M(opt, rank)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
