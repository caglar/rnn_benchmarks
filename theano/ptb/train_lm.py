import argparse
import logging
import pprint

import config_lm

from lm import train

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--proto", help="Prototype config to use for model configuration")
    parser.add_argument("--platoon", action='store_true', help="Prototype config to use for model configuration")
    args = parser.parse_args()

    config = getattr(config_lm, args.proto)()
    config['platoon'] = args.platoon
    logger.info("Model options:\n{}".format(pprint.pformat(config)))
    train(**config)
