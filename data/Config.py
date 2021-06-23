from configparser import ConfigParser
import sys
sys.path.append('..')


class Configurable(object):
    def __init__(self, config_file, extra_args):
        config = ConfigParser()
        config.read(config_file)
        if extra_args:
            extra_args = dict([ (k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        self._config = config
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    @property
    def hidden_size(self):
        return self._config.getint('Network', 'hidden_size')
    @property
    def input_size(self):
        return self._config.getint('Network', 'input_size')
    @property
    def dropout(self):
        return self._config.getfloat('Network', 'dropout')
    @property
    def class_num(self):
        return self._config.getint('Network', 'class_num')
    @property
    def arc(self):
        return self._config.getboolean('Network', 'arc')
    @property
    def k(self):
        return self._config.getint('Network', 'k')
    @property
    def loss_ma(self):
        return self._config.getfloat('Network', 'loss_ma')
    @property
    def mtl(self):
        return self._config.getboolean('Network', 'mtl')
    @property
    def dot(self):
        return self._config.getboolean('Network', 'dot')

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer','learning_rate')
    @property
    def learning_rate2(self):
        return self._config.getfloat('Optimizer','learning_rate2')

    @property
    def epochs(self):
        return self._config.getint('Run', 'epochs')
