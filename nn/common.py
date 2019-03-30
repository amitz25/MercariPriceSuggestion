import yaml
from enum import Enum


class Container:
    pass


class ConfigParser(object):
    def __init__(self, config):
        stream = open(config, 'r')
        docs = yaml.load_all(stream)
        for doc in docs:
            for k, v in doc.items():
                cmd = "self." + k + "=" + 'Container()'
                exec(cmd)
                for k1, v1 in v.items():
                    cmd = "self." + k + '.' + k1 + "=" + repr(v1)
                    print(cmd)
                    exec(cmd)
        stream.close()


class DBType(Enum):
    Train = 0,
    Validation = 1,
    Test = 2