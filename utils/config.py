from types import ModuleType
from typing import Any, List, Union

import yaml
from pytorch_lightning import callbacks, loggers

from utils._types import StringKeyedNestedDict


def read_yaml(path: str) -> StringKeyedNestedDict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def read_nested_yaml_dict(
        config: StringKeyedNestedDict, prefix: str = "file:"
) -> StringKeyedNestedDict:
    for k, v in config.items():
        # print(k, "-->", v)
        if isinstance(v, str):
            if prefix in v:
                # print('entering')
                subconfig = read_nested_yaml(v.replace(prefix, ""))
                config[k] = subconfig
            elif "[[ipaddr]]" == v:
                import socket

                ipaddr = socket.gethostbyname(socket.gethostname())
                config[k] = ipaddr
        elif isinstance(v, dict):
            config[k] = read_nested_yaml_dict(v, prefix)
        else:
            pass

    return config


def read_nested_yaml(path: str, prefix: str = "file:") -> StringKeyedNestedDict:
    config = read_yaml(path)

    return read_nested_yaml_dict(config, prefix)


def dict_to_trainer_kwargs(
        config: StringKeyedNestedDict
        ) -> StringKeyedNestedDict:
    for special_key, ns in [("logger", loggers), ("callbacks", callbacks)]:
        if special_key in config:
            skconfig = config[special_key]

            assert isinstance(skconfig, dict) or isinstance(skconfig, list)

            config[special_key] = instantiate_config(skconfig, ns)

    return config


def instantiate_config(
        config: Union[StringKeyedNestedDict, List[StringKeyedNestedDict]],
        namespace: ModuleType
) -> Any:
    if isinstance(config, dict):
        assert isinstance(config["name"], str)
        iconfig = namespace.__dict__[config["name"]](**config.get("kwargs", {}))
    elif isinstance(config, list):
        iconfig = [instantiate_config(c, namespace) for c in config]
    else:
        raise ValueError

    return iconfig
