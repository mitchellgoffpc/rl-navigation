from abc import ABC
from dataclasses import dataclass
from omegaconf import OmegaConf

class BaseConfig(ABC):
  @classmethod
  def merge(cls, *args):
    schema = OmegaConf.structured(cls)
    config = OmegaConf.merge(schema, *args)
    return OmegaConf.to_object(config)

  @classmethod
  def default(cls, **kwargs):
    return cls.merge(cls, kwargs)

  @classmethod
  def from_cli(cls, **kwargs):
    return cls.merge(cls, kwargs, OmegaConf.from_cli())
