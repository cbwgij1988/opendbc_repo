"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import os


def _get_speed_dep_config():
  """Load speed-dependent torque config from toml. Cached after first call."""
  if not hasattr(_get_speed_dep_config, '_cache'):
    import tomllib
    from opendbc.car.common.basedir import BASEDIR
    path = os.path.join(BASEDIR, 'torque_data/speed_dependent.toml')
    with open(path, 'rb') as f:
      _get_speed_dep_config._cache = tomllib.load(f)
  return _get_speed_dep_config._cache


get_speed_dep_config = _get_speed_dep_config
