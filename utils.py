import os
import shutil
import typing

# ------------------------- Parser Utils -------------------------


def register_parser_types(parser, params_named_tuple):
  """Register arguments based on the named tuple"""
  # XXX upgrade to support dataclass instead after python 3.7.0
  parser.register('type', bool, lambda v: v.lower() == 'true')
  parser.register('type', typing.List[int], lambda v: tuple(map(int, v.split(','))))

  hints = typing.get_type_hints(params_named_tuple)
  defaults = params_named_tuple()._asdict()

  for key, _type in hints.items():
    parser.add_argument(f'--{key}', type=_type, default=defaults.get(key))


# ------------------------- Path Utils -------------------------


def ensure_path(path):
  """Create the path if it does not exist"""
  if not os.path.exists(path):
    os.makedirs(path)
  return path


def remove_path(path):
  """Remove the path if it exists."""
  if os.path.exists(path):
    shutil.rmtree(path)
