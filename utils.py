import os
import shutil

def ensure_path(path):
  """Create the path if it does not exist
  """
  if not os.path.exists(path):
    os.makedirs(path)
  return path

def remove_path(path):
  """Remove the path if it exists."""
  if os.path.exists(path):
    shutil.rmtree(path)