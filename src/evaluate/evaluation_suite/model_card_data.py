from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelCardResult:
  header: Optional[str] = None
  summary: Optional[str] = None
  table: Optional[str] = None
  graphic: Optional[str] = None

@dataclass
class ModelCardResults:
  collection: List[ModelCardResult]
  
  def __init__(self, collection=[]):
      self.collection = collection
  
  def add_result(self, result):
      self.collection.append(result)
