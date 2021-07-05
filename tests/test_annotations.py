import json
from pathlib import Path
import pytest

from typing import get_args, List, Literal
from pydantic import BaseModel, conint, confloat, conlist

CATEGORY_NAMES = Literal["chair", "couch", "tv", "remote", "book", "vase"] 

class COCOCategory(BaseModel):
    id: int
    name: CATEGORY_NAMES


class COCOImage(BaseModel):
    id: int
    file_name: str


class COCOAnnotation(BaseModel):
    image_id: int
    bbox: conlist(int, min_items=4, max_items=4)
    category_id: int


class COCODetectionDataset(BaseModel):
    images: List[COCOImage]
    annotations: List[COCOAnnotation]
    categories: List[COCOCategory]


@pytest.mark.parametrize("split", ["train", "val"])
def test_coco_format(split):

    annotations_file = f"data/coco_sample/annotations/split_{split}.json"
    
    with open(annotations_file, "r") as f:
        dataset = COCODetectionDataset(**json.load(f))

    # Check image ids are unique
    image_ids = [img.id for img in dataset.images]
    image_ids_set = set(image_ids)
    assert len(image_ids) == len(image_ids_set)

    # Check annotation ids are unique
    categories = [cat.id for cat in dataset.categories]
    categories_set = set(categories)
    assert len(categories) == len(categories_set)

    # Check each annotation corresponds with existent image
    for annotation in dataset.annotations:
        assert annotation.image_id in image_ids_set
