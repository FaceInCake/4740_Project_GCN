
from functools import lru_cache
from json import load as load_json
from csv import reader as csv_reader
from typing import Literal, NamedTuple



class ImageInfo (NamedTuple):
    id : int
    width : int
    height : int
    fileName : str
    # liscense_id : int
    # coco_url : str
    # date_captured : str
    # flickr_url : str

class Category (NamedTuple):
    id : int
    name : str
    superCategory : str

class Point (NamedTuple):
    x : float
    y : float

Polygons = list[list[Point]]
"Represents a list of polygons, each polygon is a list of Points"

class RLE (NamedTuple):
    width : int
    height : int
    RLE :list[int]

class Rectangle (NamedTuple):
    x : float
    y : float
    width : float
    height : float

class ObjectSegmentation (NamedTuple):
    id : int
    image : ImageInfo
    category : Category
    boundingBox : Rectangle
    isCrowd : bool
    totalArea :float
    segmentation : Polygons | RLE



@lru_cache(1)
def get_categories () -> dict[int,Category]:
    FILE_PATH = "Data/ParsedAnnotations/CategoryList.csv"
    retr = {}
    with open(FILE_PATH) as fin:
        fin.readline()
        for line in csv_reader(fin, delimiter='\t'):
            id = int(line[0])
            retr[id] = Category(
                id,
                line[1],
                line[2]
            )
    return retr

@lru_cache(1)
def get_image_list (type:Literal["train","val"]) -> dict[int,ImageInfo]:
    FILE_PATH = f"Data/ParsedAnnotations/ImageList_{type}.csv"
    retr = {}
    with open(FILE_PATH) as fin:
        fin.readline()
        for line in csv_reader(fin, delimiter='\t'):
            id = int(line[0])
            retr[id] = ImageInfo(
                id,
                int(line[1]),
                int(line[2]),
                line[3]
            )
    return retr

@lru_cache(2)
def get_segmentations (type:Literal["train","val"]) -> dict[int,Polygons|RLE]:
    FILE_PATH = f"Data/ParsedAnnotations/Segmentations_{type}.json"
    retr = {}
    with open(FILE_PATH) as fin:
        obj = load_json(fin)
    for id, seg in obj.items():
        id = int(id)
        if isinstance(seg, dict):
            size = seg["size"]
            retr[id] = RLE(size[0], size[1], seg["counts"])
        else:
            retr[id] = Polygons(
                [
                    Point(poly[i], poly[i+1])
                    for i in range(0, len(poly)-1, 2)
                ] for poly in seg
            )
    return retr

@lru_cache(2)
def get_objects (type:Literal["train","val"]) -> dict[int,ObjectSegmentation]:
    cats = get_categories()
    images = get_image_list(type)
    segs = get_segmentations(type)
    FILE_PATH = f"Data/ParsedAnnotations/Objects_{type}.csv"
    retr = {}
    with open(FILE_PATH) as fin:
        fin.readline()
        for line in csv_reader(fin, delimiter='\t'):
            id = int(line[0])
            retr[id] = ObjectSegmentation(
                id,
                images[int(line[1])], # image
                cats[int(line[2])], # category
                Rectangle(
                    float(line[4]),
                    float(line[5]),
                    float(line[6]),
                    float(line[7])
                ),
                bool(line[8]), # isCrowd
                float(line[3]),
                segs[id]
            )
    return retr
