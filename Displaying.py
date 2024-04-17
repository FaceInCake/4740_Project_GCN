
from Parsing import ObjectSegmentation
from cv2 import Mat, rectangle as draw_rectangle, putText, FONT_HERSHEY_PLAIN, getTextSize

def draw_obj (img:Mat, obj:ObjectSegmentation):
    COLOUR = (255, 0, 255)
    x0 = round(obj.boundingBox.x / obj.image.width * img.shape[0])
    y0 = round(obj.boundingBox.y / obj.image.height * img.shape[1])
    x1 = round(obj.boundingBox.width / obj.image.width * img.shape[0])
    y1 = round(obj.boundingBox.height / obj.image.height * img.shape[1])
    temp = img.copy()
    draw_rectangle(temp, (x0, y0, x1, y1), COLOUR, 1)
    (tw, th), _ = getTextSize(obj.category.name, FONT_HERSHEY_PLAIN, 1, 1)
    draw_rectangle(temp, (x0, y0-th, tw, th), COLOUR, -1)
    putText(temp, obj.category.name, (x0, y0), FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1, bottomLeftOrigin=False)
    return temp
    

