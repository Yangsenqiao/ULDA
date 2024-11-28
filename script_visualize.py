import json


from matplotlib.path import Path
import numpy as np
from PIL import Image


# Function to calculate the area of the polygon given its vertices
def polygon_area(polygon):
    x = np.array([vertex[0] for vertex in polygon])
    y = np.array([vertex[1] for vertex in polygon])
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Define the CityscapesClass
class CityscapesClass:
    def __init__(self, name, id, color):
        self.name = name
        self.id = id
        self.color = color

# Create a list of CityscapesClass objects
classes = [
    CityscapesClass('road',                  0,  (128, 64, 128)),
    CityscapesClass('sidewalk',              1,  (244, 35, 232)),
    CityscapesClass('building',              2,  (70, 70, 70)),
    CityscapesClass('wall',                  3,  (102, 102, 156)),
    CityscapesClass('fence',                 4,  (190, 153, 153)),
    CityscapesClass('pole',                  5,  (153, 153, 153)),
    CityscapesClass('traffic light',         6,  (250, 170, 30)),
    CityscapesClass('traffic sign',          7,  (220, 220, 0)),
    CityscapesClass('vegetation',            8,  (107, 142, 35)),
    CityscapesClass('terrain',               9,  (152, 251, 152)),
    CityscapesClass('sky',                   10,  (70, 130, 180)),
    CityscapesClass('person',                11, (220, 20, 60)),
    CityscapesClass('rider',                 12,  (255, 0, 0)),
    CityscapesClass('car',                   13,  (0, 0, 142)),
    CityscapesClass('truck',                 14,  (0, 0, 70)),
    CityscapesClass('bus',                   15,  (0, 60, 100)),
    CityscapesClass('train',                 16,  (0, 80, 100)),
    CityscapesClass('motorcycle',            17, (0, 0, 230)),
    CityscapesClass('bicycle',               18,  (119, 11, 32)),
    CityscapesClass('ignore',                255,  (0, 0, 0)),
]
def process_annotation(img_path, json_path):
    
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    # Load the image
    img = Image.open(img_path)
    img_array = np.array(img)



    shapes = sorted(json_data['shapes'], key=lambda s: polygon_area(s['points']), reverse=True)

    # Initialize an array for the label image
    label_image = np.full((img_array.shape[0], img_array.shape[1]), 255, dtype=np.uint8)

    # Mapping from class names to CityscapesClass objects
    class_mapping = {cls.name: cls for cls in classes}

    # Fill each region on the image
    for shape in shapes:
        # Get the class for the current shape
        cls = class_mapping.get(shape['label'])
        if not cls:
            continue  # Skip if class not found

        # Create a mask for the current shape
        mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.bool_)
        poly_path = Path(shape['points'])
        x, y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
        coords = np.vstack((x.flatten(), y.flatten())).T
        mask[y, x] = poly_path.contains_points(coords).reshape(mask.shape)

        # Apply the color to the image and label to the label image
        img_array[mask] = cls.color
        label_image[mask] = cls.id

    # Convert the manipulated arrays back to images
    img_with_color = Image.fromarray(img_array)
    label_image_pil = Image.fromarray(label_image)
    return img_with_color, label_image_pil



if __name__ == '__main__':
    raw_img_pth = 'sand/imgs/23.jpg'
    json_path = 'sand/json/23.json'
    label_image_path = 'sand/visualize/23_label.png'
    color_image_path =  'sand/visualize/23_color.png'
    img_color, img_label = process_annotation(img_path=raw_img_pth, json_path=json_path)
    img_label.save(label_image_path)
    img_color.save(color_image_path)

