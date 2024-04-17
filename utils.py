def __xywhr2xyxyxyxy(rboxes, width, height):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xywh, rotation] to [xy1, xy2, xy3, xy4]. Rotation values should
    be in degrees from 0 to 90.

    Args:
        rboxes (numpy.ndarray | torch.Tensor): Input data in [cx, cy, w, h, rotation] format of shape (n, 5) or (b, n, 5).
        width (int): Width of the image in pixels
        height (int): Height of the image in pixels

    Returns:
        List : 4 coordinates of the OBB
    """
    cos, sin = (np.cos, np.sin)

    ctr = rboxes[:2]
    w, h, angle = (rboxes[i] for i in range(2, 5))
    angle *= np.pi / 180  # deg -> rad
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2

    x1 = pt1[0] / width
    y1 = pt1[1] / height

    x2 = pt2[0] / width
    y2 = pt2[1] / height

    x3 = pt3[0] / width
    y3 = pt3[1] / height

    x4 = pt4[0] / width
    y4 = pt4[1] / height

    return [x1, y1, x2, y2, x3, y3, x4, y4]


MIN_COLOR = 20
MAX_COLOR = 250


def draw_random_shape(image):
    """
    Main drawing func
    """
    shape_type = np.random.choice(['circle', 'rect'], p = [0.3, 0.7])  # 30% of circles et 70% rectangles

    if shape_type == 'circle':
        cls_ = 0 # Circle label
        xyxyxyxy = _draw_circle(image)
    else:
        cls_ = 1 # Rect label
        randint = random.choice(range(10))
        if randint == 0: # Generate regular rect (1 time out of 10)
            xyxyxyxy = draw_rectangle(image)           
        else:  # Generate thin and long object (9 times out of 10)
            if np.random.choice([0, 1]):
                xyxyxyxy = _draw_horizontal_thin_and_long_rect(image)
            else:
                xyxyxyxy = _draw_tilted_thin_and_long_rect(image) 
    
    label = [cls_] + xyxyxyxy  
    return label


def _draw_circle(image):
    """
    Draw a circle in image of random size and random color.
    It also outputs the xys coordinates of the bounding box of the circle
    """
    height, width = image.shape

    # Generate shape
    range_radius = np.random.choice(['small', 'medium']) # ['small', 'medium', 'large']
    if range_radius == 'small':
        # because 30pixel is smallest pred so we try to detect 30pix/2 = 15pix. With min_radius = 4, we have an area of 16 pixels
        min_radius = 4 
        max_radius = 50
    elif range_radius == 'medium':
        min_radius = 60
        max_radius = min(height, width) // 4
    elif range_radius == 'large':
        min_radius = min(height, width) // 3
        max_radius = min(height, width) // 2
    
    center = (random.randint(max_radius//2, width-max_radius//2), random.randint(max_radius//2, height-max_radius//2))
    radius = random.randint(min_radius, max_radius) 

    color = np.random.randint(MIN_COLOR, MAX_COLOR)
    cv2.circle(image, center, radius, (color), thickness=-1)

    # Generate coordinates
    eps = 1.05  # enlarge bboxes to eps ratio
    x1 = (center[0] - radius * eps) / width
    y1 = (center[1] - radius * eps) / height

    x3 = (center[0] + radius * eps) / width
    y3 = (center[1] + radius * eps) / height

    x2 = x3
    y2 = y1

    x4 = x1
    y4 = y3

    # Always TopLeft coordinates
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def _draw_rectangle(image):
    """
    Draw a rectangle in image of random size, random orientation and random color.
    It also outputs the xys coordinates of the bounding box of the rectangle
    """
    height, width = image.shape

    # Generate shape
    range_size = np.random.choice(['small', 'medium'])  # ['small', 'medium', 'large']
    # because 30pixel is smallest pred so we try to detect 30pix/2 = 15pix. With min_wh = 4, we have an area of 16 pixels
    if range_size == 'small':
        min_wh = 4
        max_wh = 40
    elif range_size == 'medium':
        min_wh = 60
        max_wh = min(height, width) // 4
    elif range_size == 'large':
        min_wh = min(height, width) // 3
        max_wh = min(height, width) // 2
    
    x_cent, y_cent = random.randint(max_wh//2, width-max_wh//2), random.randint(max_wh//2, height-max_wh//2) 
    w = random.randint(min_wh, max_wh) 
    h = random.randint(min_wh, max_wh)  

    angle = np.random.uniform(0, 90)

    color = np.random.randint(MIN_COLOR, MAX_COLOR)
    rotated_rect = cv2.RotatedRect((x_cent, y_cent), (w, h), angle)
    # Get the four corner points of the rotated rectangle
    pts = cv2.boxPoints(rotated_rect)
    cv2.fillPoly(image, [pts.astype(np.int32)], (color))
    
    # Generate coords
    eps = 4  # enlarge bboxes to eps pixels
    [x1, y1, x2, y2, x3, y3, x4, y4] = __xywhr2xyxyxyxy((x_cent, y_cent, w+eps, h+eps, angle), width, height)
    
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def _draw_horizontal_thin_and_long_rect(image):
    """
    Draw a thin and long rectangle in image of random size and random color.
    It also outputs the xys coordinates of the bounding box of the rectangle
    """
    height, width = image.shape

    # Generate shape
    max_h = 5 # In order to have thin objects
    eps = 10  # enlarge bbox height --> this allow to have larger box because it seems that thinner box trigs non detection on yolov8-obb
    # Choose TopLeft
    x1 = 0
    y1 = random.randint(eps/2, height-max_h-eps/2) # Choose TopLeft
    w = width # prend toute la largeur de l image
    h = random.randint(1, max_h+1)  

    color = np.random.randint(MIN_COLOR, MAX_COLOR)
    cv2.rectangle(image, (x1, y1), (x1+w, y1+h), (color), thickness=-1)

    # Generate label
    y1 -= eps/2  # enlarge bbox
    h += eps  # enlarge bbox

    # Normalize
    x1 /= width
    w /= width
    y1 /= height
    h /= height

    # Generate 3 others pts
    x3 = x1 + w
    y3 = y1 + h

    x2 = x3
    y2 = y1

    x4 = x1
    y4 = y3

    return [x1, y1, x2, y2, x3, y3, x4, y4]


def _draw_tilted_thin_and_long_rect(image):
    """
    Draw a thin and long tilted rect of random size, random orientation and random color.
    It also outputs the xys coordinates of the bounding box of the rectangle

    Parameters
    ----------
    image : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    height, width = image.shape

    # Generate shape
    # Choose center of rectangle in a square of eps pixels in the image's center
    eps_cent = 200
    eps_x_cent = random.randint(-eps_cent, eps_cent)
    eps_y_cent = random.randint(-eps_cent, eps_cent)

    x_cent = width /2 + eps_x_cent
    y_cent = height/2 + eps_y_cent

    # Length between 100 to 130 % of max length --> See if manage negative coordinates     
    max_length = np.sqrt( (width-eps_x_cent)**2 + (height-eps_y_cent)**2 )
    w = int(random.uniform(max_length, max_length * 1.3))

    # Thin height
    max_h = 5
    h = random.randint(1, max_h+1)

    # Negative & positive orientation
    if np.random.choice([0, 1]):
        angle = np.random.randint(40, 50)
    else:
        angle = np.random.randint(130, 140)

    color = np.random.randint(MIN_COLOR, MAX_COLOR)
    rotated_rect = cv2.RotatedRect((x_cent, y_cent), (w, h), angle)
    # Get the four corner points of the rotated rectangle
    pts = cv2.boxPoints(rotated_rect)
    cv2.fillPoly(image, [pts.astype(np.int32)], (color))

     # Generate coordinates
    eps = 10  # enlarge bboxes to eps pixels
    [x1, y1, x2, y2, x3, y3, x4, y4] = __xywhr2xyxyxyxy((x_cent, y_cent, w+eps, h+eps, angle), width, height)
    
    return [x1, y1, x2, y2, x3, y3, x4, y4]


np.random.seed(0)

MAX_SHAPES_PER_IMG = 4

def remove_all_files(folder_path):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        # Iterate over each file and remove it
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

        print(f"All files in '{folder_path}' successfully removed.")
    except Exception as e:
        print(f"Error removing files: {e}")

def generate_images(num_images, size):
    for i in tqdm(range(num_images)):
        dataset = np.random.choice(['train', 'val', 'test'], p=[0.8, 0.1, 0.1])
        image = np.zeros((size, size), dtype=np.uint8)  # Create black image
        labels = []
        for _ in range(np.random.randint(1, MAX_SHAPES_PER_IMG+1)):
            labels.append(draw_random_shape(image))
        
        labels = pd.DataFrame(labels, columns=['cls'] + [coord + str(ind) for ind in range(1, 5) for coord in ['x', 'y']])
        
        # Write csv
        labels.to_csv(f'../labels/{dataset}/image_{i}.txt', sep=' ', header=False, index=False)

        # Add noise
        std_ = 60
        image += np.abs(np.random.randn(size, size) * std_).astype(np.uint8)
        cv2.imwrite(f'../images/{dataset}/image_{i}.png', image)
