# coding:utf-8
import glob
import csv
import cv2
import time
import os
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon

import tensorflow as tf
from data_utils_mt import GeneratorEnqueuer

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_string('training_data_path', './train_data/images/',
                           'training dataset to use')
tf.app.flags.DEFINE_string('training_gt_path', './train_data/gts/',
                           'training dataset to use')
tf.app.flags.DEFINE_integer('min_text_size', 10,
                            'if the text size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.6, 
                          'when doing random crop from input image, the min length of min(H, W)')
tf.app.flags.DEFINE_string('geometry', 'RBOX',
                           'which geometry to generate, RBOX or QUAD')
tf.app.flags.DEFINE_float('poly_shrink_ratio', 0.3, 
                        'shrink ratio for input ground bounding boxes')

FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    for ext in ['jpg', 'png', 'JPG']:
        files.extend(glob.glob(os.path.join(FLAGS.training_data_path, '*.{}'.format(ext))))
    # print('files length is: %d' % len(files) )
    return files


def load_annoataion(gtfile):
    """
    load annotation from the text file
    :return: text_polys [N,4,2], text_tags [N] 
    """
    text_polys, text_tags = [], []
    if not os.path.exists(gtfile):
        return np.array([], dtype=np.float32), np.array([], dtype=np.bool)

    with open(gtfile, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            if not line: continue
            label = line[-1]
            # strip BOM. \ufeff for python3, \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            text_tags.append(True if label=='*' or label=='###' else False)
            #print(text_polys, label)
    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


def polygon_area(poly):
    # compute area of a polygon
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.


def order_clockwise(poly):
    cent = np.mean(poly, axis=0)
    poly_cented = poly - cent
    poly_angles = [math.atan2(p[1], p[0]) for p in poly_cented]
    return poly[np.argsort(poly_angles)]


def check_and_validate_polys(polys, tags, input_image_size):
    """
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    """
    (h, w) = input_image_size
    if polys.shape[0] == 0:
        return polys, tags
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys, validated_tags = [], []
    p_areas = [polygon_area(poly) for poly in polys]

    for poly, p_area, tag in zip(polys, p_areas, tags):
        if abs(p_area) < 10: continue
        # polygon in clock-wise order. p_area<0
        if p_area > 0:
            poly = order_clockwise(poly)
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(im, polys, tags, crop_background=False, max_tries=20):
    """
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    """
    if polys.shape[0]==0:
        return im, polys, tags

    h, w, _ = im.shape
    pad_h, pad_w = h//10, w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx, maxx = np.min(poly[:, 0]), np.max(poly[:, 0])
        miny, maxy = np.min(poly[:, 1]), np.max(poly[:, 1])
        w_array[minx+pad_w : maxx+pad_w] = 1
        h_array[miny+pad_h : maxy+pad_h] = 1

    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        yy = np.random.choice(h_axis, size=2)
        xmin, xmax = np.min(xx) - pad_w, np.max(xx) - pad_w
        xmin, xmax = np.clip(xmin, 0, w-1), np.clip(xmax, 0, w-1)
        ymin, ymax = np.min(yy) - pad_h, np.max(yy) - pad_h
        ymin, ymax = np.clip(ymin, 0, h-1), np.clip(ymax, 0, h-1)
        # Too small crop, just ignore.
        if xmax - xmin < FLAGS.min_crop_side_ratio*max(FLAGS.input_size, w) or \
                ymax - ymin < FLAGS.min_crop_side_ratio*max(FLAGS.input_size, h):
            continue

        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []

        # no text in this area
        if len(selected_polys) == 0:
            if crop_background:
                return im[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        # cropped im, polys, tags.
        im = im[ymin:ymax+1, xmin:xmax+1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags


def shrink_poly(poly, R=FLAGS.poly_shrink_ratio):
    """
    fit a poly inside the origin poly, used for generate the score map
    """
    poly = poly.astype(np.float32)
    side_lens = []
    for i in range(4):
        side_len = np.linalg.norm(poly[(i+1) % 4] - poly[i]) + 0.001
        side_lens.append(side_len)
    r = [None, None, None, None]
    for i in range(4):
        r[i] = min(side_lens[i], side_lens[(i+3) % 4])
    offset0 = r[0] * ((poly[1] - poly[0])/side_lens[0] + (poly[3]-poly[0])/side_lens[3])
    offset1 = r[1] * ((poly[0] - poly[1])/side_lens[0] + (poly[2]-poly[1])/side_lens[1])
    offset2 = r[2] * ((poly[1] - poly[2])/side_lens[1] + (poly[3]-poly[2])/side_lens[2])
    offset3 = r[3] * ((poly[0] - poly[3])/side_lens[3] + (poly[2]-poly[3])/side_lens[2])
    shrinked_poly = poly + R * np.stack([offset0, offset1, offset2, offset3], 0)
    return shrinked_poly


def point_dist_to_line(vec, dir_vec):
    # assert dir_vec is normalized
    return np.linalg.norm(np.cross(vec, dir_vec))


def order_rotate_rectangle(poly):
    # Only used when geometry == 'RBOX':
    # rbox is clock-wise, with -90<angle<=0
    # more details of cv2.minAreaRect can be found in notebooks
    rect = cv2.minAreaRect(poly)
    rbox = np.array(cv2.boxPoints(rect))
    angle = rect[-1]
    assert (angle <= 0)
    if angle < -45.:
        idx_start = np.argmin(rbox[:, 1])
        angle = -angle - 90.
    else:  # [-45,0)
        idx_start = np.argmin(rbox[:, 0])
        angle = -angle

    rbox = rbox[[idx_start, (idx_start+1)%4, (idx_start+2)%4, (idx_start+3)%4]]
    angle = angle * np.pi / 180.
    return rbox, angle


def generate_rbox(im_size, polys, tags):
    h, w = im_size
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    # mask used during training, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
    for poly_idx, (poly, tag) in enumerate(zip(polys, tags)):
        shrinked_poly = shrink_poly(poly).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx+1)
        # if the poly is too small, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        if min(poly_h, poly_w) < FLAGS.min_text_size:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        rbox, rotate_angle = order_rotate_rectangle(poly)
        p0_rect, p1_rect, p2_rect, p3_rect = rbox
        
        dir_p01 = (p1_rect - p0_rect) / np.linalg.norm(p1_rect - p0_rect)
        dir_p12 = (p2_rect - p1_rect) / np.linalg.norm(p2_rect - p1_rect)
        dir_p23 = (p3_rect - p2_rect) / np.linalg.norm(p3_rect - p2_rect)
        dir_p30 = (p0_rect - p3_rect) / np.linalg.norm(p0_rect - p3_rect)

        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(point - p0_rect, dir_p01)
            # right
            geo_map[y, x, 1] = point_dist_to_line(point - p1_rect, dir_p12)
            # down
            geo_map[y, x, 2] = point_dist_to_line(point - p2_rect, dir_p23)
            # left
            geo_map[y, x, 3] = point_dist_to_line(point - p3_rect, dir_p30)
            # angle
            geo_map[y, x, 4] = rotate_angle
    return score_map, geo_map, training_mask


def generator(input_size=FLAGS.input_size,
              batch_size=32,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 2.0, 3.0]),
              vis=False):
    image_list = np.array(get_images())
    print('{} training images in {}'.format(image_list.shape[0], FLAGS.training_data_path))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        images = []
        image_fns = []
        score_maps = []
        geo_maps = []
        training_masks = []
        for i in index:
            try:
                im_fn = image_list[i]
                try:
                    im = cv2.imread(im_fn)
                    im = im[:,:,::-1]
                except TypeError as e:
                    continue

                h, w, _ = im.shape
                txt_fn = os.path.join(FLAGS.training_gt_path, os.path.basename(im_fn)[:-4]+'.txt')
                if not os.path.exists(txt_fn):
                    print('GT %s not exist!'%txt_fn)

                text_polys, text_tags = load_annoataion(txt_fn)
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
                # random scale this image
                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
                text_polys *= rd_scale
                # random crop a area from image
                if text_polys.shape[0]==0 or np.random.rand() < background_ratio:
                    # crop background
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                else:
                    # crop foreground
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)

                # background find
                if text_polys.shape[0] == 0:
                    # pad and resize image
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = cv2.resize(im_padded, dsize=(input_size, input_size))
                    score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                    geo_map = np.zeros((input_size, input_size, 5), dtype=np.float32)
                    training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                else: # foreground find
                    h, w, _ = im.shape
                    # pad the image to the training input size or the longer side of image
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = im_padded
                    # resize the image to input size
                    new_h, new_w, _ = im.shape
                    resize_h = input_size
                    resize_w = input_size
                    im = cv2.resize(im, dsize=(resize_w, resize_h))
                    resize_ratio_3_x = resize_w/float(new_w)
                    resize_ratio_3_y = resize_h/float(new_h)
                    text_polys[:, :, 0] *= resize_ratio_3_x
                    text_polys[:, :, 1] *= resize_ratio_3_y
                    new_h, new_w, _ = im.shape
                    score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)

                if vis:
                    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
                    axs[0, 0].imshow(im)
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    for poly in text_polys:
                        poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                        poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                        axs[0, 0].add_artist(Patches.Polygon(
                            poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                        axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
                    axs[0, 1].imshow(score_map[::, ::])
                    axs[0, 1].set_xticks([])
                    axs[0, 1].set_yticks([])
                    axs[1, 0].imshow(geo_map[::, ::, 0])
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    axs[1, 1].imshow(geo_map[::, ::, 1])
                    axs[1, 1].set_xticks([])
                    axs[1, 1].set_yticks([])
                    axs[2, 0].imshow(geo_map[::, ::, 2])
                    axs[2, 0].set_xticks([])
                    axs[2, 0].set_yticks([])
                    axs[2, 1].imshow(training_mask[::, ::])
                    axs[2, 1].set_xticks([])
                    axs[2, 1].set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                #print('one sample has done!')
                images.append(im.astype(np.float32))
                image_fns.append(im_fn)
                score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                if len(images) == batch_size:
                    yield images, image_fns, score_maps, geo_maps, training_masks
                    images = []
                    image_fns = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    #print('sleeping')
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    gen = generator(vis=False, batch_size=10)
    i = 0
    while i < 1000:
        data = next(gen)
        #data = gen.next()
        i+=1
        #print(data)
        #print len(data)

