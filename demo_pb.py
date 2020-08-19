# coding=utf-8

import sys

import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
# from icdar import point_inside_of_quad
from icdar import restore_rectangle, shrink_poly
import locality_aware_nms as nms_locality
import lanms
# import config as cfg
from PIL import Image, ImageDraw

tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('pb_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

import model
from icdar import restore_rectangle
# from nms import nms, border_nms

FLAGS = tf.app.flags.FLAGS

def intersection(g, p):
     g = Polygon(g[:8].reshape((4, 2)))
     p = Polygon(p[:8].reshape((4, 2)))
     if not g.is_valid or not p.is_valid:
         return 0
     inter = Polygon(g).intersection(Polygon(p)).area
     union = g.area + p.area - inter
     if union == 0:
         return 0
     else:
         return inter/union, g, p


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=512):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(resize_h, 32)
    resize_w = max(resize_w, 32)

    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def resize(im, target_size=512, max_size=1024, stride=32, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, (im_scale, im_scale)


def detect(score_map, geo_map, timer, score_map_thresh=0.6, box_thresh=0.1, nms_thres=0.3):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map, default 0.8
    :param box_thresh: threshhold for boxes, default 0.1
    :param nms_thres: threshold for nms, default 0.2
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, :]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32)//4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer

def generate_boxes_from_map(sotd_map):
    print("sotd-map_shape:", sotd_map.shape)

    text_area = sotd_map[:,:]
    text_area[text_area>0.5] = 1
    print("text_area_shape:", text_area.shape)
    polys = []
    boxes = []
    # text_area = np.bitwise_or(center_map, border_map)
    image, contours, hierarchy = cv2.findContours(text_area.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
         rotrect = cv2.minAreaRect(contour)
         box = cv2.boxPoints(rotrect)
         box = np.int0(box)
         polys.append(box)

    for poly in polys:
         r = [None, None, None, None]
         poly = poly*4.
         for i in range(4):
             r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                        np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
         # score map
         shrinked_poly = shrink_poly(poly.copy(), r, R=-0.5)
         boxes.append(shrinked_poly)
    return np.asarray(boxes, np.float32)


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def py_mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.shape[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = np.split(images, num_channels, axis=-1)
    for i in range(num_channels):
        channels[i] -= means[i]
    return np.concatenate(channels, axis=-1)

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        pb_file_path = FLAGS.pb_path
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            out_nodes = [
                    'feature_fusion/Conv_7/Sigmoid:0',
                    'feature_fusion/concat_3:0'
            ]

            input_images = sess.graph.get_tensor_by_name("input_images:0")
            f_score = sess.graph.get_tensor_by_name(out_nodes[0])
            # f_border = sess.graph.get_tensor_by_name(out_nodes[1])
            f_geometry = sess.graph.get_tensor_by_name(out_nodes[1])


            im_fn_list = get_images()
            avg_times = {'count':0, 'duration':0, 'net':0, 'nms':0}
            for im_fn in im_fn_list:
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                    start_time = time.time()
                    im_resized, (ratio_h, ratio_w) = resize_image(im, 1024) #512)

                    #im_resized = py_mean_image_subtraction(im_resize.astype(np.float))
                except:
                    continue
                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                # geometry = sess.run([f_geometry], feed_dict={input_images: [im_resized]})
                #im_resized = np.random.normal(size=(1, 512, 512 , 1))
                score,  geometry = sess.run([f_score, f_geometry], 
                                            feed_dict={input_images: [im_resized]})
                
                timer['net'] = time.time() - start
                nms_start = time.time()
                # geometry= np.squeeze(geometry, axis=0)
                # cond = np.greater_equal(geometry[:, :, 0], cfg.pixel_threshold)
                # activation_pixels = np.where(cond)
                # scores, boxes = nms(geometry, activation_pixels)
                '''
                score_map = np.squeeze(score_map, axis=0)
                cv2.imwrite('result/score_map.png', score_map * 255)
                border_map = np.squeeze(border_map, axis=0)
                cv2.imwrite('result/border_map.png', border_map[:, :, :3] * 255)
                geometry = np.squeeze(geometry, axis=0)
                cv2.imwrite('result/geo_map.png', geometry[:, :, 0] * 255)
                cond = np.greater_equal(score_map, cfg.pixel_threshold)
                activation_pixels = np.where(cond)
                scores, boxes = border_nms(score_map, activation_pixels, border_map, geometry,
                                           threshold=cfg.side_vertex_pixel_threshold)
                '''
                # print score_map.shape, geometry.shape
                timer['nms'] = time.time() - nms_start
                #boxes, timer = detect(score_map=score_map, geo_map=geometry, timer=timer)
                geo_boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                #print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                #     im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                #sotd_img = np.array(score[0, :, :, :] * 255).astype(np.uint8)
                #sotd_img = cv2.resize(sotd_img, dsize=(im_resized.shape[1], im_resized.shape[0]))/255.
                #sotd_img = score[0, :, :, :]
                #score_boxes = generate_boxes_from_map(sotd_img)

                #boxes = []
                #if geo_boxes is None:
                #     continue
                # post process with score_map
                '''
                indexes = []
                for i, gbox in enumerate(geo_boxes):
                     max_iou = 0
                     max_idx = -1
                     tbox = gbox
                     for sbox in score_boxes:
                         try:
                             iou, gb, sb = intersection(gbox, sbox)
                         except:
                             continue

                         if iou > 0.3:
                             if iou > max_iou:
                                 max_iou = iou
                                 max_idx = i
                                 #ubox = polygon.orient(gb.union(sb))
                                 #print list(ubox)
                                 #boxes.append(ubox)
                                 #print("sbox:", sbox, "gbox:", gbox)
                                 print("iou:", iou)
                                 xmin, ymin = np.amin(sbox, axis=0)
                                 xmax, ymax = np.amax(sbox, axis=0)
                                 h, w = ymax - ymin, xmax - xmin
                                 if h <= w:
                                     #gbox = geo_boxes[max_idx]
                                     tbox[0] = np.minimum(gbox[0], xmin)
                                     tbox[6] = np.minimum(gbox[6], xmin)
                                     tbox[2] = np.maximum(gbox[2], xmax)
                                     tbox[4] = np.maximum(gbox[4], xmax)
                                 if h > w:
                                     tbox[1] = np.minimum(gbox[1], ymin)
                                     tbox[3] = np.minimum(gbox[3], ymin)
                                     tbox[7] = np.maximum(gbox[7], ymax)
                                     tbox[5] = np.maximum(gbox[5], ymax)
                     boxes.append(tbox)
                boxes = np.array(boxes)
                boxes = boxes if boxes.shape[0] else geo_boxes
                '''
                boxes = geo_boxes
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

                if boxes is not None:
                    # idx = np.where(np.amin(scores, axis=1) > 0)[0]
                    # boxes = boxes[idx]
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                duration = time.time() - start_time
                print('[timing] {}'.format(duration))
                #print "boxes:", boxes
                avg_times['duration'] += duration
                avg_times['nms'] += timer['nms']
                avg_times['net'] += timer['net']
                avg_times['count'] += 1

                # save to file
                if boxes is not None:
                    dir_name = os.path.basename(os.path.dirname(im_fn))
                    if not os.path.exists(os.path.join(FLAGS.output_dir, dir_name)):
                        os.mkdir(os.path.join(FLAGS.output_dir, dir_name))
                    tmp_name = os.path.basename(im_fn).split('.')
                    tmp_name[-1] = 'txt'
                    txt_name = '.'.join(tmp_name)
                    res_file = os.path.join(FLAGS.output_dir, dir_name, txt_name)

                    #res_file = os.path.join(
                    #    FLAGS.output_dir,
                    #    '{}.txt'.format(
                    #        os.path.basename(im_fn).split('.')[0]))

                    with open(res_file, 'w') as f:
                        for box in boxes:
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            '''
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            '''
                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                            ))

                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                          color=(0, 255, 0), thickness=1)
                if not FLAGS.no_write_images:
                    dir_name = os.path.basename(os.path.dirname(im_fn))
                    if not os.path.exists(os.path.join(FLAGS.output_dir, dir_name)):
                        os.mkdir(os.path.join(FLAGS.output_dir, dir_name))
                    img_name = os.path.basename(im_fn)
                    img_path = os.path.join(FLAGS.output_dir, dir_name, img_name) 
                    #cv2.imwrite(img_path, im[:, :, ::-1])
                    cv2.imencode('.jpg', im)[1].tofile(img_path)

                #if not FLAGS.no_write_images:
                #    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                #    cv2.imwrite(img_path, im[:, :, ::-1])
            avg_times['duration'] /= avg_times['count']
            avg_times['net'] /= avg_times['count']
            avg_times['nms'] /= avg_times['count']
            print("mean_time: {:.0f}ms, nms mean time: {:.0f}, net mean time: {:.0f}".format(
                avg_times['duration']*1000, avg_times['nms']*1000, avg_times['net']*1000))
            

if __name__ == '__main__':
    tf.app.run()
