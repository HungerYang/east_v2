# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import slim
import os
import sys
import tensorflow as tf
import model
from tensorflow.python.framework import graph_util
from model import mean_image_subtraction as ms

sys.path.append(os.getcwd())
if __name__ == "__main__":
    print('usage: python generate_pb.py ckpt_path graph_out_path gpu_id')
    print("hint: ckpt_path such as 'border_ckpt/model.ckpt-7071' ")
    assert len(sys.argv) > 3, ''
    ckpt_path = sys.argv[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        # f_geometry = model.advanced_model(input_images, is_training=False)
        f_score, f_geometry = model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # ckpt_state = tf.train.get_checkpoint_state(ckpt_path)
            # model_path = os.path.join(ckpt_path, os.path.basename(ckpt_state.model_checkpoint_path))

            model_path = ckpt_path
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            node_names = [node.name for node in sess.graph_def.node]
            for x in node_names:
                if 'feature_fusion' in x or 'concat' in x:
                    print x

            output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, [
                'feature_fusion/Conv_7/Sigmoid',
                'feature_fusion/concat_3'
                ])

            output_graph = sys.argv[2]
            print("start write into file:", output_graph)
            with tf.gfile.GFile(output_graph, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            sess.close()
