# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import os, sys

def optimize_graph(frozen_graph_path, input_names, output_names, \
                    minimum_segment_size=50, maximum_cached_engines=100, \
                    precision_mode='FP32', max_batch_size=1, output_path='trt_ckpt/', \
                    int8_mode=False
                    ):
    """
    tensorflow graph optimization
    """
    frozen_graph = tf.GraphDef()
    with open(frozen_graph_path, 'rb') as f:
        frozen_graph.ParseFromString(f.read())
        for node in frozen_graph.node:
            if input_names == node.name:
                node.attr['shape'].shape.dimp[1].size = 512
                node.attr['shape'].shape.dimp[2].size = 512
                node.attr['shape'].shape.dimp[3].size = 3

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.67)
    # optionally perform TensorRT optimization
    with tf.Graph().as_default() as tf_graph:
        with tf.Session(config=tf_config) as tf_sess:
            graph_size = len(frozen_graph.SerializeToString())
            num_nodes = len(frozen_graph.node)
            frozen_graph = trt.create_inference_graph(
                input_graph_def=frozen_graph,
                outputs=output_names,
                max_batch_size=max_batch_size,
                precision_mode=precision_mode,
                minimum_segment_size=minimum_segment_size,
                #is_dynamic_op=False,
                #maximum_cached_engines=maximum_cached_engines,
                max_workspace_size_bytes=4000000000
            )
            print("graph_size(MB)(native_tf): %.1f" % (float(graph_size)/(1<<20)))
            print("graph_size(MB)(trt): %.1f" %
                (float(len(frozen_graph.SerializeToString()))/(1<<20)))
            print("num_nodes(native_tf): %d" % num_nodes)
            print("num_nodes(tftrt_total): %d" % len(frozen_graph.node))
            print("num_nodes(trt_only): %d" % len([1 for n in frozen_graph.node if str(n.op)=='TRTEngineOp']))
            if int8_mode:
                frozen_graph = trt.calib_graph_to_infer_graph(frozen_graph)
    # re-enable variable batch size, this was forced to max
    # batch size during export to enable TensorRT optimization
    for node in frozen_graph.node:
        if input_names == node.name:
            node.attr['shape'].shape.dimp[0].size = -1
        print("nodes: ", node.name)
        if str(node.op) == "TRTEngineOp":
            print("trt_op_name:", node.name)

    # write optimized model to disk
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    with open(output_path+'/trt_genral.pb', 'wb') as f:
        f.write(frozen_graph.SerializeToString())
    graph_trt = tf.Graph()

    #for tensor_name in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
    #    print("tensor_name:", tensor_name)
    with graph_trt.as_default():
        tf.import_graph_def(frozen_graph)
        for tensor_name in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()):
            print("tensor_name:", tensor_name)
        _ = tf.summary.FileWriter(output_path + '/', graph_trt)
if __name__ == "__main__":
    frozen_graph_path = sys.argv[1]
    output_path = sys.argv[2]
    input_names = 'input_image:0'
    output_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']
    optimize_graph(
        frozen_graph_path=frozen_graph_path,
        input_names=input_names,
        output_names=output_names,
        output_path=output_path
        )
