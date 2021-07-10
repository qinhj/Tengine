# -*- coding: utf-8 -*-

# OPEN AI LAB is pleased to support the open source community by supporting Tengine available.
#
# Copyright (C) 2021 OPEN AI LAB. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


"""
This tool for optimizing the network structure of YOLOv5s from
https://github.com/ultralytics/yolov5

1. Replace the focus nodes by reshape and transpose nodes if necessary;
2. Remove the YOLO detection nodes of postprocess;
3. Fusion the activation HardSwish node replace the Sigmoid and Mul;
4. Update input/output tensor.

This tool is based on ONNX Framework.
Usage:
$ python3 yolov5s-opt.py --input yolov5s.v4.onnx --output yolov5s.v4.opt.onnx --in_tensor 167 --out_tensor 381,420,459
$ python3 yolov5s-opt.py --input yolov5s.v5.onnx --output yolov5s.v5.opt.onnx --in_tensor 167 --out_tensor 397,458,519
$ python3 yolov5s-opt.py --input yolov5s.v5.onnx --output yolov5s-p3p4.opt.onnx --in_tensor 167 --out_tensor 397,458

Author:
    xwwang@openailab.com, initial
    hhchen@openailab.com, update
    qinhj@lsec.cc.ac.cn, update
"""

import numpy as np
import onnx
import argparse

from onnxsim import simplify


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv5 Optimize Tool Parameters')
    
    parser.add_argument('--input', help='input model path', default='./yolov5s.onnx', type=str)  
    parser.add_argument('--output', help='output model path', default='./yolov5s-opt.onnx', type=str)
    parser.add_argument('--in_tensor', help='input tensor name', default='', type=str)
    parser.add_argument('--out_tensor', help='output tensor names', default='381,420,459', type=str)
    parser.add_argument('--img_size', type=int, default=640, help='input image size')
    parser.add_argument('--verbose', action='store_true', help='show verbose info')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    
    args = parser.parse_args()
    return args


args = parse_args()


def del_node_by_tensor(input_node, in_name, out_name):
    """
    Brief: del the node based on input and output tensor
    Args:
        input_node: the nodes of ONNX model
        in_name:    input tensor value name
        out_name:   output tensor value names
    """
    # map: output tensor name -> node index
    node_dict = {}
    for i in range(len(input_node)):
        for o in input_node[i].output:
            node_dict[o] = i
    if args.verbose:
        print("[Verbose] node_dict:", node_dict)
    
    # assign output nodes as 2
    output_pass = np.zeros((len(input_node)), dtype=np.int)
    for i in range(len(out_name)):
        output_pass[node_dict[out_name[i]]] = 2
    #if args.verbose:
    #    print("[Verbose] output_pass:", output_pass)

    for i in range(len(input_node)):
        for name in input_node[i].input:
            if name in node_dict:
                pre_node = node_dict[name] # previous node index
                if output_pass[pre_node] == 2 or output_pass[pre_node] == 1:
                    # previous node is output maker or child
                    for o in input_node[i].output:
                        output_pass[node_dict[o]] = 1
    if args.verbose:
        print("[Verbose] output_pass:", output_pass)

    # del all child nodes of output nodes
    for i in range(len(output_pass)-1, -1, -1):
        if output_pass[i] == 1:
            del input_node[i]

    # del all nodes before input node(for simplicity)
    if in_name in node_dict:
        idx = node_dict[in_name]
        for i in range(idx, -1, -1):
            del input_node[i]

    return input_node


def fusion_hardswish(input_node):
    """
    using HardSwish replace the Sigmoid and Mul
    Args:
        input_node: the nodes of ONNX model
    Returns:
        the new node
    """     
    del_list = []
    for i in range(len(input_node) - 1):
        if (input_node[i].op_type == 'Sigmoid' and input_node[i+1].op_type == 'Mul'):
            input_node[i].output[0] = input_node[i+1].output[0]
            input_node[i].op_type = 'HardSwish'
            del_list.append(i + 1)

    for i in range(len(del_list)-1, -1, -1):
        del input_node[del_list[i]]

    return input_node


def new_node_reshape(name, data, shape, out=None):
    """
    Description: Create new reshape node.
    Parameters:
        name: node name
        data: input tensor name
        shape: input shape list
        out: output tensor name
    Return:
        new node and tensor for shape
    Example:
        >>> name = "Reshape_Image"
        >>> shape = [1,3,320,2,320,2]
        >>> node, tp = new_node_reshape(name, "images", shape)
    """
    tensor_in = data + "_shape"
    tensor_out = out if out else data + "_reshaped"
    tensor = onnx.helper.make_tensor(tensor_in, onnx.TensorProto.INT64, [len(shape)], shape)
    node = onnx.helper.make_node("Reshape", inputs=[data, tensor_in], outputs=[tensor_out], name=name)
    return node, tensor


def new_node_transpose(name, perm, tensor_in, tensor_out):
    """
    Description: Create new transpose node.
    Parameters:
        name: node name
        perm: axes tuple
        tensor_in: input tensor name
        tensor_out: output tensor name
    Example:
        >>> name = "Transposed_SpaceToDepth"
        >>> perm = (0,3,5,1,2,4)
        >>> t_in = "reshaped_input"
        >>> t_out= "transposed_input"
        >>> node = new_node_transpose(name, perm, t_in, t_out)
    """
    attr = {"perm": perm}
    return onnx.helper.make_node("Transpose", inputs=[tensor_in], outputs=[tensor_out], name=name, **attr)


def keep_or_del_elem(obj, elem_name_list, keep=False):
    """
    keep/delete elem from input objectes
    """
    del_elem_list = []

    for i, n in enumerate(obj):
        if (n.name in elem_name_list and not keep) or (n.name not in elem_name_list and keep):
            del_elem_list.append(i)
    #print("del elem list:", del_elem_list)

    ## delete nodes safely: from end to start
    del_elem_list.reverse()
    [obj.pop(i) for i in del_elem_list]
    return del_elem_list


def usage_info():
    """
    usage info
    """
    print("Input params is illegal...╮(╯3╰)╭")
    print("try it again:\n python yolov5s-opt.py -h")


def main():
    """
    main function
    """
    print("---- Tengine YOLOv5 Optimize Tool ----\n")

    if args == None or args.input == None:
        usage_info()
        return None

    print("Input model      : %s" % (args.input))
    print("Output model     : %s" % (args.output))
    print("Input tensor     : %s" % (args.in_tensor))
    print("Output tensor    : %s" % (args.out_tensor))

    in_tensor = args.in_tensor #.split(',')
    out_tensor = args.out_tensor.split(',')

    # load original onnx model, graph, nodes
    print("[Quant Tools Info]: Step 0, load original onnx model from %s." % (args.input))
    onnx_model = onnx.load(args.input)
    onnx_model, check = simplify(onnx_model,
                                 dynamic_input_shape=args.dynamic,
                                 input_shapes={'images': [1, 3, 640, 640]} if args.dynamic else None)

    graph = onnx_model.graph

    # cut the focus and postprocess nodes
    print("[Quant Tools Info]: Step 1, Remove the focus and postprocess nodes.")
    del_node_by_tensor(graph.node, in_tensor, out_tensor)

    # op fusion, using HardSwish replace the Sigmoid and Mul
    print("[Quant Tools Info]: Step 2, Using hardswish replace the sigmoid and mul.")
    fusion_hardswish(graph.node)

    # add new reshape and transpose nodes
    # Note: Nodes in a graph must be topologically sorted!
    if in_tensor:
        print("[Quant Tools Info]: Step 3, Add new reshape and transpose nodes.")
        name = "Reshape_Image"
        shape = [1, 3, args.img_size // 2, 2, args.img_size // 2, 2]
        node_r, tp = new_node_reshape(name, graph.input[0].name, shape)
        graph.node.insert(0, node_r), graph.initializer.append(tp)
        name = "Transpose_Image"
        perm = (0, 5, 3, 1, 2, 4) # (0, 3, 5, 1, 2, 4)
        node_t = new_node_transpose(name, perm, node_r.output[0], "transposed_image")
        graph.node.insert(1, node_t)
        name = "Reshape_Image_Focus"
        shape = [1, 12, args.img_size // 2, args.img_size // 2]
        node_r, tp = new_node_reshape(name, node_t.output[0], shape, in_tensor)
        graph.node.insert(2, node_r), graph.initializer.append(tp)
    else:
        print("[Quant Tools Info]: Step 3, Keep slice op.")

    # get input/output tensor index of value info
    #in_tensor_idx = [None] * len(in_tensor)
    out_tensor_idx = [None] * len(out_tensor)
    value = graph.value_info
    for i, v in enumerate(value):
        #if v.name in in_tensor:
        #    in_tensor_idx[in_tensor.index(v.name)] = i
        if v.name in out_tensor:
            out_tensor_idx[out_tensor.index(v.name)] = i
    print("[Quant Tools Info]: Step 4, Update input and output tensor.")

    #keep_or_del_elem(onnx_model.graph.input, in_tensor, True)
    #for i in in_tensor_idx:
    #    if i:
    #        onnx_model.graph.input.append(value[i])
    keep_or_del_elem(onnx_model.graph.output, out_tensor, True)
    for i in out_tensor_idx:
        if i: onnx_model.graph.output.append(value[i])

    # save the new optimize onnx model
    print("[Quant Tools Info]: Step 5, save the new onnx model to %s." % (args.output))
    onnx.save(onnx_model, args.output)

    print("\n---- Tengine YOLOv5s Optimize onnx create success, best wish for your inference has a high accuracy ...\\(^0^)/ ----")


if __name__ == "__main__":
    main()
