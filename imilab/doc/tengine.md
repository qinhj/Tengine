## API ##
```
c_api.c:

1. create_graph
-> create_context
  -> context->scheduler = find_default_scheduler(); // scheduler/scheduler.c: sync_scheduler
  -> context->device = find_default_device();   // module/module.c: struct device*, default: CPU
-> find_serializer_via_name:
  source/serializer/tmfile/tm2_serializer.c:
    load_model -> load_graph -> load_graph_nodes:
      -> find_op_loader
      -> create_ir_node:
        operator/prototype/pad.c: init_op
        operator/prototype/interp.c: init_op
      -> ir_node->name = ...
      -> set_ir_node_input_tensor(ir_node, j, ir_tensor);
      -> set_ir_node_output_tensor(ir_node, j, ir_tensor);
      -> ... /* axis param from nhwc to nchw */
      -> e->loader(ir_graph, ir_node, tm_node, tm_operator):
        serializer/tmfile/op/tm2_pad.c: tm2_load_pad
        serializer/tmfile/op/tm2_interp.c: tm2_load_interp

2. prerun_graph_multithread -> infer_ir_graph_shape:
  graph/graph.c:
    infer_ir_graph_shape -> get_ir_graph_node(graph, i) -> op->infer_shape(node):
      operator/prototype/pad.c: infer_shape
      operator/prototype/interp.c: infer_shape
  device/cpu/op/interp/interp_ref.c: ...

3. run_graph -> scheduler->run(scheduler, ir_graph, block)  // default: call sched_run
```

## OP ##
```
* reshape
  The reshape op in tengine is a constant op, which means it won't be updated
automatically if one reset the input tensor shape of the graph(e.g. yolov3/...).
  So one have no choice, but to re-export these models with target input size
and convert to tmfile again, if the model contains the reshape op. Or one can
update all reshape op by hand ...
  For models without reshape op, one can reset the input tensor shape directly
and test more(e.g. mobilenet_ssd/...).

* softmax
softmax_ref.c: run -> ref_softmax_fp32/uint8/int8
softmax_kernel_ref_fp32.c: ref_softmax_fp32 -> GetMaxArray
```

## Data ##
```
1. It seems that once graph ran, one can only get the output tensor data, i.e. inplace.
All internal nodes' tensor buffer data are gone, it's ok. To debug the feature map, one
may need to reset the output tensor node or find some runtime callback api.
```

## Onnx ##
```
1. It seems that tengine doesn't support onnxsim dynamic_input_shape;
```

## Optimize ##
```
CPU:
»ã±à
im2col + sgemm
direct
winograd
int8

GPU:
cuda
tensorRT

NPU:
tim-vx
```