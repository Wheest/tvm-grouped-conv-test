## Generate grouped models

``` sh
export MODEL_DIR=~/models/onnx/gspc/resnet34
python prepare_models.py --output_dir $MODEL_DIR

export MODEL_DIR=~/models/onnx/gspc/wrn-40-2_cifar10
python prepare_models.py --model_set wrn-40-2_cifar10 --output_dir $MODEL_DIR
```

## Run grouped models

``` sh
export TVM_NUM_THREADS=4
export DEVICE='i7_cpu'

export OUTPUT_FILE=/tmp/${DEVICE}/tvm_${TVM_NUM_THREADS}thr_DLIS_${TAG}.csv
python tvm_bench.py --model_dir  $MODEL_DIR \
    --output_file $OUTPUT_FILE \
    --threads ${TVM_NUM_THREADS} \
    --device_name $DEVICE \
    --opt_level 3 
```
