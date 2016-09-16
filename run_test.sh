CAFFE_BIN=~/caffe/build/tools/caffe.bin

WEIGHTS=$(ls -t snapshots/*.caffemodel | head -n1)

echo $WEIGHTS

$CAFFE_BIN test \
  -model=test_densenet.prototxt \
  -weights=$WEIGHTS \
  -iterations=157 \
  -gpu=0
