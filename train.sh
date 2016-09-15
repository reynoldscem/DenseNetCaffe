set -e

TOOLS=/home/reynoldscem/caffe/build/tools

$TOOLS/caffe train \
  --solver=solver.prototxt
