cd /home/sweekar/SDN_main/services/SDN
/home/sweekar/SDN_main/caffe/build/tools/caffe train \
--solver="/home/sweekar/SDN_main/models/VGG/300x300/solver.prototxt" \
--snapshot="/home/sweekar/SDN_main/models/VGG/300x300/VGG_300x300_iter_7700.solverstate" \
--gpu 0 2>&1 | tee /home/sweekar/SDN_main/tasks/VGG/300x300/VGG_300x300.log
