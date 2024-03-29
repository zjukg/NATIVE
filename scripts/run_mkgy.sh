DATA=MKG-Y
EMB_DIM=250
NUM_BATCH=1024
MARGIN=4
LR=1e-4
LRG=1e-4
NEG_NUM=128
MU=0.0001
EPOCH=1000

CUDA_VISIBLE_DEVICES=0 nohup python run_adv_wgan_gp.py -dataset=$DATA \
  -batch_size=$NUM_BATCH \
  -margin=$MARGIN \
  -epoch=$EPOCH \
  -dim=$EMB_DIM \
  -save=./checkpoint/$DATA-$NUM_BATCH-$EMB_DIM-$NEG_NUM-$MU-$MARGIN-$LR-$LRG-$EPOCH \
  -neg_num=$NEG_NUM \
  -mu=$MU \
  -learning_rate=$LR\
  -lrg=$LRG > $DATA-$EMB_DIM-$NUM_BATCH-$NEG_NUM-$MU-$MARGIN-$LR-$LRG-$EPOCH.txt &
