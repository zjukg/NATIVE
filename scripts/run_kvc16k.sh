DATA=Kuai16K
EMB_DIM=250
NUM_BATCH=1024
MARGIN=4
LR=2e-5
LRG=2e-5
NEG_NUM=32
MU=0.00001
EPOCH=1000

CUDA_VISIBLE_DEVICES=0 nohup python run_adv_wgan_gp_4modal.py -dataset=$DATA \
  -batch_size=$NUM_BATCH \
  -margin=$MARGIN \
  -epoch=$EPOCH \
  -dim=$EMB_DIM \
  -save=./checkpoint/$DATA-$NUM_BATCH-$EMB_DIM-$NEG_NUM-$MU-$MARGIN-$LR-$LRG-$EPOCH \
  -neg_num=$NEG_NUM \
  -mu=$MU \
  -learning_rate=$LR\
  -lrg=$LRG > $DATA-$EMB_DIM-$NUM_BATCH-$NEG_NUM-$MU-$MARGIN-$LR-$LRG-$EPOCH.txt &
