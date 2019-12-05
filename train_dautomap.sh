BATCH_SIZE=4
NUM_EPOCHS=100
DEVICE='cuda:0'
CHALLENGE='singlecoil'

MODEL='fastmri_dautomap'
DATASET_TYPE='fastmri'
DATA_PATH='/media/student1/NewVolume/MR_Reconstruction/datasets/fastmri'

SAMPLE_RATE=0.01

<<ACC_FACTOR_2x
ACC_FACTOR=2
ACC='2x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/fastmri/acc_'${ACC}'/'${MODEL}
CENTER_FRACTION=0.04
python train_unet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --data-path ${DATA_PATH} --accelerations ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE} --center-fraction ${CENTER_FRACTION} --challenge ${CHALLENGE}
ACC_FACTOR_2x

#<<ACC_FACTOR_4x
ACC_FACTOR=4
ACC='4x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/fastmri/acc_'${ACC}'/'${MODEL}
CENTER_FRACTION=0.08
python models/dautomap/train_dautomap.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --data-path ${DATA_PATH} --accelerations ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE} --center-fraction ${CENTER_FRACTION} --challenge ${CHALLENGE}
#ACC_FACTOR_4x

<<ACC_FACTOR_8x
ACC_FACTOR=8
ACC='8x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/fastmri/acc_'${ACC}'/'${MODEL}
CENTER_FRACTION=0.04
python train_unet.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --data-path ${DATA_PATH} --accelerations ${ACC_FACTOR} --sample-rate ${SAMPLE_RATE} --center-fraction ${CENTER_FRACTION} --challenge ${CHALLENGE}
ACC_FACTOR_4x
