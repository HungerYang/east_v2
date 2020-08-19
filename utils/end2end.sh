python end2end.py \
--test_data_path=$1 \
--pb_path=$2 \
--gpu_list=$3 \
--output_dir=result \
--reg_model_path=crnn/models/ocr-model.pb \
--dict_path=dicts/char_6100

