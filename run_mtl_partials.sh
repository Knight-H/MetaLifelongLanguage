
# Note: removed early stopping since this is going to take a lot of time!
# Order 1 Partials
# python3.8 train_text_cls_ori_v2.py --order 1 --partials 1 --learner multi_task --model roberta --log_freq 2000
# mkdir /data/model_runs/original_oml/aMUL-order1-part1
# mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order1-part1
# mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order1-part1

# python3.8 train_text_cls_ori_v2.py --order 1 --partials 2 --learner multi_task --model roberta --log_freq 2000
# mkdir /data/model_runs/original_oml/aMUL-order1-part2
# mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order1-part2
# mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order1-part2


# python3.8 train_text_cls_ori_v2.py --order 1 --partials 3 --learner multi_task --model roberta --log_freq 2000
# mkdir /data/model_runs/original_oml/aMUL-order1-part3
# mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order1-part3
# mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order1-part3

# python3.8 train_text_cls_ori_v2.py --order 1 --partials 4 --learner multi_task --model roberta --log_freq 2000
# mkdir /data/model_runs/original_oml/aMUL-order1-part4
# mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order1-part4
# mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order1-part4


# Order 2 Partials
python3.8 train_text_cls_ori_v2.py --order 2 --partials 1 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order2-part1
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order2-part1
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order2-part1

python3.8 train_text_cls_ori_v2.py --order 2 --partials 2 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order2-part2
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order2-part2
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order2-part2

python3.8 train_text_cls_ori_v2.py --order 2 --partials 3 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order2-part3
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order2-part3
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order2-part3

python3.8 train_text_cls_ori_v2.py --order 2 --partials 4 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order2-part4
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order2-part4
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order2-part4


# Order 3 Partials
python3.8 train_text_cls_ori_v2.py --order 3 --partials 1 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order3-part1
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order3-part1
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order3-part1

python3.8 train_text_cls_ori_v2.py --order 3 --partials 2 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order3-part2
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order3-part2
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order3-part2

python3.8 train_text_cls_ori_v2.py --order 3 --partials 3 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order3-part3
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order3-part3
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order3-part3

python3.8 train_text_cls_ori_v2.py --order 3 --partials 4 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order3-part4
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order3-part4
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order3-part4


# Order 4 Partials
python3.8 train_text_cls_ori_v2.py --order 4 --partials 1 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order4-part1
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order4-part1
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order4-part1

python3.8 train_text_cls_ori_v2.py --order 4 --partials 2 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order4-part2
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order4-part2
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order4-part2

python3.8 train_text_cls_ori_v2.py --order 4 --partials 3 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order4-part3
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order4-part3
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order4-part3

python3.8 train_text_cls_ori_v2.py --order 4 --partials 4 --learner multi_task --model roberta --log_freq 2000
mkdir /data/model_runs/original_oml/aMUL-order4-part4
mv /data/model_runs/original_oml/checkpoint* /data/model_runs/original_oml/aMUL-order4-part4
mv /data/model_runs/original_oml/MUL* /data/model_runs/original_oml/aMUL-order4-part4