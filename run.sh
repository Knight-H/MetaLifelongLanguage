# Training with order 1
source env/bin/activate
python train_text_cls.py --order 1


# python test_text_cls2.py --order 1 --learner oml2 --model gpt2 --max_length 1024 --run_id 20220608T165449_1_oml2gpt2 --test && python train_text_cls2.py --order 1 --learner oml2 --model gpt2 --max_length 1024 --inner_lr 6.25e-5 --meta_lr 6.25e-5 --n_epochs 2