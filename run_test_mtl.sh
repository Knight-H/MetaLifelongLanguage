# python3.8 test_text_cls_ori_v2.py --order 1 --learner multi_task --model roberta --model_path /data/model_runs/original_oml/aMUL/checkpoint-1.pt

# 2023-04-25 05:25:11,287 - ContinualLearningLog - INFO - ----------Testing on test set starts here----------
# 2023-04-25 05:25:11,287 - Baseline-Log - INFO - Testing on YelpDataset
#  23%|██████████████▌                                                | 110/475 [00:19<01:04,  5.64it/s]
# 2023-04-25 05:26:32,218 - Baseline-Log - INFO - Test metrics: Loss = 0.7591, accuracy = 0.6657, precision = 0.6674, recall = 0.6665, F1 score = 0.6660
# 2023-04-25 05:26:32,226 - Baseline-Log - INFO - Testing on AGNewsDataset
#  11%|███████▎                                                        | 54/475 [00:08<01:05,  6.42it/s]
# 2023-04-25 05:27:45,952 - Baseline-Log - INFO - Test metrics: Loss = 0.2430, accuracy = 0.9189, precision = 0.9221, recall = 0.9189, F1 score = 0.9201
# 2023-04-25 05:27:45,959 - Baseline-Log - INFO - Testing on DBPediaDataset
# 100%|███████████████████████████████████████████████████████████████| 475/475 [01:15<00:00,  6.28it/s]2023-04-25 05:29:01,590 - Baseline-Log - INFO - Test metrics: Loss = 0.0641, accuracy = 0.9828, precision = 0.9833, recall = 0.9828, F1 score = 0.9829
# 2023-04-25 05:29:01,597 - Baseline-Log - INFO - Testing on AmazonDataset
# 100%|███████████████████████████████████████████████████████████████| 475/475 [01:16<00:00,  6.25it/s]2023-04-25 05:30:17,617 - Baseline-Log - INFO - Test metrics: Loss = 0.8521, accuracy = 0.6342, precision = 0.6336, recall = 0.6345, F1 score = 0.6314
# 2023-04-25 05:30:17,625 - Baseline-Log - INFO - Testing on YahooAnswersDataset
#  75%|███████████████████████████████████████████████                | 355/475 [00:59<00:19,  6.11it/s]
# 2023-04-25 05:31:36,833 - Baseline-Log - INFO - Done writing CSV File at /data/model_runs/original_oml/aMUL/checkpoint-1_results.csv
# 2023-04-25 05:31:36,833 - Baseline-Log - INFO - COPY PASTA - not really but ok
# 2023-04-25 05:31:36,833 - Baseline-Log - INFO - 0.6656578947368421
# 2023-04-25 05:31:36,833 - Baseline-Log - INFO - 0.9189473684210526
# 2023-04-25 05:31:36,833 - Baseline-Log - INFO - 0.9827631578947369
# 2023-04-25 05:31:36,833 - Baseline-Log - INFO - 0.6342105263157894
# 2023-04-25 05:31:36,833 - Baseline-Log - INFO - 0.7578947368421053
# 2023-04-25 05:31:36,833 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.7919, precision = 0.7944, recall = 0.7916, F1 score = 0.7908
# 2023-04-25 05:31:36,833 - Baseline-Log - INFO - Total Time used: 6.0 minutes


python3.8 test_text_cls_ori_v2.py --order 1 --learner multi_task --model roberta --model_path /data/model_runs/original_oml/aMUL/checkpoint-2.pt

# 2023-04-25 05:40:14,195 - Baseline-Log - INFO - Done writing CSV File at /data/model_runs/original_oml/aMUL/checkpoint-2_results.csv
# 2023-04-25 05:40:14,195 - Baseline-Log - INFO - COPY PASTA - not really but ok
# 2023-04-25 05:40:14,195 - Baseline-Log - INFO - 0.6477631578947368
# 2023-04-25 05:40:14,195 - Baseline-Log - INFO - 0.9355263157894737
# 2023-04-25 05:40:14,195 - Baseline-Log - INFO - 0.9853947368421052
# 2023-04-25 05:40:14,195 - Baseline-Log - INFO - 0.6028947368421053
# 2023-04-25 05:40:14,195 - Baseline-Log - INFO - 0.7714473684210527
# 2023-04-25 05:40:14,195 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.7886, precision = 0.7919, recall = 0.7892, F1 score = 0.7871
# 2023-04-25 05:40:14,195 - Baseline-Log - INFO - Total Time used: 6.0 minutes