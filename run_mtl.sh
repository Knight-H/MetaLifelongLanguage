# Doesn't matter what order this is!
python3.8 train_text_cls_ori_v2.py --order 1 --learner multi_task --model roberta --early_stopping --log_freq 2000

# Batch size 16
# 2023-04-22 09:49:46,712 - ContinualLearningLog - INFO - ----------Training starts here----------
# 2023-04-22 09:49:46,712 - Baseline-Log - INFO - Training multi-task model on all datasets
# 2023-04-22 09:49:46,713 - Baseline-Log - INFO - Starting Epoch 0
#   0%|▎                                                              | 156/35938 [01:15<4:46:54,  2.08it/s]

# Batch size 32
# 2023-04-22 09:52:38,655 - ContinualLearningLog - INFO - ----------Training starts here----------
# 2023-04-22 09:52:38,655 - Baseline-Log - INFO - Training multi-task model on all datasets
# 2023-04-22 09:52:38,657 - Baseline-Log - INFO - Starting Epoch 0
#   0%|                                                                | 26/17969 [00:24<4:34:51,  1.09it/s]



# Actual Run
# 2023-04-23 16:35:13,080 - ContinualLearningLog - INFO - ----------Training starts here----------      
# 2023-04-23 16:35:13,081 - Baseline-Log - INFO - Training multi-task model on all datasets             
# 2023-04-23 16:35:13,082 - Baseline-Log - INFO - Starting Epoch 1                                 
# 6%|███▏                                                      | 1999/35938 [15:36<4:24:21,  2.14it/s]
# 2023-04-23 16:50:49,898 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.8791, accuracy = 0.6970, precision = 0.7841, recall = 0.7713, F1 score = 0.7762                                                  
# 11%|██████▍                                                   | 3999/35938 [31:05<4:08:36,  2.14it/s]
# 2023-04-23 17:06:18,797 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.6484, accuracy = 0.7577, precision = 0.8334, recall = 0.8342, F1 score = 0.8334
# 2023-04-23 20:13:16,638 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.5633, accuracy = 0.7[59/651$cision = 0.8525, recall = 0.8531, F1 score = 0.8521                                                    
# 83%|███████████████████████████████████████████████▌         | 29999/35938 [3:53:37<46:25,  2.13it/s]
# 2023-04-23 20:28:51,083 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.5604, accuracy = 0.7863, precision = 0.8512, recall = 0.8525, F1 score = 0.8513                                                    
# 89%|██████████████████████████████████████████████████▊      | 31999/35938 [4:09:10<30:37,  2.14it/s]
# 2023-04-23 20:44:24,291 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.5623, accuracy = 0.7827, precision = 0.8504, recall = 0.8507, F1 score = 0.8498                                                  
# 95%|█████████████████████████████████████████████████████▉   | 33999/35938 [4:24:43<15:03,  2.15it/s]
# 2023-04-23 20:59:57,081 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.5649, accuracy = 0.7803, precision = 0.8500, recall = 0.8504, F1 score = 0.8496                                                   
# 100%|█████████████████████████████████████████████████████████| 35938/35938 [4:39:47<00:00,  2.14it/s]
# 2023-04-23 21:15:00,728 - Baseline-Log - INFO - Starting Validation...
# 100%|█████████████████████████████████████████████████████████████| 1563/1563 [04:09<00:00,  6.25it/s]
# 2023-04-23 21:19:10,740 - Baseline-Log - INFO - Test metrics: Loss = 0.5322, accuracy = 0.7947, precision = 0.8618, recall = 0.8581, F1 score = 0.8582                                                      
# 2023-04-23 21:19:10,741 - Baseline-Log - INFO - [VAL_LOSS] Epoch 1: 0.5322208586825214                
# 2023-04-23 21:19:10,741 - Baseline-Log - INFO - Saving Model on Epoch: 1                              
# 2023-04-23 21:19:11,869 - Baseline-Log - INFO - Starting Epoch 2
# 6%|███▏                                                      | 1999/35938 [15:05<4:16:30,  2.21it/s]
# 2023-04-23 21:34:17,663 - Baseline-Log - INFO - Epoch 2 metrics: Loss = 0.5048, accuracy = 0.8057, precision = 0.8698, recall = 0.8695, F1 score = 0.8690    
# 2023-04-24 00:49:41,333 - Baseline-Log - INFO - Epoch 2 metrics: Loss = 0.5056, accuracy = 0.8050, precision = 0.8680, recall = 0.8671, F1 score = 0.8666
#  83%|███████████████████████████████████████████████▌         | 29999/35938 [3:45:29<44:46,  2.21it/s]2023-04-24 01:04:42,292 - Baseline-Log - INFO - Epoch 2 metrics: Loss = 0.5023, accuracy = 0.8064, precision = 0.8708, recall = 0.8705, F1 score = 0.8701
#  89%|██████████████████████████████████████████████████▊      | 31999/35938 [4:00:20<28:51,  2.27it/s]2023-04-24 01:19:33,299 - Baseline-Log - INFO - Epoch 2 metrics: Loss = 0.5098, accuracy = 0.8034, precision = 0.8664, recall = 0.8659, F1 score = 0.8653
#  95%|█████████████████████████████████████████████████████▉   | 33999/35938 [4:15:02<14:10,  2.28it/s]2023-04-24 01:34:14,483 - Baseline-Log - INFO - Epoch 2 metrics: Loss = 0.5141, accuracy = 0.8008, precision = 0.8638, recall = 0.8643, F1 score = 0.8633
# 100%|█████████████████████████████████████████████████████████| 35938/35938 [4:29:30<00:00,  2.22it/s]2023-04-24 01:48:42,307 - Baseline-Log - INFO - Starting Validation...
# 100%|█████████████████████████████████████████████████████████████| 1563/1563 [04:06<00:00,  6.35it/s]2023-04-24 01:52:48,549 - Baseline-Log - INFO - Test metrics: Loss = 0.5331, accuracy = 0.7920, precision = 0.8627, recall = 0.8620, F1 score = 0.8607
# 2023-04-24 01:52:48,550 - Baseline-Log - INFO - [VAL_LOSS] Epoch 2: 0.5331437942772741
# 2023-04-24 01:52:48,550 - Baseline-Log - INFO - Saving Model on Epoch: 2
# 2023-04-24 01:52:49,419 - Baseline-Log - INFO - Stopping from Early Stopping... min_loss 0.5322208586825214
# 2023-04-24 01:52:49,421 - Baseline-Log - INFO - Total Training Time used: 557.0 minutes
# 2023-04-24 01:52:49,421 - ContinualLearningLog - INFO - [TIME] End Run at 2023-04-24T01:52:49 within 9.297429750627941 hours