# Training and inference: 
# python efficientad.py --dataset mvtec_ad --subdataset bottle
# python efficientad.py --dataset mvtec_ad --subdataset bottle --model_size medium --weights models/teacher_medium.pth --imagenet_train_path ./ILSVRC/Data/CLS-LOC/train


# Evaluation with Mvtec evaluation code:
# python mvtec_ad_evaluation/evaluate_experiment.py --dataset_base_dir './mvtec_anomaly_detection/' \
#                                                   --anomaly_maps_dir './output/1/anomaly_maps/mvtec_ad/' \
#                                                   --output_dir './output/1/metrics/mvtec_ad/' 
#                                                   --evaluated_objects bottle
# python mvtec_loco_ad_evaluation/evaluate_experiment.py --dataset_base_dir './mvtec_loco_anomaly_detection/' --anomaly_maps_dir './output/1/anomaly_maps/mvtec_loco/' --output_dir './output/1/metrics/mvtec_loco/' --object_name breakfast_box