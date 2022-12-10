_base_ = [
    '../_base_/models/false_segmodel.py', '../_base_/datasets/segmentation_data.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1300.py'
]
# potsdam example:
model = dict(
    decode_head=dict(num_classes=7), auxiliary_head=dict(num_classes=7))