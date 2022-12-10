_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1300.py'
]
# potsdam example:
model = dict(
    decode_head=dict(num_classes=7), auxiliary_head=dict(num_classes=7))