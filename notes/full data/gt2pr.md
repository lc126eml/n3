## qqmail4092/gt-align-to-pts-normpr-lr2-5-r9-42
71	0	0.326134026

# lr of heads 1e-2
## denghaimeng/gt-align-to-pts-normpr-r9-42
82	0	0.298197716

## top1pcl/gt-align-to-pts-1st-cam-r8-42
67	0	0.352448106

## asdsad0000/gt-align-to-pts-normpr-r8-42
  warmup_epochs: 15
  - name: camera_head
    ratio: 0.1
    param_names:
    - camera_head.*
  - name: point_head
    ratio: 0.1
    param_names:
    - point_head.*
val_hypersim_recon_abs_rel': tensor(0.2740, device='cuda:0'), 'val_hypersim_recon_mae': tensor(1.1071, device='cuda:0'), 'val_hypersim_rot_error_mean_deg': tensor(44.0924, device='cuda:0'), 'val_hypersim_trans_error_mean': tensor(3.4920, device='cuda:0'), 'val_hypersim_trans_angle_error_mean_deg': tensor(93.9928, device='cuda:0'), 'Trainer/where': 0.5168195374486523, 'Trainer/epoch': 67

## asssmer/gt-align-to-pts-1st-cam-r9-42
      to_first_cam:
        enabled: true
        points: true
      pr_align_cam:
        enabled: true
        points: true
    normalize:
      gt_pts: true
      gt_depth: false
      gt_pts_invariant:
        enabled: false
        translate: false
      pr_pts:
        enabled: true
        metric: false
      pr_pts_invariant:
        enabled: false
        translate: false
  - name: camera_head
    ratio: 0.1
    param_names:
    - camera_head.*
  - name: point_head
    ratio: 0.1
    param_names:
    - point_head.*
74	0	0.333279133

## xuwenhui123/gt-align-to-pts-normpr-r9-42
      to_first_cam:
        enabled: false
        points: true
      pr_align_cam:
        enabled: false
        points: false
    normalize:
      gt_pts: false
      gt_depth: false
      gt_pts_invariant:
        enabled: true
        translate: true
      pr_pts:
        enabled: false
        metric: false
      pr_pts_invariant:
        enabled: true
        translate: true
  warmup_epochs: 15
    - name: camera_head
    ratio: 0.001
    param_names:
    - camera_head.*
  - name: point_head
    ratio: 0.001
    param_names:
    - point_head.*

    - module_name:
      - camera
      max_norm: 1.0
      norm_type: 2
68	0	0.264515281
81	0	0.289599508
82	0	0.293367118

## roseqw/gt-align-to-pts-normpr-r9-42
  warmup_epochs: 20
    - name: camera_head
    ratio: 0.01
    param_names:
    - camera_head.*
  - name: point_head
    ratio: 0.01
    param_names:
    - point_head.*
71	0	0.291454852
77	0	0.29652673
78	0	0.301002681

## sollasi/gt-align-to-pts-1st-cam-r9-42
  warmup_epochs: 20
    - name: camera_head
    ratio: 0.01
    param_names:
    - camera_head.*
  - name: point_head
    ratio: 0.01
    param_names:
    - point_head.*

    - module_name:
      - camera
      max_norm: 5.0
      norm_type: 2
71	0	0.336665094

## yuanhahah/gt-align-to-pts-1st-cam-point1-r9-42
  warmup_epochs: 20
    - name: camera_head
    ratio: 0.01
    param_names:
    - camera_head.*
  - name: point_head
    ratio: 0.01
    param_names:
    - point_head.*
72	0	0.349721462

## kezzymik/gt-align-to-pts-1st-cam-lr05-r9-42
  warmup_epochs: 20
    - name: camera_head
    ratio: 0.005
    param_names:
    - camera_head.*
  - name: point_head
    ratio: 0.005
    param_names:
    - point_head.*
73	0	0.336959362

## gorgeous0912/gt-align-to-pts-normpr-lr05-r9-42
80	0	0.319999039

## gorgeous0912/gt-align-to-pts-normpr-point1-r9-42
80	0	0.302060157

