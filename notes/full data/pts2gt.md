## zhuangminghui/pts-align-to-gt-normpr-r6-42
zhuangminghui/pts-align-to-gt-normpr-r5-42
      distributed_sampler_mode: drop_last
51 0.2490323781967163
### zhaotianchi/pts-align-to-gt-normpr-r10-42
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
54	0	0.245160207
60	0	0.243644848
83	0	0.273322701


## rrrrmm/pts-align-to-gt-1st-cam-r10-42
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
2026-04-23 13:26:37
''aug_crop'': 2, ''random_crop_prob'': 0.0, ''random_crop_prob_schedule'':
        None, ''prot'': 0.0, ''pcrop'': 0.95, ''scales'': [0.8, 1.2], ''aspects'':
        [1.0, 1.5],

49	0	0.218852609
83	0	0.276424795

## xpeng01/pts-align-to-gt-normpr-r9-42
66	0	0.28563112

## linyunlk243/pts-align-to-gt-1st-cam-r9-42
2026-05-11 12:09:14
''aug_crop'': 2, ''random_crop_prob'': 0.0,
        ''random_crop_prob_schedule'': None, ''prot'': 0.0, ''pcrop'': 0.95, ''scales'':
        [0.8, 1.2], ''aspects'': [0.33, 1.0],
test: cojitter_ratio: 0.5
    - module_name:
      - point_head
      max_norm: 5.0

77	0	0.258244812

# lr of heads 1e-2
## huoqiuxia/pts-align-to-gt-1st-cam-r9-42
2026-05-11 12:12:34
''aug_crop'': 9, ''random_crop_prob'': 0.0,
        ''random_crop_prob_schedule'': None, ''prot'': 0.0, ''pcrop'': 0.0, ''scales'':
        [0.8, 1.2], ''aspects'': [1.0, 1.5],
test: cojitter_ratio: 0.3
      - point_head
      max_norm: 1.0
74	0	0.278005302

## top1pcl/pts-align-to-gt-normpr-r9-42
74	0	0.272124112
78	0	0.272558928

## asdsad0000/pts-align-to-gt-1st-cam-1ratio-r9-42  warmup_epochs: 15
    - module_name:
      - camera
      max_norm: 1.0
55	0	0.271872789
73	0	0.270068616
74	0	0.281850606

## hsdfuieqg/pts-align-to-gt-1st-cam-r9-42  
**camera_head 0.1 seems good**
warmup_epochs: 15
  - name: camera_head
    ratio: 0.1
    param_names:
    - camera_head.*
  - name: point_head
    ratio: 0.1
    param_names:
    - point_head.*

        - module_name:
      - aggregator
      max_norm: 1.0
        - module_name:
      - camera
      max_norm: 1.0
74	0	0.254975528

## xx03071425/pts-align-to-gt-normpr-r9-42
77	0	0.265590996
78	0	0.271738768

## huoqiuxia/pts-align-to-gt-normpr-r9-42
60	0	0.266103208
77	0	0.279282093

## xuwenhui123/pts-align-to-gt-1st-cam-1ratio-r9-42  warmup_epochs: 20
  - name: camera_head
    ratio: 0.01
    param_names:
    - camera_head.*
  - name: point_head
    ratio: 0.01
    param_names:
    - point_head.*
        - module_name:
      - aggregator
      max_norm: 1.0
          - camera
      max_norm: 5.0
      norm_type: 2
73	0	0.277879149

