## tianxianglii/first-cam-r9-42
44	0	0.387200296
77	0	0.450815707


## hysz0821/first-cam3-r9-42
77	0	0.422506332

## sinayliu/first-cam-vggt-ep25-r9-42
84	0	0.432174832

## qqmail4092/first-cam-vggt-ep30-r9-42
81	0	0.427701205

## qcx2333/first-cam-vggt-ep15-r9-42
80	0	0.42833212

## qcx2333/first-cam-vggt-ep20-r9-42
82	0	0.435785085

# lr of heads 1e-2
## denghaimeng/first-cam-vggt-ep20-r9-42
78	0	0.300269991

## smartchaochao/first-cam-vggt-ep20-1ratio-r9-42
60	0	0.281474978
80	0	0.298145741

## jufuchao/first-cam-vggt-ep20-r9-42
79	0	0.302574426
80	0	0.310555786

## yccwff/first-cam-vggt-cam5-r9-42
lr_multipliers:
  - name: patch_embed
    ratio: 0.01
    param_names:
    - aggregator.patch_embed.*
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
      norm_type: 2
    - module_name:
      - shared_dpt
      max_norm: 5.0
      norm_type: 2
    - module_name:
      - point_head
      max_norm: 5.0
      norm_type: 2
    - module_name:
      - depth
      max_norm: 5.0
      norm_type: 2
    - module_name:
      - camera
      max_norm: 5.0
      norm_type: 2
71	0	0.279932439
81	0	0.280257136
82	0	0.29280445

## straghtwizard/first-cam-vggt-head1-r9-42
lr_multipliers:
  - name: patch_embed
    ratio: 0.01
    param_names:
    - aggregator.patch_embed.*
  - name: camera_head
    ratio: 0.01
    param_names:
    - camera_head.*
  - name: point_head
    ratio: 0.01
    param_names:
    - point_head.*
gradient_clip:
    _target_: train_utils.gradient_clip.GradientClipper
    configs:
    - module_name:
      - aggregator
      max_norm: 1.0
      norm_type: 2
    - module_name:
      - shared_dpt
      max_norm: 1.0
      norm_type: 2
    - module_name:
      - point_head
      max_norm: 1.0
      norm_type: 2
    - module_name:
      - depth
      max_norm: 1.0
      norm_type: 2
    - module_name:
      - camera
      max_norm: 1.0
      norm_type: 2    
81	0	0.294829458
82	0	0.295264989

## xuwenhui123/first-cam-vggt-agg5-r9-42
- module_name:
      - aggregator
      max_norm: 5.0
      norm_type: 2
    - module_name:
      - shared_dpt
      max_norm: 5.0
      norm_type: 2
    - module_name:
      - point_head
      max_norm: 5.0
      norm_type: 2
    - module_name:
      - depth
      max_norm: 5.0
      norm_type: 2
    - module_name:
      - camera
      max_norm: 1.0
      norm_type: 2
81	0	0.292560399
82	0	0.301978409

## yccwff/first-cam-vggt-all5-r9-42
- module_name:
      - aggregator
      max_norm: 5.0
      norm_type: 2
    - module_name:
      - shared_dpt
      max_norm: 5.0
      norm_type: 2
    - module_name:
      - point_head
      max_norm: 5.0
      norm_type: 2
    - module_name:
      - depth
      max_norm: 5.0
      norm_type: 2
    - module_name:
      - camera
      max_norm: 5.0
      norm_type: 2
81	0	0.279110044
82	0	0.292058349

## fanhuayang/first-cam-vggt-all5-r9-42
81	0	0.295772582

## yuanhahah/first-cam-vggt-r9-42
    - module_name:
      - aggregator
      max_norm: 1.0
      norm_type: 2
81	0	0.300092578


# vit large
## djiangjiang/first-cam-vggt-l12-2-r9-42
warmup_epochs: 1
embed_dim: 1024
  depth: 12
  num_heads: 16
  mlp_ratio: 4.0
  num_register_tokens: 4
  qkv_bias: true
  proj_bias: true
  ffn_bias: true
  patch_embed: /kaggle/input/datasets/sinayliu/dino-ds/vitl16
  patch_size: 16
  dpt_frames_chunk_size: 8
126	0	0.245284051

## houwen/first-cam-vggt-l12-r9-42
warmup_epochs: 20
98	0	0.252698183
122	0	0.255981892


# warmup epoch
## yccwff/first-cam-vggt-wep20-r9-32
128	0	0.29477489

## yccwff/first-cam-vggt-wep15-r9-32
129	0	0.288164169

## qqmail4092/first-cam-vggt-wep25-r9-32
129	0	0.300576091

## qqmail4092/first-cam-vggt-wep30-r9-32
114	0	0.269567698
128	0	0.271552593

