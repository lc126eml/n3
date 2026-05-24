# methods: pred_center (gt_to_pr),
same world coordinate
appropriate normalization: 
gt_pts_invariant
    pr_pts_invariant.translate: false, gt_pts_invariant.translate: false -> pred_center (gt_to_pr)
gt_pts
    getting pred_center, the coordinate is depend on normalization. gt_pts is fine after aligning gt to pr. 
    gt_pts_invariant (no translate), pr_pts_invariant (no translate) -> pred_center (gt_to_pr) -> gt_pts, pr_pts

# center_world
same world coordinate
appropriate normalization: 
gt_pts_invariant
    gt_pts_invariant (no translate) -> pr_pts_invariant (no translate)
gt_pts
    gt_pts -> pr_pts

# gt_align_to_pts
same world coordinate
appropriate normalization: 
gt_pts_invariant
    gt_pts_invariant (pr_pts_invariant? ) translate -> align, no scale 
    pr_pts_invariant (translate?) -> align, scale 

gt_pts
    pr_pts (scale loss to enforce scale approach 1) -> align, scale 

# global register
besides more registers for each frame, global registers for the whole scene
decoder variants:
1. global registers along with all frame pixels;
2. global registers increase, while frame patch tokens decrease (in ratio) with decoder layers (how to merge patch tokens, and how to add more global registers);
prediction variants:
1. still predict with all patch tokens;
2. predict only with global tokens and frame camera tokens;
3. predict with global tokens and camera tokens of novel frames.

# dpt ablation
1. dense head
2. token for each pixel (but dino encoder is on patch)
2.1 only predict downsampled depth map

# active pos (ASG)