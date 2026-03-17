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