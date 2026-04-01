# Analysis of Normalizations
## pts_invariant
scale the points and poses by scale of points
### wt translation
with translation, pts_invariant re-locate world coordinate to point centers.
Both rotation and translation of poses are changed.
Raw camera poses should not be used in loss.
### wo translation
translation scale of camera poses are changed.


# center_world
desc: center of translation and rotation of poses as world coordinate
since it is unique for non-trivial poses, same world coordinate for gt and preds
Procedure: Re-locate GT world coordinate to pose centers; scale the GT pts3d and translation of poses; Predict the pts3d and poses; scale the predicted pts3d and translation of poses; compare the pts3d and poses to calculate loss.
camera (raw) and pts3d can be compared in loss.
appropriate normalization: 
gt_pts_invariant (point center)
    gt_pts_invariant (no translate) <-> pr_pts_invariant (no translate)
pts coordinates normalization (world center) most natural
    gt_pts <-> pr_pts
pts coordinates normalization (first frame) not suitable
    gt_pts -> pr_pts

# pts_align_to_gt
Pred pts3d registered to gt pts3d. same world coordinate after transformation.

Raw camera should not be used in loss. Transformed poses can be compared in loss.
Procedure: Scale the GT pts3d and translation of poses; Predict the pts3d and poses; scale the predicted pts3d and translation of poses (optional); Register pred pts3d to gt pts3d (rotation, translation, scale);
The world coordinate for scale and pts3d coordiante of GT should be certain, e.g., first frame, pose center. point center is determinitic for scale, but not for pts3d coordinates (rotation). But registration of points might be considering both translation and rotation.

What is the world coordinate of Pred pts3d? Arbitrary? independent of pred poses? The first frame or pose center normalization of pred pts3d is not suitable since it is decoupled from pred poses. If not decoupled, then poses are in same world coordinate of pts3d.
appropriate normalization: 
gt_pts_invariant
    gt_pts_invariant (pr_pts_invariant? ) translate -> align, no scale 
    pr_pts_invariant (translate?) -> align, scale 

gt_pts
    pr_pts (scale loss to enforce scale approaching 1) -> align, scale 

# gt_align_to_pred
GT pts3d registered to pred pts3d. same world coordinate after transformation.
Raw camera should not be used in loss. Transformed poses can be compared in loss.
Procedure: Scale the GT pts3d and translation of poses (optional); Predict the pts3d and poses; scale the predicted pts3d and translation of poses; Register GT pts3d to pred pts3d (rotation, translation, scale);
The world coordinate for scale and pts3d coordiante of pred pts3d should be certain, e.g., first frame, pose center. point center is determinitic for scale, but not for pts3d coordinates (rotation).
appropriate normalization: 
gt_pts_invariant
    gt_pts_invariant (pr_pts_invariant? ) translate -> align, no scale 
    pr_pts_invariant (translate?) -> align, scale 

gt_pts
    pr_pts (scale loss to enforce scale approaching 1) -> align, scale 

# methods: pred_center 
GT has a certain world coordinate (first frame, pose center, etc). Pred has a presumed world coordinate by its poses. Since each pose has errors, the center of error (least error center) as the pred world coordinate. Transformation between these two world coordinate contains translation and rotation. But scale normalization has effect of scale of loss. 
## pr to gt
GT normalized scale to first frame or pts3d center. No translation.

## gt to pr
pred normalized scale to first frame or pts3d center. No translation.

    gt_pts_invariant (no translate), pr_pts_invariant (no translate) -> pred_center (gt_to_pr) -> gt_pts, pr_pts?

unfreeze(self.model, True) make metrics worse