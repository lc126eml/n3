import torch
from torch.utils.data.dataloader import default_collate

def stack_multiview_batch(batch):
    """
    Custom collate function to transform a batch of multi-view data.

    The default data loader returns a list of lists of dictionaries:
    `B x [view_1_dict, view_2_dict, ..., view_N_dict]`

    This function transforms it into a single dictionary where each key
    maps to a tensor stacked along a new "views" dimension (S).

    Args:
        batch (list): A list of length B, where each element is a list of N view dictionaries.
                      Example: `[[view_0_scene_0, view_1_scene_0], [view_0_scene_1, view_1_scene_1]]`

    Returns:
        dict: A single dictionary with stacked tensors.
              - 'img': (B, S, 3, H, W)
              - 'camera_pose': (B, S, 4, 4)
              - 'dataset': List[List[str]] of shape (B, S)
              ... and so on for other keys.
    """
    # batch is a list of length B, each element is a list of N views
    # e.g., [[view1_sample1, view2_sample1], [view1_sample2, view2_sample2]]
    
    num_scenes = len(batch)
    if num_scenes == 0:
        return {}
    
    num_views = len(batch[0])
    
    # Transpose the batch from (B, N, dict) to (N, B, dict)
    # e.g., [[view1_sample1, view1_sample2], [view2_sample1, view2_sample2]]
    views_batched_by_type = list(zip(*batch))

    # Collate each view type separately. This gives a list of N dictionaries.
    # e.g., [ {img: (B, 3, H, W), ...}, {img: (B, 3, H, W), ...} ]
    collated_views = [default_collate(view_batch) for view_batch in views_batched_by_type]

    # Stack the tensors from each view's dictionary along a new dimension.
    stacked_batch = {}
    for key in collated_views[0]:
        if isinstance(collated_views[0][key], torch.Tensor):
            stacked_batch[key] = torch.stack([v[key] for v in collated_views], dim=1)
        else: # For non-tensor data like lists of strings
            stacked_batch[key] = [v[key] for v in collated_views]
            # Transpose from (S, B) to (B, S)
            stacked_batch[key] = list(map(list, zip(*stacked_batch[key])))

    return stacked_batch