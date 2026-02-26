from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
# from .arkitscenes import ARKitScenes_Multi  # noqa
# from .arkitscenes_highres import ARKitScenesHighRes_Multi
# from .bedlam import BEDLAM_Multi
from .blendedmvs_claude import BlendedMVS_Multi  # noqa
# from .co3d import Co3d_Multi  # noqa
# from .cop3d import Cop3D_Multi
# from .dl3dv import DL3DV_Multi
# from .dynamic_replica import DynamicReplica
# from .eden import EDEN_Multi
from .hypersim import HyperSim_Multi
from .infinigen import Infinigen_Multi
from .seven_scene import SevenScenes_Multi
# from .hoi4d import HOI4D_Multi
# from .irs import IRS
# from .mapfree import MapFree_Multi
# from .megadepth import MegaDepth_Multi  # noqa
# from .mp3d import MP3D_Multi
# from .mvimgnet import MVImgNet_Multi
# from .mvs_synth import MVS_Synth_Multi
# from .omniobject3d import OmniObject3D_Multi
# from .pointodyssey import PointOdyssey_Multi
# from .realestate10k import RE10K_Multi
# from .scannet import ScanNet_Multi
from .scannetpp2 import ScanNetpp_Multi  # noqa
# from .smartportraits import SmartPortraits_Multi
# from .spring import Spring
# from .synscapes import SynScapes
# from .tartanair import TartanAir_Multi
# from .threedkb import ThreeDKenBurns
# from .uasol import UASOL_Multi
# from .urbansyn import UrbanSyn
# from .unreal4k import UnReal4K_Multi
from .vkitti2 import VirtualKITTI2_Multi  # noqa
# from .waymo import Waymo_Multi  # noqa
# from .wildrgbd2 import WildRGBD_Multi  # noqa
from .dtu import DTU_Multi 
from .eth3d import ETH3D_Multi
# from .base.collation import stack_multiview_batch


num_processes=1

def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    persistent_workers=False,
    prefetch_factor=None,
    fixed_length=False,
    seed=None,
    accum_steps=1,
    debug_enumerate_batches=False,
):
    import torch

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=num_processes,
            fixed_length=fixed_length,
            seed=seed,
            accum_steps=accum_steps,
            debug_enumerate_batches=debug_enumerate_batches,
        )
        dataset.seed = seed
        shuffle = False

        if num_workers == 0:
            persistent_workers = False
            prefetch_factor = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_mem,
            prefetch_factor=prefetch_factor,
        )

    except (AttributeError, NotImplementedError):
        sampler = None

        if num_workers == 0:
            persistent_workers = False
            prefetch_factor = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    return data_loader
