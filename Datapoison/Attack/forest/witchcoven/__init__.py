"""Interface for poison recipes."""
from .witch_matching import WitchGradientMatching, WitchGradientMatchingNoisy, WitchGradientMatchingHidden, WitchMatchingMultiTarget
from .witch_metapoison import WitchMetaPoison, WitchMetaPoisonHigher, WitchMetaPoison_v3
from .witch_watermark import WitchWatermark
from .witch_poison_frogs import WitchFrogs
from .witch_bullseye import WitchBullsEye
from .witch_patch import WitchPatch
from .witch_htbd import WitchHTBD
from .witch_convex_polytope import WitchConvexPolytope

import torch


def Witch(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args['algorithm'] == 'gradient-matching':
        return WitchGradientMatching(args, setup)
    elif args['algorithm'] == 'gradient-matching-private':
        return WitchGradientMatchingNoisy(args, setup)
    elif args['algorithm'] == 'gradient-matching-hidden':
        return WitchGradientMatchingHidden(args, setup)
    elif args['algorithm'] == 'gradient-matching-mt':
        return WitchMatchingMultiTarget(args, setup)
    elif args['algorithm'] == 'watermark':
        return WitchWatermark(args, setup)
    elif args['algorithm'] == 'patch':
        return WitchPatch(args, setup)
    elif args['algorithm'] == 'hidden-trigger':
        return WitchHTBD(args, setup)
    elif args['algorithm'] == 'metapoison':
        return WitchMetaPoison(args, setup)
    elif args['algorithm'] == 'metapoison-v2':
        return WitchMetaPoisonHigher(args, setup)
    elif args['algorithm'] == 'metapoison-v3':
        return WitchMetaPoison_v3(args, setup)
    elif args['algorithm'] == 'poison-frogs':
        return WitchFrogs(args, setup)
    elif args['algorithm'] == 'bullseye':
        return WitchBullsEye(args, setup)
    elif args['algorithm'] == 'convex-polytope':
        return WitchConvexPolytope(args, setup)
    else:
        raise NotImplementedError()


__all__ = ['Witch']
