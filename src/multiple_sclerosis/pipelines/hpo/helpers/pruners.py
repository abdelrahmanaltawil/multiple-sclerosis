# env imports
import optuna


def get_pruner(pruner: str) -> optuna.pruners.BasePruner:
    '''
    Placeholder
    '''

    optuna_pruners = {
        "SuccessiveHalvingPruner"   : optuna.pruners.SuccessiveHalvingPruner(),
        "HyperbandPruner"           : optuna.pruners.HyperbandPruner(), 
        "MedianPruner"              : optuna.pruners.MedianPruner(),
        "NopPruner"                 : optuna.pruners.NopPruner()
        }

    if pruner not in optuna_pruners.keys(): raise NotImplemented("pruner not implemented")

    return optuna_pruners[pruner]