# env imports
import optuna


def get_sampler(sampler: str) -> optuna.samplers.BaseSampler:
    '''
    Placeholder
    '''

    optuna_samplers = {
        "NSGAIISampler"     : optuna.samplers.NSGAIISampler(),
        "RandomSampler"     : optuna.samplers.RandomSampler(),
        "CmaEsSampler"      : optuna.samplers.CmaEsSampler(),
        "TPESampler"        : optuna.samplers.TPESampler()
        }

    if sampler not in optuna_samplers.keys(): raise NotImplemented("sampler not implemented")

    return optuna_samplers[sampler]