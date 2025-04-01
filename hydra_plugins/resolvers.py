from omegaconf import OmegaConf


#This allows to do evaluation in the config file (for example for arithmetic on the hidden sizes.)
OmegaConf.register_new_resolver("eval", eval)