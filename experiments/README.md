# Learning scripts

The scripts contained here should have a common structure. Most of the functionality is taken care of by the 
[Module](https://github.com/opeltre/revert/revert/models/module.py) class and the [revert.cli](https://github.com/opeltre/revert/revert/cli) 
module. Have a look at [twins.py](twins.py) or [pulse-gen.py](pulse-gen.py) for examples. 

In particular, scripts should accept either as CLI arguments or via a `script.toml` configuration file the following options:
- `--data`: path to specific dataset, relative to `$INFUSION_DATASETS` or `$PCMRI_DATASETS` directory
- `--input`: path to a pretrained model state
- `--output`: path to save the final model state

Using methods from the cli module one can define whatever additional arguments needed along with their default values. 

## Looping over models and parameters

The `@cli.args(**defaults)` decorator will read arguments either from the command-line or configuration file and pass them to the main function. 

When the configuration file has a `[models]` section, the main function will be looped over subsections, using nested argument definitions at every run.
