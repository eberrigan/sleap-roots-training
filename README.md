# sleap-roots-training
Jupyter notebooks for training and evaluating models for use in the sleap-roots-pipeline. Weights and Biases logging is integrated.

See (guide)[https://www.notion.so/SLEAP-Roots-Model-Training-Guide-21b4a67a766780ec91c0f92ae459ef2c?source=copy_link]

## Quickstart

```
conda activate <sleap_environment>
```

```
pip install wandb
```

```
(sleap_v1.4.1) C:\repos\sleap-roots-training>wandb login
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit:
wandb: Appending key for api.wandb.ai to your netrc file: C:\Users\Elizabeth\_netrc
```

Make sure you run notebook from the root of this repo so that `sleap_roots_training` functions are imported.
Save a **copy** of helper notebooks with experiment name modifying helper notebook name. Work on a separate branch so experiments can be merged in a tracked with the code.