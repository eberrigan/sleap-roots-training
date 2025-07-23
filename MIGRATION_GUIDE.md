# Configuration System Migration Guide

This guide helps you migrate from the old YAML-based configuration system to the new OmegaConf/Hydra-based dynamic configuration system.

## 🚀 Why Migrate?

The new configuration system provides:

- **Dynamic Updates**: Configuration changes are immediately reflected across all modules (no kernel restarts!)
- **Type Safety**: Structured configs with dataclasses and validation
- **Runtime Validation**: Catch configuration errors early
- **Event System**: Get notified when configurations change
- **Hydra Integration**: Advanced composition and CLI overrides
- **Backward Compatibility**: Existing code continues to work

## 📊 Old vs New System Comparison

### Old System Problems ❌

```python
# OLD WAY - Static, kernel restart required
from sleap_roots_training.config import CONFIG, update_config

# These were constants loaded at import time
print(CONFIG["project_name"])  # "sleap-roots"

# Update required kernel restart to take effect
update_config(project_name="new-project")
print(CONFIG["project_name"])  # Still "sleap-roots" - no change!
```

### New System Benefits ✅

```python
# NEW WAY - Dynamic, immediate updates
from sleap_roots_training.config_new import get_config, update_config, CONFIG

# Get live configuration
config = get_config()
print(config.wandb.project_name)  # "sleap-roots"

# Update configuration dynamically
update_config(**{"wandb.project_name": "new-project"})
print(config.wandb.project_name)  # "new-project" - immediate change!

# Legacy compatibility still works
print(CONFIG["project_name"])  # "new-project" - also updated!
```

## 🔄 Migration Steps

### Step 1: Install Dependencies

```bash
pip install hydra-core omegaconf
```

### Step 2: Automatic Migration

```bash
# Run the migration script
python -m sleap_roots_training.migrate_config --backup --verify --examples
```

### Step 3: Update Your Code (Optional)

While legacy compatibility means your existing code will continue to work, you can optionally update to use the modern API:

#### Before (Old System):
```python
from sleap_roots_training.config import CONFIG

# Access config values
project_name = CONFIG["project_name"]
entity_name = CONFIG["entity_name"]

# Update config (required kernel restart)
from sleap_roots_training.config import update_config
update_config(project_name="new-project")
```

#### After (New System):
```python
# Option 1: Modern API (recommended)
from sleap_roots_training.config_new import get_config, update_config, get_value

config = get_config()
project_name = config.wandb.project_name
entity_name = config.wandb.entity_name

# Dynamic updates (immediate effect)
update_config(**{"wandb.project_name": "new-project"})

# Option 2: Legacy compatibility (no changes needed)
from sleap_roots_training.config_new import CONFIG
project_name = CONFIG["project_name"]  # Still works!
CONFIG["project_name"] = "new-project"  # Dynamic updates!
```

## 🎯 Key Features

### 1. Dynamic Configuration Updates

```python
from sleap_roots_training.config_new import update_config, get_value

# Update any configuration value dynamically
update_config(**{
    "wandb.project_name": "my-new-project",
    "wandb.experiment_name": "experiment-001",
    "custom.my_setting": "custom_value"
})

# Changes are immediately available everywhere
print(get_value("wandb.project_name"))  # "my-new-project"
```

### 2. Type-Safe Configuration

```python
from sleap_roots_training.config_new import get_config

config = get_config()

# Structured access with IDE autocomplete
project = config.wandb.project_name
entity = config.wandb.entity_name
experiment = config.wandb.experiment_name

# Custom configurations
config.custom.my_setting = "value"
```

### 3. Configuration Change Notifications

```python
from sleap_roots_training.config_new import register_config_callback

def on_config_change(config):
    print(f"Configuration changed! New project: {config.wandb.project_name}")

# Register to get notified of all config changes
register_config_callback(on_config_change)

# Any config update will trigger the callback
update_config(**{"wandb.project_name": "triggers-callback"})
```

### 4. Validation and Error Handling

```python
from sleap_roots_training.config_new import update_config, ConfigurationError

try:
    # This will fail validation
    update_config(**{"wandb.project_name": ""})  # Empty project name
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### 5. File-Based Configuration

```python
from sleap_roots_training.config_new import save_config, load_config_from_file

# Save current configuration
save_config("my_config.yaml")

# Load configuration from file
config = load_config_from_file("my_config.yaml")
```

### 6. Hydra Integration

```python
# Advanced: Use Hydra overrides from command line
# python my_script.py wandb.project_name=cli-project custom.debug=true
```

## 🔧 Configuration Schema

The new system uses structured configuration:

```yaml
wandb:
  project_name: "sleap-roots"
  entity_name: "eberrigan-salk-institute-for-biological-studies"
  experiment_name: null
  registry: null
  collection_name: null
  job_type: null

paths:
  config_dir: "${oc.env:HOME}/.sleap_roots_training"
  data_dir: null
  models_dir: null
  logs_dir: null

validation:
  strict_mode: true
  validate_paths: true
  validate_wandb: true
  allow_missing_optional: true

custom: {}  # For your custom settings
version: "2.0.0"
```

## 🛠️ Troubleshooting

### Import Error
```bash
# If you get import errors, install dependencies:
pip install hydra-core omegaconf
```

### Validation Errors
```python
# Disable strict validation for testing
from sleap_roots_training.config_new import update_config
update_config(**{"validation.strict_mode": False})
```

### Legacy Code Not Working
```python
# Use legacy compatibility layer
from sleap_roots_training.config_new import CONFIG
# All old CONFIG usage should work unchanged
```

## 📝 Complete Migration Example

Here's a complete example showing how to migrate a typical usage:

### Old Code:
```python
# old_script.py
from sleap_roots_training.config import CONFIG
import sleap_roots_training.train as train

def run_training():
    project_name = CONFIG["project_name"]
    entity_name = CONFIG["entity_name"]
    
    print(f"Training project: {project_name}")
    # ... training code ...

if __name__ == "__main__":
    run_training()
```

### New Code (Option 1 - Minimal Changes):
```python
# new_script.py - minimal changes with legacy compatibility
from sleap_roots_training.config_new import CONFIG  # Just change import!
import sleap_roots_training.train as train

def run_training():
    project_name = CONFIG["project_name"]  # Same as before!
    entity_name = CONFIG["entity_name"]    # Same as before!
    
    print(f"Training project: {project_name}")
    # ... training code ...

if __name__ == "__main__":
    run_training()
```

### New Code (Option 2 - Modern API):
```python
# new_script_modern.py - using modern API
from sleap_roots_training.config_new import get_config, update_config
import sleap_roots_training.train as train

def run_training():
    config = get_config()
    project_name = config.wandb.project_name
    entity_name = config.wandb.entity_name
    
    print(f"Training project: {project_name}")
    
    # Dynamic configuration during runtime
    if some_condition:
        update_config(**{"wandb.experiment_name": "dynamic-experiment"})
    
    # ... training code ...

if __name__ == "__main__":
    run_training()
```

## 🎉 Benefits Summary

After migration, you get:

1. **No More Kernel Restarts**: Configuration changes take effect immediately
2. **Type Safety**: IDE autocomplete and error checking
3. **Validation**: Catch configuration errors early  
4. **Flexibility**: Advanced configuration composition with Hydra
5. **Backward Compatibility**: Existing code continues to work
6. **Event System**: React to configuration changes
7. **File Operations**: Save/load configurations easily

## 📞 Need Help?

- Run the migration script: `python -m sleap_roots_training.migrate_config --help`
- Check the comprehensive tests in `tests/test_config_new.py` for examples
- The new system is fully backward compatible - your existing code will work!

Happy migrating! 🚀