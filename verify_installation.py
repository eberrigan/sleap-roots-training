#!/usr/bin/env python3
"""
Installation verification script for sleap-roots-training.

This script verifies that SLEAP and sleap-roots-training are properly installed
and can be imported successfully.
"""

import sys
import platform

def check_python_version():
    """Check if Python version is compatible."""
    print(f"Python version: {sys.version}")
    major, minor = sys.version_info[:2]
    if major == 3 and 7 <= minor <= 9:
        print("[OK] Python version compatible")
        return True
    else:
        print("[ERROR] Python version not compatible (requires Python 3.7-3.9 for SLEAP)")
        return False

def check_sleap_installation():
    """Check if SLEAP is properly installed."""
    try:
        import sleap
        print(f"[OK] SLEAP imported successfully (version: {sleap.__version__})")
        return True
    except ImportError as e:
        print(f"[ERROR] Failed to import SLEAP: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] SLEAP import error: {e}")
        return False

def check_package_installation():
    """Check if sleap-roots-training is properly installed."""
    try:
        import sleap_roots_training
        print(f"[OK] sleap-roots-training imported successfully (version: {sleap_roots_training.__version__})")
        
        # Check all modules
        modules = ['config', 'train', 'evaluate', 'models', 'datasets']
        for module in modules:
            try:
                mod = getattr(sleap_roots_training, module)
                print(f"[OK] {module} module accessible")
            except AttributeError:
                print(f"[ERROR] {module} module not accessible")
                return False
        
        return True
    except ImportError as e:
        print(f"[ERROR] Failed to import sleap-roots-training: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] sleap-roots-training import error: {e}")
        return False

def check_dependencies():
    """Check if key dependencies are available."""
    dependencies = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'wandb', 'yaml', 'jupyterlab'
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"[OK] {dep} available")
        except ImportError:
            print(f"[ERROR] {dep} missing")
            missing.append(dep)
    
    return len(missing) == 0

def check_config_functionality():
    """Check if configuration system works."""
    try:
        from sleap_roots_training.config import load_config, DEFAULT_CONFIG
        config = load_config()
        print(f"[OK] Configuration loaded successfully")
        print(f"   Project: {config.get('project_name', 'Not set')}")
        print(f"   Entity: {config.get('entity_name', 'Not set')}")
        return True
    except Exception as e:
        print(f"[ERROR] Configuration system error: {e}")
        return False

def main():
    """Run all installation checks."""
    print("=" * 60)
    print("SLEAP Roots Training Installation Verification")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("SLEAP Installation", check_sleap_installation),
        ("Package Installation", check_package_installation),
        ("Dependencies", check_dependencies),
        ("Configuration System", check_config_functionality)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 20)
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Unexpected error in {name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for i, (name, _) in enumerate(checks):
        status = "[PASS]" if results[i] else "[FAIL]"
        print(f"{name}: {status}")
    
    all_passed = all(results)
    if all_passed:
        print("\n[SUCCESS] All checks passed! Installation is working correctly.")
        print("\nYou can now:")
        print("- Run tests: make test")
        print("- Start development: jupyter lab")
        print("- Check formatting: make lint")
    else:
        print("\n[WARNING] Some checks failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure SLEAP is properly installed for your platform")
        print("2. Run: pip install -e .[dev]")
        print("3. Check the README.md for platform-specific instructions")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())