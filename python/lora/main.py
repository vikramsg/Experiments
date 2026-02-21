"""Repository entrypoint.

This delegates to the unified experiment runner.
"""

from lora_training.runners.experiment import main

if __name__ == "__main__":
    main()
