#!/usr/bin/env python3
"""Run the gated Smol vision-transfer confirmatory sweep in Colab."""

from __future__ import annotations

import sys

from run_lfm_transfer_pilot_colab import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--pilot", "smol-confirmatory"]
    main()
