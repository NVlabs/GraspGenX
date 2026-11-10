# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static PTV3 plan shapes shared by the exporter and installer."""

import json
import operator
from pathlib import Path

from graspgenx.fast_inference._constants import artifact_subdirectory

DEFAULT_STAGE_CAPACITIES = (3500, 2048, 1024, 512, 512)
PROFILE_FILENAME = "ptv3_profile.json"


def validate_stage_capacities(values):
    try:
        values = tuple(values)
        if any(isinstance(x, bool) for x in values):
            raise TypeError
        capacities = tuple(operator.index(x) for x in values)
    except TypeError as error:
        raise ValueError("PTV3 capacities must contain five integers") from error
    if len(capacities) != 5:
        raise ValueError("PTV3 capacities must contain exactly five integers")
    for stage, capacity in enumerate(capacities):
        minimum = 100 if stage == 0 else 1
        if not minimum <= capacity <= 3500:
            raise ValueError(f"PTV3 stage {stage} capacity must be in [{minimum},3500]")
    return capacities


def read_stage_capacities(directory):
    path = Path(directory) / PROFILE_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
        return validate_stage_capacities(payload["stage_capacities"])
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Invalid or incomplete PTV3 profile: {path}") from error


def write_stage_capacities(directory, capacities):
    capacities = validate_stage_capacities(capacities)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PROFILE_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"stage_capacities": capacities}, indent=2) + "\n")
    temporary.replace(path)


def profile_artifact_subdirectory(denoiser, capacities):
    capacities = validate_stage_capacities(capacities)
    name = artifact_subdirectory(denoiser)
    name += "-p" + "-".join(map(str, capacities))
    return name
