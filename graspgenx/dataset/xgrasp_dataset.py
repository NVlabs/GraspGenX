#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data loader for training grasp models.
"""

import glob
import json
import logging
import os
import random
import time
from typing import Tuple, Union

import h5py
import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
from omegaconf import DictConfig
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset
from tqdm import tqdm

from copy import deepcopy as copy

from graspgenx.dataset.dataset_utils import (
    GraspGenDatasetCache,
    ObjectGraspDataset,
    dump_object_list,
    load_from_json,
    load_object_grasp_data,
    GraspJsonDatasetReader,
    get_rotation_augmentation,
)
from graspgenx.dataset.xgrasp_dataset_utils import (
    XGraspJsonDatasetReader,
    load_object_xgrasp_data,
    XGraspGenDatasetCache,
    XGraspGenDatasetCacheLazy,
    filter_xgripper_grasps_by_point_cloud_visibility,
)
from graspgenx.dataset.dataset import (
    generate_negative_hardnegatives,
    generate_negative_retract,
    generate_negative_freespace,
    load_discriminator_batch_with_stratified_sampling,
    collate_batch_keys,
    collate,
)
from graspgenx.dataset.eval_utils import check_collision
from graspgenx.dataset.exceptions import DataLoaderError

try:
    from graspgenx.dataset.renderer import render_pc
except (ImportError, AttributeError, OSError):
    render_pc = None
from graspgenx.dataset.visualize_utils import (
    MAPPING_ID2NAME,
    MAPPING_NAME2ID,
    visualize_discriminator_dataset,
    visualize_xgripper_discriminator_dataset,
    visualize_xgripper_generator_dataset,
    visualize_generator_dataset,
)
from graspgenx.x_grippers import resolve_gripper_info, sample_gripper_pointclouds
from graspgenx.utils.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


def get_cache_path(cache_dir: str, cache_name: str) -> str:
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/{cache_name}"
    os.makedirs(cache_path, exist_ok=True)
    return cache_path


def get_cache_prefix(prob_point_cloud: float, load_discriminator_dataset: bool) -> str:
    cache_name = get_pc_setting_name(prob_point_cloud)
    prefix = "dis" if load_discriminator_dataset else "gen"
    cache_name = f"{cache_name}_{prefix}"
    return cache_name


def get_pc_setting_name(prob_point_cloud: float) -> str:
    name = "mesh" if prob_point_cloud <= 0 else "pc"
    if prob_point_cloud < 1.0 and prob_point_cloud > 0:
        name = "meshandpc"
    return name


def load_from_text_file(file_path: str) -> list:
    """
    Load object IDs or gripper IDs to object meshes from a text file.

    Each line in the text file should contain one object ID or relative path.
    Lines are stripped of whitespace and empty lines are ignored.

    Args:
        file_path: Path to the text file containing object IDs/paths

    Returns:
        List of object IDs or relative paths as strings

    Raises:
        FileNotFoundError: If the text file doesn't exist
    """
    all_objects = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                all_objects.append(line)
    return all_objects


def _safe_array_copy(arr):
    """Copy array whether numpy ndarray or torch Tensor (shared-memory compatible).

    When using shared-memory tensors for multi-GPU DDP training, attributes on
    ObjectGraspDataset may be torch Tensors instead of numpy arrays.  This
    helper transparently handles both cases so __getitem__ code stays clean.
    """
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr.numpy().copy()
    return arr.copy()


class XGraspObjectPickDataset(Dataset):
    def __init__(
        self,
        root_dir,
        cache_dir,
        cache_name,
        use_cache,
        obj_split_path,
        gripper_split_path,
        grasp_split_path,
        num_points,
        num_obj_points,
        cam_coord,
        num_rotations,
        grid_res,
        jitter_scale,
        contact_radius,
        dist_above_table,
        robot_prob,
        random_seed,
        min_grasps_gen_th=5,
        render_redundancy=1,
        cache_save_freq=5000,
        inference=False,
        load_onehot_vec=True,
        rotation_augmentation=False,
        downsample_points=True,
        add_depth_noise=False,
        load_patch=False,
        patch_width=200,
        prob_point_cloud=-1.0,
        object_root_dir="",
        grasp_root_dir="",
        num_grasps_per_object=20,
        load_discriminator_dataset=False,
        load_contact=False,
        discriminator_ratio=[0.50, 0.20, 0.25, 0.05, 0.0],
        visualize_batch=False,
        onpolicy_dataset_name=None,
        onpolicy_dataset_json_dir=None,
        onpolicy_dataset_h5_dir=None,
        shared_cache=None,
        single_gripper=None,
        loading_mode="preload",
        alternative_json_file_path=None,
        shard_dir_nfs=None,
        shard_dir_local=None,
        shard_local_rank=0,
        shard_min_ready=3,
    ):

        self.obj_split = load_from_text_file(f"{root_dir}/{obj_split_path}.txt")
        self.gripper_split = load_from_text_file(f"{root_dir}/{gripper_split_path}.txt")

        # Filter to a single gripper for parallel cache generation.
        # When single_gripper is set, only scenes for that gripper are
        # enumerated and a gripper-specific cache file is produced.
        self.single_gripper = single_gripper
        if single_gripper is not None:
            assert (
                single_gripper in self.gripper_split
            ), f"Gripper '{single_gripper}' not found in split '{gripper_split_path}'"
            self.gripper_split = [single_gripper]

        self.obj_split_path = obj_split_path
        self.gripper_split_path = gripper_split_path
        self.grasp_split_path = grasp_split_path

        self.scenes = self._enumerate_scenes(self.obj_split, self.gripper_split)
        self.grippers = self.load_gripper_info(self.gripper_split)

        self._cache_dir = get_cache_path(cache_dir, cache_name)
        cache_token = get_cache_prefix(prob_point_cloud, load_discriminator_dataset)
        self._cache_file = f"{grasp_split_path}_{gripper_split_path}_{cache_token}"
        if onpolicy_dataset_name is not None:
            self._cache_file = f"{onpolicy_dataset_name}_{self._cache_file}"

        # Per-gripper cache file gets a unique suffix so it doesn't
        # collide with the combined cache used during training.
        if single_gripper is not None:
            self._cache_file = f"{self._cache_file}__gripper_{single_gripper}"

        logger.info(f"cache_dir: {self._cache_dir}, cache_file: {self._cache_file}.h5")

        self.load_discriminator_dataset = load_discriminator_dataset

        self.load_patch = load_patch
        self.patch_width = patch_width
        self.load_contact = load_contact
        self.load_onehot_vec = load_onehot_vec

        self.prob_point_cloud = prob_point_cloud
        self.render_redundancy = render_redundancy
        self.cache_save_freq = cache_save_freq

        self.num_points = num_points
        self.num_obj_points = num_obj_points

        self.cam_coord = cam_coord
        self.num_rotations = num_rotations
        self.grid_res = grid_res
        self.jitter_scale = jitter_scale
        self.robot_prob = robot_prob
        self.contact_radius = contact_radius
        self.dist_above_table = dist_above_table
        self.random_seed = random_seed

        self.inference = inference
        self.rotation_augmentation = rotation_augmentation
        self.downsample_points = downsample_points
        self.add_depth_noise = add_depth_noise

        self.root_dir = root_dir
        self.object_root_dir = object_root_dir
        self.grasp_root_dir = f"{grasp_root_dir}/{grasp_split_path}"

        self.num_grasps_per_object = num_grasps_per_object
        self.min_grasps_gen_th = min_grasps_gen_th
        self.discriminator_ratio = discriminator_ratio
        self.visualize_batch = visualize_batch
        self.onpolicy_dataset_name = onpolicy_dataset_name
        self.onpolicy_dataset_h5_dir = onpolicy_dataset_h5_dir
        self.onpolicy_dataset_json_dir = onpolicy_dataset_json_dir
        self._onpolicy_per_gripper = False
        self._onpolicy_h5_base_dir = None

        self.alternative_json_file_path = alternative_json_file_path

        # Initialize the appropriate grasp dataset reader
        self.grasp_dataset_reader = XGraspJsonDatasetReader(
            self.grasp_root_dir,
            self.object_root_dir,
            alternative_json_file_path=alternative_json_file_path,
        )
        print("DEBUG EVAL: GRASP DATASET READER LOADED")

        cache_load_path = f"{self._cache_dir}/{self._cache_file}.h5"
        self.loading_mode = loading_mode

        if shared_cache is not None:
            # Use pre-loaded shared-memory cache (multi-GPU DDP, preload mode).
            # The cache was already loaded and converted to shared tensors
            # in the main process before mp.spawn, so all workers share the
            # same physical memory (~250GB total instead of per-process).
            logger.info("Using pre-loaded shared-memory cache")
            self.cache = shared_cache

        elif loading_mode in ("shard", "shard-gripper") and use_cache:
            # Shard mode: progressive background loading from H5 shards.
            #   "shard"         → random-split shards (pattern __shard_NNN.h5)
            #   "shard-gripper" → per-gripper shards  (pattern __gripper_*.h5)
            # Falls back to lazy NFS loading if no matching shards exist
            # (e.g., validation set which is a single small file).
            import glob as _glob
            from graspgenx.dataset.xgrasp_dataset_utils import ShardedH5Cache

            assert shard_dir_nfs is not None, "shard_dir_nfs required for shard mode"
            assert (
                shard_dir_local is not None
            ), "shard_dir_local required for shard mode"

            if loading_mode == "shard-gripper":
                shard_pattern = f"{self._cache_file}__gripper_*.h5"
            else:
                shard_pattern = f"{self._cache_file}__shard_*.h5"
            matching_shards = _glob.glob(os.path.join(shard_dir_nfs, shard_pattern))

            if matching_shards:
                logger.info(
                    f"{loading_mode} loading mode — {len(matching_shards)} shards, "
                    f"shard_dir_nfs={shard_dir_nfs}, "
                    f"shard_dir_local={shard_dir_local}, pattern={shard_pattern}, "
                    f"local_rank={shard_local_rank}"
                )
                self.cache = ShardedH5Cache(
                    shard_dir_nfs=shard_dir_nfs,
                    shard_dir_local=shard_dir_local,
                    shard_pattern=shard_pattern,
                    local_rank=shard_local_rank,
                    min_shards_before_training=shard_min_ready,
                )
                # Block until first few shards are copied
                self.cache.wait_for_initial_shards(timeout=600)
            else:
                # No shards found — fall back to lazy loading from
                # the merged cache file (typical for small validation sets).
                nfs_cache_path = f"{shard_dir_nfs}/{self._cache_file}.h5"
                logger.info(
                    f"{loading_mode} mode: no shards found for "
                    f"pattern '{shard_pattern}'. Falling back to lazy "
                    f"loading from NFS: {nfs_cache_path}"
                )
                assert os.path.exists(
                    nfs_cache_path
                ), f"Neither shards nor merged cache found: {nfs_cache_path}"
                self.cache = XGraspGenDatasetCacheLazy(nfs_cache_path)

        elif loading_mode == "lazy" and use_cache:
            print("DEBUG EVAL: LAZY MODE")
            # Lazy mode: read from H5 on demand — no bulk RAM load.
            assert os.path.exists(
                cache_load_path
            ), f"cache file not found {cache_load_path}"
            logger.info(
                f"Lazy loading mode — indexing cache keys from: {cache_load_path}"
            )
            self.cache = XGraspGenDatasetCacheLazy(cache_load_path)
            print("DEBUG EVAL: LAZY MODE CACHE LOADED")
        else:
            # Original preload path
            self.cache = XGraspGenDatasetCache()

            if use_cache:
                assert os.path.exists(
                    cache_load_path
                ), f"cache file not found {cache_load_path}"
                logger.info(f"Using cache file: {cache_load_path}")
                self.cache.load_from_h5_file(cache_load_path)

                # verify all cache keys is within the range
                for k in self.cache.get_keys():
                    assert (
                        k in self.scenes
                    ), "error: cache dataset do not fit the range."
            else:

                # remove the old one, create a new one
                if os.path.exists(cache_load_path):
                    # os.system(f"rm -rf {cache_load_path}")
                    logger.info(
                        f"Cache File Found: {cache_load_path}. Please delete manually if not needed."
                    )
                    self.cache.load_from_h5_file(cache_load_path)

                if (
                    self.onpolicy_dataset_h5_dir is not None
                    and self.onpolicy_dataset_json_dir is not None
                    and self.onpolicy_dataset_name is not None
                ):
                    self.load_onpolicy_dataset()

                self.load_cache(cache_load_path)

        self.scenes = list(self.cache.get_keys())

    def _enumerate_scenes(self, obj_split, gripper_split):
        scenes = []
        for gripper in gripper_split:
            for obj in obj_split:
                scenes.append(f"{gripper}/{obj}")

        return scenes

    def load_gripper_info(self, grippers):
        info = dict()
        for gripper in grippers:
            info[gripper] = resolve_gripper_info(gripper)
        return info

    def load_onpolicy_dataset(self):
        """
        Loads the onpolicy dataset.
        Falls back to per-gripper H5 files when x_grippers.h5 is not found.
        """
        logger.info(
            f"Onpolicy dataset: Loading json path from scratch {self.onpolicy_dataset_name}"
        )
        import h5py

        possible_grasp_keys = [
            "grasps.json"
        ]  # This is because ACRONYM pipeline generates grasp jsons with different names

        h5_base_dir = (
            f"{self.onpolicy_dataset_h5_dir}/x_grasp_"
            f"{self.grasp_split_path}_{self.gripper_split_path}_{self.onpolicy_dataset_name}"
        )
        combined_h5_path = f"{h5_base_dir}/x_grippers.h5"
        json_dir = (
            f"{self.onpolicy_dataset_json_dir}/"
            f"x_grasp_{self.grasp_split_path}_{self.gripper_split_path}_{self.onpolicy_dataset_name}"
        )

        self._onpolicy_h5_base_dir = h5_base_dir

        if os.path.exists(combined_h5_path):
            self._onpolicy_per_gripper = False
            h5 = h5py.File(combined_h5_path, "r")
            gripper_names = list(h5["objects"].keys())
        else:
            self._onpolicy_per_gripper = True
            per_gripper_files = sorted(glob.glob(f"{h5_base_dir}/*.h5"))
            gripper_names = [
                os.path.splitext(os.path.basename(f))[0] for f in per_gripper_files
            ]
            logger.info(
                f"Onpolicy dataset: x_grippers.h5 not found, falling back to "
                f"{len(per_gripper_files)} per-gripper H5 files in {h5_base_dir}"
            )

        map_h5_id_to_uuid = {}
        map_uuid_to_json_file = {}
        t0 = time.time()

        for gripper in gripper_names:
            if gripper not in self.grippers.keys():
                logger.info(f"Gripper {gripper} not in the split list.")
                continue

            if self._onpolicy_per_gripper:
                gripper_h5 = h5py.File(f"{h5_base_dir}/{gripper}.h5", "r")
                h5_objects = gripper_h5["objects"]
            else:
                h5_objects = h5["objects"][gripper]

            h5_object_ids = list(h5_objects.keys())

            json_files = []
            for grasp_key in possible_grasp_keys:
                logger.info(f"Searching using grasp key {grasp_key}")
                json_files += sorted(
                    glob.glob(
                        f"{json_dir}/{gripper}/**/{grasp_key}",
                        recursive=True,
                    )
                )  # This is to account for the case when the dataset is built with MapReduce

            for h5_object_id in h5_object_ids:
                uuid = (
                    h5_objects[h5_object_id]["asset_path"][...].item().decode("utf-8")
                )
                map_h5_id_to_uuid[f"{gripper}/{h5_object_id}"] = uuid

                for json_file_path in json_files:
                    json_path_object_id = json_file_path.split("/")[-2].split("_", 1)[1]
                    map_uuid_to_json_file[f"{gripper}/{json_path_object_id}"] = (
                        json_file_path
                    )

            if self._onpolicy_per_gripper:
                gripper_h5.close()

        if not self._onpolicy_per_gripper:
            h5.close()

        self.map_key_to_json_path_online_dataset = dict()
        for key in self.scenes:
            if key not in map_h5_id_to_uuid:
                logger.info(f"Onpolicy dataset: Key {key} not in h5 dataset")
                continue

            if key not in map_uuid_to_json_file:
                logger.info(f"Onpolicy dataset: Key {key} not in json dataset")
                continue

            self.map_key_to_json_path_online_dataset[key] = map_uuid_to_json_file[key]

        logger.info(f"Onpolicy dataset: That took {time.time() - t0}s. Phew...")

    def load_cache(self, cache_save_path):
        """
        Converts the dataset into a cached file, loads to system memory.
        Logs all exceptions and warnings to a companion JSON file.
        """
        import json as _json
        from tqdm import tqdm

        logger.info("Preloading dataset to memory")

        exceptions_log = []
        entry_stats = []
        NON_FATAL_DATA_ERRORS = {
            DataLoaderError.INSUFFICIENT_GRASPS_FOR_GENERATOR_DATASET,
            DataLoaderError.GRASPS_FILE_NOT_FOUND_OFFLINE,
            DataLoaderError.GRASPS_FILE_NOT_FOUND_ONLINE,
        }
        NON_FATAL_RENDER_ERRORS = {
            DataLoaderError.RENDERING_ERROR_POINT_CLOUD_TOO_SMALL,
        }

        idx_cache_loaded = 0
        for idx in tqdm(range(len(self.scenes))):
            key = self.scenes[idx]
            gripper_name = key.split("/")[0]
            object_id = key.split("/")[1]

            if key in self.cache:
                continue
            else:
                idx_cache_loaded += 1

            onpolicy_json_path = None
            onpolicy_data_found = True
            if self.onpolicy_dataset_json_dir is not None:
                try:
                    onpolicy_json_path = self.map_key_to_json_path_online_dataset[key]
                except:
                    onpolicy_data_found = False

                if (
                    self._onpolicy_per_gripper
                    and self._onpolicy_h5_base_dir is not None
                ):
                    h5_path = f"{self._onpolicy_h5_base_dir}/{gripper_name}.h5"
                else:
                    h5_path = (
                        f"{self.onpolicy_dataset_h5_dir}/x_grasp_"
                        f"{self.grasp_split_path}_{self.gripper_split_path}_{self.onpolicy_dataset_name}/x_grippers.h5"
                    )
                json_dir = (
                    f"{self.onpolicy_dataset_json_dir}/"
                    f"x_grasp_{self.grasp_split_path}_{self.gripper_split_path}_{self.onpolicy_dataset_name}"
                )
            else:
                h5_path = None
                json_dir = None

            error_code, object_grasp_data = load_object_xgrasp_data(
                key,
                self.object_root_dir,
                self.grasp_root_dir,
                min_grasps_gen=self.min_grasps_gen_th,
                load_discriminator_dataset=self.load_discriminator_dataset,
                onpolicy_dataset_json_dir=json_dir,
                onpolicy_dataset_h5_path=h5_path,
                onpolicy_json_path=onpolicy_json_path,
                onpolicy_data_found=onpolicy_data_found,
                grasp_dataset_reader=self.grasp_dataset_reader,
            )

            if error_code in NON_FATAL_DATA_ERRORS:
                detail = getattr(object_grasp_data, "_insufficient_grasps_detail", {})
                exceptions_log.append(
                    {
                        "gripper": gripper_name,
                        "object": object_id,
                        "key": key,
                        "error_code": error_code.value.code,
                        "error_name": error_code.name,
                        "error_description": error_code.value.description,
                        "fatal": False,
                        "detail": detail or {},
                    }
                )
            elif error_code != DataLoaderError.SUCCESS:
                exceptions_log.append(
                    {
                        "gripper": gripper_name,
                        "object": object_id,
                        "key": key,
                        "error_code": error_code.value.code,
                        "error_name": error_code.name,
                        "error_description": error_code.value.description,
                        "fatal": True,
                        "detail": {},
                    }
                )
                continue

            rendering_output = []
            rendering_warnings = []
            POINT_CLOUD_REDUNDANCY = 3
            for _ in range(self.render_redundancy):

                mesh_mode = (
                    False if np.random.random() <= self.prob_point_cloud else True
                )
                load_contact_batch = self.load_contact

                outputs, render_error = render_pc(
                    object_grasp_data,
                    self.num_points * POINT_CLOUD_REDUNDANCY,
                    mesh_mode=mesh_mode,
                )

                if render_error in NON_FATAL_RENDER_ERRORS:
                    rendering_warnings.append(
                        {
                            "error_code": render_error.value.code,
                            "error_name": render_error.name,
                            "detail": outputs.get("rendering_detail", {}),
                        }
                    )

                if render_error in (
                    DataLoaderError.RENDERING_SUCCESS,
                    *NON_FATAL_RENDER_ERRORS,
                ):
                    outputs["mesh_mode"] = mesh_mode
                    outputs["load_contact_batch"] = load_contact_batch

                    n_parent_pos = len(object_grasp_data.positive_grasps)
                    if mesh_mode:
                        positive_grasps = object_grasp_data.positive_grasps.copy()
                        outputs["positive_grasps"] = positive_grasps
                        outputs["grasp_visibility_mask"] = np.ones(
                            n_parent_pos, dtype=bool
                        )
                    else:
                        T_move_to_pc_mean = outputs["T_move_to_pc_mean"]
                        grasps = np.array(
                            [
                                T_move_to_pc_mean @ g
                                for g in object_grasp_data.positive_grasps.copy()
                            ]
                        )
                        gripper_info = self.grippers[gripper_name]

                        mask_grasp_visibility = (
                            filter_xgripper_grasps_by_point_cloud_visibility(
                                grasps, outputs["points"], gripper_info=gripper_info
                            )
                        )
                        if mask_grasp_visibility is not None:
                            positive_grasps = object_grasp_data.positive_grasps.copy()[
                                mask_grasp_visibility
                            ]
                            outputs["positive_grasps"] = positive_grasps
                            outputs["grasp_visibility_mask"] = np.asarray(
                                mask_grasp_visibility, dtype=bool
                            )
                        else:
                            rendering_warnings.append(
                                {
                                    "error_code": DataLoaderError.RENDERING_NO_GRASPS_IN_VISIBLE_POINT_CLOUD.value.code,
                                    "error_name": DataLoaderError.RENDERING_NO_GRASPS_IN_VISIBLE_POINT_CLOUD.name,
                                    "detail": {},
                                }
                            )
                            continue

                    outputs.pop("rendering_detail", None)
                    rendering_output.append(outputs)
                else:
                    rendering_warnings.append(
                        {
                            "error_code": render_error.value.code,
                            "error_name": render_error.name,
                            "detail": outputs.get("rendering_detail", {}),
                        }
                    )

            if len(rendering_output) == 0:
                logger.error(
                    f"{idx}: Unable to preload {key} even after sampling pc {self.render_redundancy} times"
                )
                exceptions_log.append(
                    {
                        "gripper": gripper_name,
                        "object": object_id,
                        "key": key,
                        "error_code": -1,
                        "error_name": "ALL_RENDERINGS_FAILED",
                        "error_description": f"All {self.render_redundancy} rendering attempts failed",
                        "fatal": True,
                        "detail": {"rendering_attempts": rendering_warnings},
                    }
                )
                continue

            if rendering_warnings:
                exceptions_log.append(
                    {
                        "gripper": gripper_name,
                        "object": object_id,
                        "key": key,
                        "error_code": rendering_warnings[0]["error_code"],
                        "error_name": rendering_warnings[0]["error_name"],
                        "error_description": "Some rendering attempts had warnings",
                        "fatal": False,
                        "detail": {"rendering_attempts": rendering_warnings},
                    }
                )

            n_mesh = sum(1 for r in rendering_output if r.get("mesh_mode", False))
            n_rendered = len(rendering_output) - n_mesh
            entry_stats.append(
                {
                    "gripper": gripper_name,
                    "object": object_id,
                    "key": key,
                    "num_renderings": len(rendering_output),
                    "num_mesh": n_mesh,
                    "num_partial_pc": n_rendered,
                }
            )

            self.cache[key] = (object_grasp_data, rendering_output)

            if idx % self.cache_save_freq == 0:
                self.cache.save_to_h5_file(f"{cache_save_path}")

        self.cache.save_to_h5_file(f"{cache_save_path}")

        exceptions_path = cache_save_path.replace(".h5", "_stats.json")

        prev_entry_stats = []
        prev_exceptions = []
        if os.path.exists(exceptions_path):
            try:
                with open(exceptions_path, "r") as f:
                    prev_data = _json.load(f)
                prev_entry_stats = prev_data.get("entry_stats", [])
                prev_exceptions = prev_data.get("exceptions", [])
                logger.info(
                    f"Merging with existing stats: {len(prev_entry_stats)} entries, "
                    f"{len(prev_exceptions)} exceptions"
                )
            except Exception as e:
                logger.warning(f"Could not load existing stats JSON: {e}")

        prev_entry_keys = {s["key"] for s in prev_entry_stats}
        prev_exception_keys = {e["key"] for e in prev_exceptions}
        for s in entry_stats:
            if s["key"] not in prev_entry_keys:
                prev_entry_stats.append(s)
        for e in exceptions_log:
            if e["key"] not in prev_exception_keys:
                prev_exceptions.append(e)

        all_entry_stats = prev_entry_stats
        all_exceptions = prev_exceptions

        n_fatal = sum(1 for e in all_exceptions if e["fatal"])
        n_warn = sum(1 for e in all_exceptions if not e["fatal"])
        total_renderings = sum(s["num_renderings"] for s in all_entry_stats)
        total_mesh = sum(s["num_mesh"] for s in all_entry_stats)
        total_partial_pc = sum(s["num_partial_pc"] for s in all_entry_stats)
        logger.info(
            f"Cache complete: {len(self.cache)} entries cached, "
            f"{n_fatal} fatal errors, {n_warn} warnings. "
            f"Renderings: {total_renderings} total ({total_mesh} mesh, "
            f"{total_partial_pc} partial PC). "
            f"Stats log: {exceptions_path}"
        )
        with open(exceptions_path, "w") as f:
            _json.dump(
                {
                    "summary": {
                        "total_scenes": len(self.scenes),
                        "cached_entries": len(self.cache),
                        "fatal_errors": n_fatal,
                        "warnings": n_warn,
                        "total_renderings": total_renderings,
                        "total_mesh_renderings": total_mesh,
                        "total_partial_pc_renderings": total_partial_pc,
                        "render_redundancy": self.render_redundancy,
                        "prob_point_cloud": self.prob_point_cloud,
                    },
                    "entry_stats": all_entry_stats,
                    "exceptions": all_exceptions,
                },
                f,
                indent=2,
            )

    @classmethod
    def from_config(cls, cfg):
        args = {}
        args["root_dir"] = cfg.root_dir
        args["cache_dir"] = cfg.cache_dir
        args["cache_name"] = cfg.cache_name
        args["num_points"] = cfg.num_points
        args["num_obj_points"] = cfg.num_object_points
        args["cam_coord"] = cfg.cam_coord
        args["num_rotations"] = cfg.num_rotations
        args["grid_res"] = cfg.grid_resolution
        args["jitter_scale"] = cfg.jitter_scale
        args["contact_radius"] = cfg.contact_radius
        args["dist_above_table"] = cfg.dist_above_table
        args["robot_prob"] = cfg.robot_prob
        args["random_seed"] = cfg.random_seed
        args["rotation_augmentation"] = cfg.rotation_augmentation
        args["downsample_points"] = cfg.downsample_points
        args["add_depth_noise"] = cfg.add_depth_noise
        args["load_patch"] = cfg.load_patch
        args["patch_width"] = cfg.patch_width
        args["prob_point_cloud"] = cfg.prob_point_cloud
        args["object_root_dir"] = cfg.object_root_dir
        args["grasp_root_dir"] = cfg.grasp_root_dir
        args["num_grasps_per_object"] = cfg.num_grasps_per_object
        args["load_discriminator_dataset"] = cfg.load_discriminator_dataset
        args["load_contact"] = cfg.load_contact
        args["load_onehot_vec"] = cfg.load_onehot_vec
        args["discriminator_ratio"] = cfg.discriminator_ratio
        args["visualize_batch"] = cfg.visualize_batch
        args["onpolicy_dataset_name"] = cfg.onpolicy_dataset_name
        args["onpolicy_dataset_json_dir"] = cfg.onpolicy_dataset_json_dir
        args["onpolicy_dataset_h5_dir"] = cfg.onpolicy_dataset_h5_dir
        args["render_redundancy"] = cfg.render_redundancy
        args["cache_save_freq"] = cfg.cache_save_freq
        args["single_gripper"] = cfg.get("single_gripper", None)
        args["loading_mode"] = cfg.get("loading_mode", "preload")
        args["alternative_json_file_path"] = cfg.get("alternative_json_file_path", None)
        args["shard_dir_nfs"] = cfg.get("shard_dir_nfs", None)
        args["shard_dir_local"] = cfg.get("shard_dir_local", None)
        args["shard_local_rank"] = cfg.get("shard_local_rank", 0)
        args["shard_min_ready"] = cfg.get("shard_min_ready", 3)
        return args

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        import time as _time

        _t0 = _time.monotonic()

        key = self.scenes[idx]

        object_id = key.split("/")[1]
        gripper_name = key.split("/")[0]

        object_grasp_data, outputs_red = self.cache[key]
        _t_cache = _time.monotonic()

        # Deep-copy numpy arrays from the selected rendering so that the
        # original cache stays read-only.  When using fork-based multi-GPU
        # training this prevents copy-on-write page faults; for single-GPU
        # it avoids silently mutating the cache across __getitem__ calls.
        _sel = random.choice(outputs_red)
        outputs = {
            k: (
                v.copy()
                if isinstance(v, np.ndarray)
                else v.clone() if isinstance(v, torch.Tensor) else v
            )
            for k, v in _sel.items()
        }

        _t_deepcopy = _time.monotonic()

        # If arrays were stored as shared-memory tensors (legacy path),
        # convert back to numpy so all downstream code works unchanged.
        for _k in list(outputs.keys()):
            if isinstance(outputs[_k], torch.Tensor):
                outputs[_k] = outputs[_k].numpy()

        load_contact_batch = outputs["load_contact_batch"]
        mask = torch.randint(0, outputs["points"].shape[0], (self.num_points,))
        outputs["points"] = outputs["points"][mask]
        outputs["gripper_depth"] = self.grippers[gripper_name].depth
        outputs["gripper_symmetry"] = self.grippers[gripper_name].symmetric

        T_move_to_pc_mean = outputs["T_move_to_pc_mean"]
        grasps = outputs["positive_grasps"]
        grasps = np.array([T_move_to_pc_mean @ g for g in grasps])

        # Extra stuff added to outputs later. TODO - Clean this all up
        xyz = outputs["points"]
        if type(xyz) == np.ndarray:
            xyz = torch.from_numpy(xyz).float()
        num_points = self.num_points
        seg = 5 * np.ones(num_points).astype(np.int64)
        rgb = (
            150.0
            * np.vstack(
                [np.ones(num_points), np.zeros(num_points), np.zeros(num_points)]
            ).T
        )
        cam_pose = np.eye(4)
        rgb = torch.from_numpy(rgb).float()
        seg = torch.from_numpy(seg).float()
        cam_pose = torch.from_numpy(cam_pose).float()
        outputs["inputs"] = torch.cat([xyz, rgb], dim=1)
        outputs["seg"] = seg
        outputs["rgb"] = rgb
        outputs["cam_pose"] = cam_pose

        outputs["scene"] = self.scenes[idx]
        outputs["task"] = "pick"

        try:
            grasps_ground_truth = _safe_array_copy(object_grasp_data.positive_grasps)
            grasps_ground_truth = np.array(
                [T_move_to_pc_mean @ g for g in grasps_ground_truth]
            )
            outputs["grasps_ground_truth"] = grasps_ground_truth
        except Exception as e:
            logger.warning(
                f"[GRASPS GROUND TRUTH FAILED] scene={self.scenes[idx]} idx={idx} | "
                f"grasps_ground_truth failed: {e}"
            )
            outputs["grasps_ground_truth"] = None

        if self.load_discriminator_dataset:
            negative_grasps = None
            if object_grasp_data.negative_grasps is not None:
                try:
                    negative_grasps = _safe_array_copy(
                        object_grasp_data.negative_grasps
                    )
                    if len(negative_grasps) > 0:
                        negative_grasps[:, 3, 3] = 1.0
                        negative_grasps = np.array(
                            [T_move_to_pc_mean @ g for g in negative_grasps]
                        )
                    else:
                        negative_grasps = None
                except Exception as e:
                    negative_grasps = None

            outputs["negative_grasps"] = negative_grasps

        outputs["grasps"] = [grasps]
        outputs["names"] = ["obj0"]

        if not self.load_contact:
            # Sanitize the dictionary output for diffusion model training.
            blacklist_keys = [
                "instance_masks",
                "grasping_masks",
                "contact_dirs",
                "approach_dirs",
                "ee_pose",
                "obj_pose",
                "object_inputs",
                "bottom_center",
                "object_center",
                "placement_masks",
                "placement_region",
            ]
            for key in blacklist_keys:
                if key in outputs:
                    del outputs[key]

        _t_preprocess = _time.monotonic()

        positive_grasps_onpolicy = None
        negative_grasps_onpolicy = None
        _n_pos_op, _n_neg_op = 0, 0
        _cached_pos_shape, _cached_neg_shape = None, None
        if self.load_discriminator_dataset:
            _cached_pos_op = object_grasp_data.positive_grasps_onpolicy
            _cached_neg_op = object_grasp_data.negative_grasps_onpolicy
            _cached_pos_shape = (
                _cached_pos_op.shape if _cached_pos_op is not None else None
            )
            _cached_neg_shape = (
                _cached_neg_op.shape if _cached_neg_op is not None else None
            )
            if object_grasp_data.positive_grasps_onpolicy is not None:
                try:
                    if len(object_grasp_data.positive_grasps_onpolicy) > 0:
                        positive_grasps_onpolicy = _safe_array_copy(
                            object_grasp_data.positive_grasps_onpolicy
                        )
                        _n_pos_op = len(positive_grasps_onpolicy)
                        positive_grasps_onpolicy[:, 3, 3] = 1.0
                        positive_grasps_onpolicy = np.array(
                            [
                                T_move_to_pc_mean @ np.array(g)
                                for g in positive_grasps_onpolicy.tolist()
                            ]
                        )
                except Exception as e:
                    logger.warning(
                        f"[ONPOLICY POS COPY FAILED] idx={idx} scene={self.scenes[idx]} | {e}"
                    )
                    positive_grasps_onpolicy = None

            if object_grasp_data.negative_grasps_onpolicy is not None:
                try:
                    if len(object_grasp_data.negative_grasps_onpolicy) > 0:
                        negative_grasps_onpolicy = _safe_array_copy(
                            object_grasp_data.negative_grasps_onpolicy
                        )
                        _n_neg_op = len(negative_grasps_onpolicy)
                        negative_grasps_onpolicy[:, 3, 3] = 1.0
                        negative_grasps_onpolicy = np.array(
                            [T_move_to_pc_mean @ g for g in negative_grasps_onpolicy]
                        )
                except Exception as e:
                    logger.warning(
                        f"[ONPOLICY NEG COPY FAILED] idx={idx} scene={self.scenes[idx]} | {e}"
                    )
                    negative_grasps_onpolicy = None

        _t_onpolicy = _time.monotonic()

        if self.rotation_augmentation:
            pc = outputs["points"]
            if type(pc) == torch.Tensor:
                pc = pc.cpu().numpy()

            if len(pc.shape) == 3 and pc.shape[0] == 1:
                pc = pc.squeeze(0)

            T_world_to_pcmean = tra.translation_matrix(-pc.mean(axis=0))

            T_pcmean_to_world = tra.inverse_matrix(T_world_to_pcmean)
            T_rotation = get_rotation_augmentation(stratified_sampling=False)
            T_aug = T_pcmean_to_world @ T_rotation @ T_world_to_pcmean
            pc = tra.transform_points(pc, T_aug)
            xyz = torch.from_numpy(pc).float()

            T_aug = T_aug.reshape(1, 4, 4)

            def _can_augment(arr):
                return (
                    arr is not None
                    and hasattr(arr, "ndim")
                    and arr.ndim >= 2
                    and arr.shape[-2] > 0
                )

            if "grasps_ground_truth" in outputs and _can_augment(
                outputs["grasps_ground_truth"]
            ):
                grasps_ground_truth = outputs["grasps_ground_truth"]
                outputs["grasps_ground_truth"] = T_aug @ grasps_ground_truth

            if "grasps" in outputs:
                output_grasps_rotated = []
                for grasps in outputs["grasps"]:
                    if _can_augment(grasps):
                        grasps_rotated = T_aug @ grasps
                        output_grasps_rotated.append(grasps_rotated)
                outputs["grasps"] = output_grasps_rotated

                if self.load_discriminator_dataset:
                    if "negative_grasps" in outputs and _can_augment(
                        outputs["negative_grasps"]
                    ):
                        negative_grasps = outputs["negative_grasps"]
                        outputs["negative_grasps"] = T_aug @ negative_grasps

            outputs["points"] = xyz
            rgb = outputs["rgb"]
            center = xyz.mean(dim=0)
            outputs["inputs"] = torch.cat([xyz - center, rgb], dim=1)

            if _can_augment(positive_grasps_onpolicy):
                positive_grasps_onpolicy = T_aug @ positive_grasps_onpolicy

            if _can_augment(negative_grasps_onpolicy):
                negative_grasps_onpolicy = T_aug @ negative_grasps_onpolicy

        _t_rotation = _time.monotonic()

        obj_asset_path = object_grasp_data.object_asset_path
        obj_scale = object_grasp_data.object_scale
        obj_pose = T_move_to_pc_mean

        if self.rotation_augmentation:
            obj_pose = (T_aug @ obj_pose)[0]

        scene_info = {
            "assets": [
                obj_asset_path,
            ],
            "scales": [
                obj_scale,
            ],
            "poses": [
                obj_pose,
            ],
            "grippers": [
                gripper_name,
            ],
        }

        outputs["scene_info"] = scene_info

        # gripper data
        if self.load_onehot_vec:
            gripper_index = self.gripper_split.index(gripper_name)
            assert gripper_index < 10, "Only support maximal 10 grippers"
            outputs["onehot"] = torch.zeros((10,), dtype=torch.float32)
            outputs["onehot"][gripper_index] = 1.0

        outputs["z_offset"] = torch.tensor(
            (self.grippers[gripper_name].depth,), dtype=torch.float32
        )
        outputs["sweep_volume"] = torch.from_numpy(
            self.grippers[gripper_name].sweep_volume.astype(np.float32)
        )
        outputs["gripper_type"] = self.grippers[gripper_name].gripper_type

        sweep_volume_open = self.grippers[gripper_name].sweep_volume.astype(np.float32)
        sweep_volume_mid = self.grippers[gripper_name].sweep_volume_mid.astype(
            np.float32
        )
        outputs["sweep_volume_open_and_mid"] = torch.from_numpy(
            np.concatenate([sweep_volume_open, sweep_volume_mid], axis=0)
        )

        gripper_open_ptc, gripper_close_ptc = sample_gripper_pointclouds(
            self.grippers[gripper_name], self.num_points
        )
        outputs["gripper_open_ptc"] = torch.from_numpy(gripper_open_ptc)
        outputs["gripper_close_ptc"] = torch.from_numpy(gripper_close_ptc)

        vol_tsdf = np.stack(
            [
                self.grippers[gripper_name].vol_tsdf[f"open_tsdf"],
                self.grippers[gripper_name].vol_tsdf[f"close_tsdf"],
            ],
            axis=0,
        )
        outputs["gripper_vol_tsdf"] = torch.from_numpy(vol_tsdf.astype(np.float32))

        outputs["gripper_pointnet_repr"] = torch.from_numpy(
            np.concatenate(
                [
                    self.grippers[gripper_name].pointnet_vae["open"],
                    self.grippers[gripper_name].pointnet_vae["half"],
                    self.grippers[gripper_name].pointnet_vae["close"],
                ],
                axis=0,
            ).astype(np.float32)
        )

        _t_gripper = _time.monotonic()

        if self.load_discriminator_dataset:
            if len(outputs.get("grasps", [])) == 0:
                _n_pos_onpol = (
                    len(positive_grasps_onpolicy)
                    if positive_grasps_onpolicy is not None
                    else 0
                )
                _n_neg_onpol = (
                    len(negative_grasps_onpolicy)
                    if negative_grasps_onpolicy is not None
                    else 0
                )
                logger.warning(
                    f"[NO POSITIVE GRASPS] scene={self.scenes[idx]} idx={idx} | "
                    f"cached_pos_op={_cached_pos_shape} cached_neg_op={_cached_neg_shape} "
                    f"post_transform_pos_op={_n_pos_onpol} post_transform_neg_op={_n_neg_onpol} "
                    f"grasps_ground_truth={type(outputs.get('grasps_ground_truth')).__name__}"
                )
                positive_grasps = None
            else:
                positive_grasps = outputs["grasps"][0]

            if "negative_grasps" in outputs:
                negative_grasps = outputs["negative_grasps"]
            else:
                negative_grasps = None

            batch_data = load_discriminator_batch_with_stratified_sampling(
                self.num_grasps_per_object,
                positive_grasps,
                self.discriminator_ratio,
                negative_grasps,
                positive_grasps_onpolicy=positive_grasps_onpolicy,
                negative_grasps_onpolicy=negative_grasps_onpolicy,
            )

            if batch_data is None:
                return {"invalid": True}

            outputs.update(batch_data)
            outputs["grasps_highres"] = positive_grasps
        else:
            if len(outputs.get("grasps", [])) == 0:
                logger.warning(
                    f"[NO POSITIVE GRASPS - GEN] gripper={gripper_name} object={object_id} "
                    f"idx={idx} scene={self.scenes[idx]} | "
                    f"grasps_ground_truth={type(outputs.get('grasps_ground_truth')).__name__} — resampling"
                )
                # Random resample instead of returning {"invalid": True}.
                # The previous behavior set up a DDP deadlock at end of epoch:
                # an all-bad batch would be fully dropped by collate, the
                # train loop's `if data is None: continue` then skipped the
                # per-step collective on that rank only, while the other 15
                # ranks called it — leaving them stuck for the watchdog
                # timeout. With seed=0 this fired deterministically at
                # Epoch 10 batch 480 every multinode run (jobs 27392259,
                # 27392347, 27401338, 27440286, 27450949). Resampling
                # guarantees __getitem__ always returns a real sample, so
                # batches are never empty and per-step collectives stay in
                # sync. Bad rate is ~1-3% so expected recursion depth ~1.01.
                return self[np.random.randint(0, len(self))]

            if self.num_grasps_per_object != -1:
                grasps_gt = outputs["grasps"][0]

                mask_grasps_filtered = np.random.randint(
                    0, len(grasps_gt), self.num_grasps_per_object
                )

                outputs["grasps"] = grasps_gt[mask_grasps_filtered]
                outputs["grasps_highres"] = grasps_gt

            for key in ["points"]:
                if type(outputs[key]) == np.ndarray:
                    outputs[key] = torch.from_numpy(outputs[key])
                outputs[key] = outputs[key].unsqueeze(0).repeat(1, 1, 1)
            outputs["grasps"] = torch.from_numpy(
                np.array(outputs["grasps"]).astype(np.float32)
            )
            outputs["grasps_highres"] = torch.from_numpy(
                np.array(outputs["grasps_highres"]).astype(np.float32)
            )

            if self.visualize_batch and gripper_name in [
                "franka_panda",
                "fetch_robot, robotiq_2f_85",
            ]:
                grasps_gt = outputs["grasps"]
                if type(grasps_gt) == list and len(grasps_gt) == 1:
                    grasps_gt = grasps_gt[0]

                pc = outputs["points"][0].cpu().numpy()

                visualize_xgripper_generator_dataset(
                    obj_pose,
                    grasps_gt,
                    pc,
                    gripper_visual_mesh=self.grippers[gripper_name].visual_mesh,
                )

        if len(outputs["points"].shape) == 2:
            outputs["points"] = outputs["points"].unsqueeze(0)

        # _t_end = _time.monotonic()
        # _total = _t_end - _t0
        # logger.info(
        #     f"[GETITEM TIMING] idx={idx} total={_total:.4f}s | "
        #     f"cache_read={_t_cache - _t0:.4f} "
        #     f"deepcopy={_t_deepcopy - _t_cache:.4f} "
        #     f"preprocess={_t_preprocess - _t_deepcopy:.4f} "
        #     f"onpolicy={_t_onpolicy - _t_preprocess:.4f} (cached={_cached_pos_shape},{_cached_neg_shape} post={_n_pos_op}+{_n_neg_op}) "
        #     f"rotation={_t_rotation - _t_onpolicy:.4f} "
        #     f"gripper={_t_gripper - _t_rotation:.4f} "
        #     f"dis_batch={_t_end - _t_gripper:.4f}"
        # )

        return outputs
