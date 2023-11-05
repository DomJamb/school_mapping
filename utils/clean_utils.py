import re
import os
import geojson
import itertools
import logging

import numpy as np
import pandas as pd
import geopandas as gpd

import ee
import eeconvert as eec

import wxee
import xarray as xr
import rasterio as rio

import networkx as nx
from rapidfuzz import fuzz
from tqdm import tqdm
import data_utils

logging.basicConfig(level=logging.INFO)

def _filter_uninhabited_locations(data, buffer_size, pbar=None):
    data = data.reset_index(drop=True)
    ghsl_sum = []
    for index in range(len(data)):
        if pbar:
            iso_code = data.iso.values[0]
            pbar.set_description(f"Processing {iso_code} {index}/{len(data)}")

        subdata = data.iloc[[index]]
        subdata = subdata.to_crs("EPSG:3857")
        subdata["geometry"] = subdata["geometry"].centroid.buffer(buffer_size)
        subdata = subdata.to_crs("EPSG:4326")
        
        geometry = eec.gdfToFc(subdata[["geometry"]])
        image = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_C/2018")
        image = image.select("built_characteristics").clip(geometry)
        file = image.wx.to_tif(
            out_dir="./",
            description="temp",
            region=geometry.geometry(),
            scale=10,
            crs="EPSG:4326",
            progress=False,
        )

        with rio.open(file[0], "r") as src:
            image = src.read(1)
            image[image == -32768] = 0
            sum_ = image.sum()
            ghsl_sum.append(sum_)

    data["GHSL"] = ghsl_sum
    data = data[data["GHSL"] > 0]
    data = data.reset_index(drop=True)
    return data


def _filter_pois_within_school_vicinity(
    config, 
    buffer_size, 
    iso_codes=None, 
    sname="clean", 
    name="filtered"
):
    cwd = os.path.dirname(os.getcwd())
    data_dir = config["vectors_dir"]
    out_dir = os.path.join(data_dir, "non_school", name)
    out_dir = data_utils._makedir(out_dir)

    #school_file = os.path.join(cwd, data_dir, "school", f"{sname}.geojson")
    #school = gpd.read_file(school_file)
    #logging.info(f"School data dimensions: {school.shape}")

    nonschool_dir = os.path.join(data_dir, "non_school")
    exclude = [f"{sname}.geojson", f"{name}.geojson"]
    nonschool = data_utils._read_data(nonschool_dir, exclude=exclude)
    logging.info(f"Non-school data dimensions: {nonschool.shape}")

    if not iso_codes:
        iso_codes = list(nonschool.iso.value_counts()[::-1].index)

    data = []
    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    pbar = tqdm(iso_codes, total=len(iso_codes), bar_format=bar_format)
    for iso_code in pbar:
        ########################################################
        # TODO: Put this back at the top and remove sname folder 
        #school_file = os.path.join(
        #    cwd, 
        #    data_dir, 
        #    "school", 
        #    sname, 
        #    f"{iso_code}_{sname}.geojson"
        #)
        #if not os.path.exists(school_file):
        #    continue
        #school = gpd.read_file(school_file)
        ##########################################################
        
        pbar.set_description(f"Processing {iso_code}")
        out_subfile = os.path.join(out_dir, f"{iso_code}_{name}.geojson")

        if not os.path.exists(out_subfile):
            school_sub = school[school["iso"] == iso_code]
            nonschool_sub = nonschool[nonschool["iso"] == iso_code]

            # Convert school and non-school data CRS to EPSG:3857
            nonschool_temp = data_utils._convert_to_crs(
                nonschool_sub, target_crs="EPSG:3857"
            )
            nonschool_temp["geometry"] = nonschool_temp["geometry"].buffer(buffer_size)
            nonschool_temp["index"] = nonschool_sub.index
            school_temp = data_utils._convert_to_crs(school_sub, target_crs="EPSG:3857")
            school_temp["geometry"] = school_temp["geometry"].buffer(buffer_size)

            # Filter out non-school POIs that intersect with buffered school locations
            intersecting = school_temp.sjoin(nonschool_temp, how="inner")["index"]
            nonschool_sub = nonschool_sub[~nonschool_temp["index"].isin(intersecting)]

            # Save country-level dataset
            columns = config["columns"]
            nonschool_sub = nonschool_sub[columns]
            nonschool_sub.to_file(out_subfile, driver="GeoJSON")

        subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        data.append(subdata)

    # Combine datasets
    filtered_file = os.path.join(cwd, nonschool_dir, f"{name}.geojson")
    data = data_utils._concat_data(data)
    data.to_file(filtered_file, driver="GeoJSON")
    return data


def _filter_pois_with_matching_names(data, priority, threshold, buffer_size):
    # Get connected components within a given buffer size and get groups with size > 1
    data = data_utils._connect_components(data, buffer_size)
    group_count = data.group.value_counts()
    groups = group_count[group_count > 1].index

    uid_network = []
    for index in range(len(groups)):
        # Get pairwise combination of names within a group
        subdata = data[data.group == groups[index]][["UID", "name"]]
        subdata = list(subdata.itertuples(index=False, name=None))
        combs = itertools.combinations(subdata, 2)

        # Compute rapidfuzz partial ratio score
        uid_edge_list = []
        for comb in combs:
            score = fuzz.partial_ratio(
                data_utils._clean_text(comb[0][1]), data_utils._clean_text(comb[1][1])
            )
            uid_edge_list.append(
                (comb[0][0], comb[0][1], comb[1][0], comb[1][1], score)
            )
        columns = ["source", "name_1", "target", "name_2", "score"]
        uid_edge_list = pd.DataFrame(uid_edge_list, columns=columns)
        uid_network.append(uid_edge_list)

    # Generate graph and get connected components
    if len(uid_network) > 0:
        uid_network = pd.concat(uid_network)
        uid_network = uid_network[uid_network.score > threshold]
        columns = ["source", "target", "score"]
        graph = nx.from_pandas_edgelist(uid_network[columns])
        connected_components = nx.connected_components(graph)
        groups = {
            num: index
            for index, group in enumerate(connected_components, start=1)
            for num in group
        }

        if len(groups) > 0:
            data["group"] = np.nan
            for uid, value in groups.items():
                data.loc[data["UID"] == uid, "group"] = value
            max_group = int(data["group"].max()) + 1
            fillna = list(range(max_group, len(data) + max_group))
            data["group"] = data.apply(
                lambda x: x["group"]
                if not np.isnan(x["group"])
                else fillna[int(x.name)],
                axis=1,
            )
            data = data_utils._drop_duplicates(data, priority)

    return data


def clean_data(
    config,
    category,
    iso_codes=None,
    name="clean",
    gee=True
):
    if gee:
        ee.Authenticate()
        ee.Initialize()

    data_dir = os.path.join(config["vectors_dir"], category)
    out_dir = os.path.join(config["vectors_dir"], category, name)
    out_dir = data_utils._makedir(out_dir)

    if category == "non_school":
        data = _filter_pois_within_school_vicinity(
            config, 
            buffer_size=config['school_buffer_size'], 
            iso_codes=iso_codes,
        )
    else:
        data = data_utils._read_data(
            data_dir, 
            exclude=[f"{name}.geojson"]
        )
    data = data.drop_duplicates("geometry", keep="first")

    if not iso_codes:
        iso_codes = list(data.iso.value_counts()[::-1].index)

    out_data = []
    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
    pbar = tqdm(iso_codes, total=len(iso_codes), bar_format=bar_format)
    for iso_code in pbar:
        pbar.set_description(f"Processing {iso_code}")
        out_subfile = os.path.join(out_dir, f"{iso_code}_{name}.geojson")

        if not os.path.exists(out_subfile):
            subdata = data[data["iso"] == iso_code].reset_index(drop=True)
            geoboundaries = data_utils._get_geoboundaries(config, iso_code, adm_level="ADM1")
            geoboundaries = geoboundaries[["shapeName", "geometry"]]
            geoboundaries = geoboundaries.dropna(subset=["shapeName"])
            subdata = subdata.sjoin(geoboundaries, how="left", predicate="within")

            # Split the data into smaller admin boundaries fo scalability
            out_subdata = []
            for shape_name in subdata.shapeName.unique():
                pbar.set_description(f"Processing {iso_code} {shape_name}")
                subsubdata = subdata[subdata["shapeName"] == shape_name]
                subsubdata = subsubdata.drop(["index_right", "shapeName"], axis=1)
                subsubdata = subsubdata.reset_index(drop=True)

                if len(subsubdata) > 0:
                    columns = config["columns"]
                    subsubdata = data_utils._connect_components(subsubdata, config['buffer_size'])
                    subsubdata = data_utils._drop_duplicates(subsubdata, config['priority'])

                    if "giga_id_school" in subsubdata.columns:
                        columns = columns + ["giga_id_school"]
                    subsubdata = subsubdata[columns]

                    subsubdata = _filter_pois_with_matching_names(
                        subsubdata,
                        priority=config['priority'],
                        threshold=config['name_match_threshold'],
                        buffer_size=config['name_match_buffer_size'],
                    )[columns]
                    out_subdata.append(subsubdata)

            # Save cleaned file
            out_subdata = data_utils._concat_data(out_subdata, verbose=False)
            if (category == "school") and ("GHSL" not in out_subdata.columns):
                out_subdata = _filter_uninhabited_locations(
                    out_subdata, config['ghsl_buffer_size'], pbar=pbar
                )
            out_subdata.to_file(out_subfile, driver="GeoJSON")
        out_subdata = gpd.read_file(out_subfile).reset_index(drop=True)
        out_data.append(out_subdata)

    # Save combined dataset
    out_file = os.path.join(os.path.dirname(out_dir), f"{name}.geojson")
    data = data_utils._concat_data(out_data, out_file)
    data.to_file(out_file, driver="GeoJSON")

    return data
