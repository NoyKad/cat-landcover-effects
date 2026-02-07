"""Author details: Noy Kadosh (noykadosh123@gmail.com), Jul 2020.

This program is part of a preprocessing stage in a cats' roaming research.
The program takes aerial image (with geographical anchoring, e.g. world file),
along with gps points, and returns the area of interest as GIS file, for
farther processing in a manual stage.

The main method to call is 'shell'
"""

# Standard library imports
import os
import re
import shutil
import warnings
from os.path import splitext

# Related third party imports
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.mask import mask
from shapely.geometry import LineString, box, shape
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


# step 1


class Getcover:
    def __init__(self, path_to_satellite_img, path_to_building_img):
        self.path_to_satellite_img = path_to_satellite_img
        self.path_to_building_img = path_to_building_img
        self.satellite_img = cv2.imread(
            path_to_satellite_img, 3
        )  # read 3 channels (BGR)
        self.building_img = cv2.imread(
            path_to_building_img, cv2.IMREAD_GRAYSCALE
        )  # read in gray scale

    def polygonized(self, input_path, output_path, bld=True):
        """input_path  : reference image path
        output_path : path to temp shp file"""

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(input_path) as src:
            band = src.read(1)
            transform = src.transform
            raster_crs = src.crs if src.crs else "EPSG:2039"
            nodata = src.nodata
            mask_arr = None if nodata is None else band != nodata
            shapes_generator = features.shapes(band, mask=mask_arr, transform=transform)
            geometries, values = [], []
            for geom, value in shapes_generator:
                if value is None:
                    continue
                geometries.append(shape(geom))
                values.append(float(value))

        polygonized_shp = gpd.GeoDataFrame(
            {"DN": values}, geometry=geometries, crs=raster_crs
        )

        if bld and not polygonized_shp.empty:
            polygonized_shp["Area"] = polygonized_shp.area
            polygonized_shp = polygonized_shp[
                (polygonized_shp.DN == 248) & (polygonized_shp.Area > 20)
            ]

        polygonized_shp.to_file(output_path)
        self.polygonized_shp_path = output_path
        self.polygonized_shp = polygonized_shp
        print(
            f"Done! - `polygonized` method: shape file created at {output_path}"
            f"with the shape of {polygonized_shp.shape} and columns {polygonized_shp.columns}"
        )

    def polynegative(self, tmp_path):
        """tmp_path : path to temp tif file"""

        # Create a negative buildings polygon
        tmp_shp_path = splitext(tmp_path)[0] + ".shp"
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

        with rasterio.open(self.path_to_satellite_img) as src:
            meta = src.meta.copy()
            bounds = src.bounds
            raster_crs = src.crs if src.crs else "EPSG:2039"
            height, width = src.height, src.width

        neg = np.zeros((height, width), dtype=np.float32)
        meta.update(driver="GTiff", count=1, dtype="float32", nodata=0)
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(neg[np.newaxis, ...])

        square_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        square_gdf = gpd.GeoDataFrame(
            {"DN": [0]}, geometry=[square_geom], crs=raster_crs
        )
        square_gdf.to_file(tmp_shp_path)

        self.polynegative_shp_path = tmp_shp_path

        polygonized = self.polygonized_shp
        if polygonized.crs != square_gdf.crs:
            polygonized = polygonized.to_crs(square_gdf.crs)

        build_inter = gpd.overlay(polygonized, square_gdf, how="intersection")
        build_symdiff = gpd.overlay(build_inter, square_gdf, how="symmetric_difference")
        build_symdiff.crs = square_gdf.crs
        build_symdiff.to_file(self.polynegative_shp_path)
        self.polynegative_gdf = build_symdiff
        print(
            f"Done! - `polynegative` method: file created at {self.polynegative_shp_path}"
        )

    def clip_raster(self, output_path):
        """output_path : cliped satellite image path"""
        cutline = getattr(self, "polynegative_gdf", None)
        if cutline is None:
            cutline = gpd.read_file(self.polynegative_shp_path)
        with rasterio.open(self.path_to_satellite_img) as src:
            if cutline.crs != src.crs and src.crs is not None:
                cutline = cutline.to_crs(src.crs)
            shapes = [geom.__geo_interface__ for geom in cutline.geometry]
            out_image, out_transform = mask(src, shapes, crop=False)
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "driver": "GTiff",
                }
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"Done! - `clip_raster` method: file created at {output_path}")
        self.cliped_raster = output_path

    def denoise_and_cluster(self):
        # De-noised
        img = cv2.imread(self.cliped_raster, 3)
        args = {
            "src": img,
            "dst": None,
            "h": 3,
            "hColor": 3,
            "templateWindowSize": 7,
            "searchWindowSize": 21,
        }
        denoised = cv2.fastNlMeansDenoisingColored(**args)
        kmeans_denoised_img = self.kmeans_cv(denoised)
        self.set_reference(self.cliped_raster, kmeans_denoised_img)
        print(
            "Done! - `denoise_and_cluster` method: de-noised and clustered image created"
        )

    def kmeans_cv(self, img_array):
        Z = img_array.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 7
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()].reshape((img_array.shape))
        return res

    def bgr2gray(self, bgr):
        # B G R
        b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

    def set_reference(self, path_to_src, new_array):
        with rasterio.open(path_to_src) as src:
            transform = src.transform
            crs = src.crs
            height, width = src.height, src.width

        array = self.bgr2gray(new_array).astype(np.float32)
        out_path = splitext(path_to_src)[0] + "_kmean.tif"

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "nodata": None,
        }

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(array, 1)

        print(f"Saved to {out_path}")
        self.gray_kmean_path = out_path


def del_directory_content(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# step 2


class FindNodes:

    def __init__(self, activityRadius, activityTime, nodesDistance):

        self.activityRadius = activityRadius  # meter
        self.activityTime = activityTime  # seconds
        self.nodesDistance = nodesDistance  # meter

    def get_mean_center(self, tmp):
        return self.points[tmp, :].mean(axis=0)

    def clean_activity(self, tmp, center):
        # clean node's starting and ending points
        ptmp = self.points[tmp, :]
        L = np.array([np.linalg.norm(p - center) for p in ptmp]).mean() * 2
        for t, p in zip(tmp, ptmp):
            if np.linalg.norm(p - center) > L:
                tmp.remove(t)
            else:
                break

        for t, p in zip(tmp[::-1], ptmp[::-1]):
            if np.linalg.norm(p - center) > L:
                tmp.remove(t)
            else:
                break
        return tmp

    def iterate_over_points(self, gdfpoints):

        self.points = np.vstack((gdfpoints.geometry.x.values, gdfpoints.geometry.y)).T
        points = self.points
        pointsTime = gdfpoints.DATETIME.values.astype("datetime64[s]")
        center = self.points[0, :]  # firts x and y coords
        tmp = []  # initialize emply list
        activity = np.zeros(points.shape[0], dtype=int)  # initialize activity array
        nodeTable = {}  # pd.DataFrame() # not sure yet
        count = 1

        for i in np.arange(points.shape[0] - 1, dtype=int):
            if np.linalg.norm(points[i, :] - center) <= self.activityRadius:
                tmp.append(i)
                center = self.get_mean_center(tmp)
                centerTime = (pointsTime[tmp[-1]] - pointsTime[tmp[0]]).astype(int)
            else:
                if centerTime >= self.activityTime:
                    # node found!
                    tmp = self.clean_activity(tmp, center)

                    if nodeTable:

                        centers_dis = np.array(
                            [
                                np.linalg.norm(nodeTable[n]["center"] - center)
                                for n in nodeTable.keys()
                            ]
                        )
                        centers_idx = np.array(list(nodeTable.keys()))

                        if centers_dis.min() < self.nodesDistance:
                            # if one of the previous nodes matches, than update
                            idx = np.where(centers_dis == centers_dis.min())[0][0]
                            node_num = centers_idx[idx]
                            oldtmp = nodeTable[node_num]["TMP"]
                            tmp = np.append(oldtmp, tmp)
                            nodeTable[node_num]["center"] = self.get_mean_center(tmp)
                            nodeTable[node_num]["TMP"] = tmp.tolist()
                            activity[tmp] = node_num
                            tmp = []

                        else:
                            # if not match, create new node
                            count += 1
                            nodeTable.update({count: {"center": center, "TMP": tmp}})
                            activity[tmp] = count
                            tmp = []

                    else:
                        # if nodeTable is empyt, populate with first node
                        nodeTable.update({count: {"center": center, "TMP": tmp}})
                        activity[tmp] = count
                        tmp = []

                    centerTime = 0

                else:
                    # not a node
                    center = points[i, :]
                    tmp = []

        return activity, nodeTable


class CleanGpsReads(FindNodes):

    def __init__(self, filepath):
        self.filepath = filepath
        super().__init__(activityRadius=30, activityTime=60 * 12, nodesDistance=20)

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        v1_u = self.unit_vector(v1.tolist())
        v2_u = self.unit_vector(v2.tolist())
        return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    def trackBuilder(self, tracks, autolist):
        tmp_df = pd.DataFrame(
            columns=[
                "activity",
                "totalDuration",
                "geometry",
                "startNode",
                "endNode",
                "numOfpoints",
            ]
        )
        for trackid in tracks.activity.unique():
            steps = tracks.loc[tracks.activity == trackid, "geometry"]
            if len(steps) <= 1:
                continue
            time = pd.to_datetime(tracks.loc[tracks.activity == trackid, "DATETIME"])
            dt = time.iloc[-1] - time.iloc[0]
            line = []
            for step in steps:
                line.append((step.x, step.y))

            idx = autolist.index(trackid)
            data = {
                "activity": trackid,
                "totalDuration": dt.seconds,
                "geometry": [LineString(line)],
                "startNode": autolist[idx - 1],
                "endNode": autolist[idx + 1],
                "numOfpoints": len(steps),
            }
            df_line = pd.DataFrame(
                data,
                columns=[
                    "activity",
                    "totalDuration",
                    "geometry",
                    "startNode",
                    "endNode",
                    "numOfpoints",
                ],
            )

            # Add record DataFrame of compiled records
            tmp_df = pd.concat([tmp_df, df_line], sort=False)

        df_out = gpd.GeoDataFrame(tmp_df).set_crs("EPSG:2039")
        # df_out.crs = {"init": "epsg:2039", "no_defs": True}
        df_out["Length"] = df_out.length
        df_out["speed"] = df_out.Length / df_out.totalDuration
        df_out = df_out.astype({"activity": "int", "numOfpoints": "int"})
        return df_out[
            [
                "geometry",
                "activity",
                "Length",
                "totalDuration",
                "speed",
                "startNode",
                "endNode",
                "numOfpoints",
            ]
        ].reset_index(drop=True)

    def load_points_and_clean(self):
        df = pd.read_csv(self.filepath, parse_dates=["DATETIME"], dayfirst=False)
        gdf = gpd.GeoDataFrame(
            data=df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude)
        )
        gdf.crs = {"init": "epsg:4326", "no_defs": True}  #'EPSG:4326'
        gdf.to_crs(epsg=2039, inplace=True)
        gdf["DT"] = (gdf.DATETIME - gdf.DATETIME.shift()).dt.seconds
        gdf["dis"] = gdf.geometry.distance(gdf.geometry.shift())
        gdf["speed_kmh"] = gdf.dis * 3.6 / gdf.DT

        # point(i)'speed' is based on the previous point(i-1) information
        # angle - refers to current point angle between previous and upcoming points
        np.seterr(divide="ignore", invalid="ignore")  # ignore divide by nan
        crispy_data = gpd.GeoDataFrame(
            {
                "pre": gdf.geometry.shift(-1, fill_value=gdf.geometry.iloc[-1]).apply(
                    lambda S: np.array([S.x, S.y])
                ),
                "precent": gdf.geometry.apply(lambda S: np.array([S.x, S.y])),
                "post": gdf.geometry.shift(1, fill_value=gdf.geometry.iloc[0]).apply(
                    lambda S: np.array([S.x, S.y])
                ),
            }
        )
        crispy_data["AB"] = crispy_data.pre - crispy_data.precent
        crispy_data["BC"] = crispy_data.post - crispy_data.precent
        crispy_data["angle"] = crispy_data.apply(
            lambda x: self.angle_between(x.AB, x.BC), axis=1
        )
        gdf["angle"] = crispy_data["angle"]

        activity, nodes = self.iterate_over_points(gdf)
        self.nodes = nodes
        # id trips
        gdf["activity"] = activity
        autoid = 1000
        tmp = []
        for i in range(len(gdf) - 1):
            if gdf.activity.iloc[i] == 0 and gdf.activity.iloc[i + 1] == 0:
                tmp.append(i)

            if gdf.activity.iloc[i] == 0 and gdf.activity.iloc[i + 1] != 0:
                tmp.append(i)
                gdf.loc[tmp, "activity"] = autoid
                tmp = []
                autoid += 1

            if i == len(gdf) - 2:
                if gdf.activity.iloc[i] == 0 and gdf.activity.iloc[i + 1] == 0:
                    tmp.append(i)
                    tmp.append(i + 1)
                    gdf.loc[tmp, "activity"] = autoid

        autolist = [gdf.activity.iloc[0]]
        for i in gdf.activity:
            if autolist[-1] != i:
                autolist.append(i)
        if autolist[-1] >= 1000:
            autolist.pop(-1)
        if autolist[0] >= 1000:
            autolist.pop(0)
        autoarr = np.array(autolist)

        def extend_tracks(act):
            x = gdf.loc[gdf.activity == act].index
            lsx = x.tolist()
            return [lsx[0] - 1] + lsx + [lsx[-1] + 1]

        for act in autoarr[autoarr >= 1000]:
            inx = extend_tracks(act)
            gdf.loc[inx, "activity"] = act

        self.tracks = self.trackBuilder(
            gdf.loc[gdf.activity.isin(autoarr[autoarr >= 1000])], autolist
        )
        self.gdf = gdf

    def save(self, path):
        self.gdf["DATETIME"] = self.gdf.DATETIME.astype("str")
        self.gdf.to_file(path + "package.gpkg", layer="gdf", driver="GPKG")
        # self.tracks.to_file(path + "package.gpkg", layer='trecks', driver="GPKG")
        print(
            f"Done! - `save` method: file created at {path}package.gpkg with the shape of {self.gdf.shape} and columns {self.gdf.columns}"
        )

    def area_of_interest(self, path):
        path = path + "/package.gpkg"
        gdf = gpd.read_file(path, layer="gdf")

        x = gdf.geometry.x.values
        y = gdf.geometry.y.values

        scaler = StandardScaler()
        points = scaler.fit_transform(np.vstack((x, y)).T)

        labels = OneClassSVM(gamma=0.1, kernel="rbf", nu=0.05).fit_predict(points)

        gdf["labels_95"] = labels
        gdf.to_file(path, layer="gdf", driver="GPKG")
        AOI = gdf.loc[gdf.labels_95 == +1, "geometry"].unary_union.convex_hull
        gpd.GeoSeries(AOI, name="RBF95").to_file(path, layer="RBF95", driver="GPKG")
        print(
            f"Done! - `area_of_interest` method: file created at {path} with the shape of {gdf.shape} and columns {gdf.columns}"
        )

    def intersect_poly(self, path):
        poly = gpd.read_file(path + "/Polygonized.shp")
        aoi = gpd.read_file(path + "/package.gpkg", layer="RBF95").set_crs(poly.crs)
        aoi["geometry"] = aoi.buffer(30)
        inter = gpd.overlay(poly, aoi, how="intersection")
        inter.to_file(path + "/package.gpkg", layer="interpoly", driver="GPKG")
        print(
            f"Done! - `intersect_poly` method: file created at {path}/package.gpkg with the shape of {inter.shape} and columns {inter.columns}"
        )


# This function uses the classes above
# import this one only


def main(
    path_to_satellite_img: str,
    path_to_building_img: str,
    output_path: str,
    path_to_gps_points: str,
):
    """The main function organized in the right sequence the call for each one
    of the classes and the methods in the main file

        Inputs:
            path_to_satellite_img : str
                - A path to aerial image, must contain some kind of geographical anchoring.
            path_to_building_img : str
                - A path to geographical image, containing clear building polygones, must share the same dimensions
                of the aerial image and the same geographical anchoring. This input uses to remove roof tops from the aerial image,
                and can be ignored by passing bld=False in the Polygonized method.
            output_path : str
                - Path to output folder, where the output files will be saved (if not exists it will be created)
            path_to_gps_points : str
                - Path to gps points, excepted to be in .csv format. Must share the same geographical area of the other inputs.
    """

    name = os.path.basename(path_to_satellite_img).split(".")[0]
    output_directory = output_path + name + "/"
    tmp_directory = output_path + "/tmp/"
    make_dir(output_directory)
    make_dir(tmp_directory)
    # initialize
    api = Getcover(
        path_to_satellite_img=path_to_satellite_img,
        path_to_building_img=path_to_building_img,
    )
    # Polygonized
    api.polygonized(
        api.path_to_building_img,
        output_path=tmp_directory + "Polygonized.shp",
        bld=True,
    )
    # Polynegative
    api.polynegative(tmp_path=tmp_directory + "Polynegative.tif")
    # Clip raster
    api.clip_raster(output_path=output_directory + name + "_cliped.tif")
    # Denoise and Cluster
    api.denoise_and_cluster()
    # Polygonized the Clustered raster
    api.polygonized(
        api.gray_kmean_path,
        output_path=output_directory + "/Polygonized.shp",
        bld=False,
    )

    # add original image to directory
    path_to_satellite_world = (
        os.path.join(*path_to_satellite_img.split("/")[:-1])
        + "/"
        + path_to_satellite_img.split("/")[-1].split(".")[0]
        + ".pgw"
    )
    shutil.copyfile(
        path_to_satellite_img, output_directory + path_to_satellite_img.split("/")[-1]
    )
    shutil.copyfile(
        path_to_satellite_world,
        output_directory + path_to_satellite_world.split("/")[-1],
    )
    # del directory
    del_directory_content(tmp_directory)
    # load, clean gps points and save
    cleaner = CleanGpsReads(filepath=path_to_gps_points)
    cleaner.load_points_and_clean()
    try:
        cleaner.save(output_directory)
        cleaner.area_of_interest(output_directory)
        cleaner.intersect_poly(output_directory)
    except Exception as e:
        print(f"Error with gps points file: {name}, {e}")
    # remove polygonized.shp files from folder
    for file in os.listdir(output_directory):
        if re.match("Polygonized", file):
            os.remove(output_directory + "/" + file)
    print(f"Done! - `main` function: processing completed for file {name}")
