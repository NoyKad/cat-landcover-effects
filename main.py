# Standard library imports
import os
from os.path import splitext
import re
import shutil

# Related third party imports
import cv2 as cv
import geopandas as gpd
import imageio
import numpy as np
from osgeo import osr, ogr, gdal
import pandas as pd
from shapely.geometry import LineString, Point
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# step 1


class Getcover:
    def __init__(self, path_to_satellite_img, path_to_building_img):
        self.path_to_satellite_img = path_to_satellite_img
        self.path_to_building_img = path_to_building_img
        self.satellite_img = cv.imread(path_to_satellite_img, 3)  # read 3 chanels (BGR)
        self.building_img = cv.imread(
            path_to_building_img, cv.IMREAD_GRAYSCALE
        )  # read in gary scale

    def Polygonized(self, input_path, output_path, bld=True):
        """input_path  : reference image path
        output_path : path to temp shp file"""

        raster = gdal.Open(input_path)
        band = raster.GetRasterBand(1)
        drv = ogr.GetDriverByName("ESRI Shapefile")
        outfile = drv.CreateDataSource(output_path)
        outlayer = outfile.CreateLayer("polygonized raster", srs=None)
        newField = ogr.FieldDefn("DN", ogr.OFTReal)
        outlayer.CreateField(newField)
        gdal.Polygonize(band, None, outlayer, 0, [])
        outfile = None

        polygonized_shp = gpd.read_file(output_path)  # load polygonized sahpe
        polygonized_shp.crs = {"init": "epsg:2039"}  # set coords Israeli TM grid
        # take buildings color value only and clean too small shapes

        if bld:
            polygonized_shp["Area"] = polygonized_shp.area
            # polygonized_shp = polygonized_shp[((polygonized_shp.DN == 248) | (polygonized_shp.DN == 255)) & (polygonized_shp.Area > 2)] # blds and roads
            polygonized_shp = polygonized_shp[
                (polygonized_shp.DN == 248) & (polygonized_shp.Area > 20)
            ]

        polygonized_shp.to_file(output_path)
        self.polygonized_shp_path = output_path
        self.polygonized_shp = polygonized_shp
        print(
            f"Polygonized done! shape file created at {output_path} \nwith the shape of {polygonized_shp.shape} and colums {polygonized_shp.columns}"
        )

    def Polynegative(self, tmp_path):
        """tmp_path : path to temp tif file"""

        # Create a negative buildings polygon
        neg = np.zeros(self.satellite_img.shape[:-1])  # (self.building_img)
        # create a blank img in the size and reference of the buildings img

        reference = gdal.Open(
            self.path_to_satellite_img, gdal.GA_ReadOnly
        ).GetGeoTransform()
        nrows, ncols = neg.shape
        output_raster = gdal.GetDriverByName("GTiff").Create(
            tmp_path, ncols, nrows, 1, gdal.GDT_Float32
        )  # Open the file
        output_raster.SetGeoTransform(reference)  # Specify its coordinates
        srs = osr.SpatialReference()  # Establish its coordinate encoding
        srs.ImportFromEPSG(2039)  # This one specifies TM Israel.

        output_raster.SetProjection(srs.ExportToWkt())  # Exports the coordinate system
        # to the file
        output_raster.GetRasterBand(1).WriteArray(neg)  # Writes my array to the raster
        output_raster.FlushCache()

        tmp_shp_path = splitext(tmp_path)[0] + ".shp"

        # Polygonize
        raster = gdal.Open(tmp_path)
        band = raster.GetRasterBand(1)
        drv = ogr.GetDriverByName("ESRI Shapefile")
        outfile = drv.CreateDataSource(tmp_shp_path)
        outlayer = outfile.CreateLayer("polygonized raster", srs=None)
        gdal.Polygonize(band, None, outlayer, 0, [])
        outfile = None

        # Create squer polygon in size and reference of buildings image
        self.polynegative_shp_path = tmp_shp_path

        squer_polygon = gpd.read_file(self.polynegative_shp_path)
        squer_polygon.crs = {"init": "epsg:2039"}
        build_inter = gpd.overlay(
            self.polygonized_shp, squer_polygon, how="intersection"
        )
        build_symdiff = gpd.overlay(
            build_inter, squer_polygon, how="symmetric_difference"
        )
        build_symdiff.crs = {"init": "epsg:2039"}
        build_symdiff.to_file(self.polynegative_shp_path)
        print(f"Polynegative done! file at {self.polynegative_shp_path}")

    def clip_raster(self, output_path):
        """output_path : cliped satellite image path"""
        options = gdal.WarpOptions(
            cutlineDSName=self.polynegative_shp_path, cropToCutline=False
        )
        outBand = gdal.Warp(
            srcDSOrSrcDSTab=self.path_to_satellite_img,
            destNameOrDestDS=output_path,
            options=options,
        )
        if outBand == None:
            print("ERROR with clip_raster method!")
        else:
            print(f"clip_raster done! file at {output_path}")
            self.cliped_raster = output_path
        outBand = None

    def denoise_and_cluster(self):
        # Denoised
        img = cv.imread(self.cliped_raster, 3)
        args = {
            "src": img,
            "dst": None,
            "h": 3,
            "hColor": 3,
            "templateWindowSize": 7,
            "searchWindowSize": 21,
        }
        denoised = cv.fastNlMeansDenoisingColored(**args)
        kmeans_denoised_img = self.kmeans_cv(denoised)
        self.set_reference(self.cliped_raster, kmeans_denoised_img)
        print("denoise_and_cluster() Done!")

    def kmeans_cv(self, img_array):
        Z = img_array.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 7
        _, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_PP_CENTERS)
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
        reference = gdal.Open(path_to_src, gdal.GA_ReadOnly).GetGeoTransform()
        array = self.bgr2gray(new_array)
        nrows, ncols = array.shape

        out_path = splitext(path_to_src)[0] + "_kmean.tif"  # for buildings clip

        output_raster = gdal.GetDriverByName("GTiff").Create(
            out_path, ncols, nrows, bands=1
        )  # , eType=gdal.GDT_Float32)  # Open the file
        output_raster.SetGeoTransform(reference)  # Specify its coordinates
        srs = osr.SpatialReference()  # Establish its coordinate encoding
        srs.ImportFromEPSG(2039)  # This one specifies WGS84 lat long.

        output_raster.SetProjection(srs.ExportToWkt())  # Exports the coordinate system
        # to the file
        output_raster.GetRasterBand(1).WriteArray(
            array
        )  # Writes my array to the raster
        output_raster.FlushCache()
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


class Find_nodes:
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


class Clean_gps_reads(Find_nodes):
    def __init__(self, filepath):
        self.filepath = filepath
        # super(Clean_gps_reads, self).__init__(activityRadius, activityTime, nodesDistance)
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
        # linetracks = []
        for trackid in tracks.activity.unique():
            steps = tracks.loc[tracks.activity == trackid, "geometry"]
            if len(steps) <= 1:
                continue
            time = pd.to_datetime(tracks.loc[tracks.activity == trackid, "DATETIME"])
            dt = time.iloc[-1] - time.iloc[0]
            #         inx = idlist.index(trackid)
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

        df_out = gpd.GeoDataFrame(tmp_df)
        df_out.crs = {"init": "epsg:2039", "no_defs": True}
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
        df = pd.read_csv(self.filepath, parse_dates=["DATETIME"], dayfirst=True)
        gdf = gpd.GeoDataFrame(
            data=df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude)
        )
        gdf.crs = {"init": "epsg:4326", "no_defs": True}  #'EPSG:4326'
        gdf.to_crs(epsg=2039, inplace=True)
        gdf["DT"] = (gdf.DATETIME - gdf.DATETIME.shift()).dt.seconds
        gdf["dis"] = gdf.geometry.distance(gdf.geometry.shift())
        gdf["speed_kmh"] = gdf.dis * 3.6 / gdf.DT

        # point(i)'speed' is based on the previous point(i-1) information
        # angle - referes to current point angle between previous and up coming points
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
        print("saved...")

    def Area_of_Interest(self, path):
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
        print("Area_of_Interest pass!")

    def intersect_poly(self, path):
        poly = gpd.read_file(path + "/Polygonized.shp")
        aoi = gpd.read_file(path + "/package.gpkg", layer="RBF95")
        aoi["geometry"] = aoi.buffer(30)
        inter = gpd.overlay(poly, aoi, how="intersection")
        inter.to_file(path + "/package.gpkg", layer="interpoly", driver="GPKG")
        print("intersect_poly pass!")


# This function uses the classes above
# import this one only


def shell(path_to_satellite_img, path_to_building_img, output_path, path_to_gps_points):
    """The shell function organized in the right sequence the call for each one
    of the classes and the methods in the main file

        Inputs:
        path_to_satellite_img : str
        A path to aerial image, must contain some kind of geographical anchoring.
        path_to_building_img : str
        A path to geographical image, containing clear building polygones, must shere the same dimensions
        of the aerial image and the same geographical anchoring.
        This input uses to remove roof tops from the aerial image, and can be ignored by passing bld=False
        in the Polygonized method.
        output_path : str
        path to output folder, where the output files will be saved (if not exists it will be created)
        path_to_gps_points : str
        path to gps points, excepted to be in .csv format. Must share the same geographical area of the
        other inoputs.
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
    api.Polygonized(
        api.path_to_building_img,
        output_path=tmp_directory + "Polygonized.shp",
        bld=True,
    )
    # Polynegative
    api.Polynegative(tmp_path=tmp_directory + "Polynegative.tif")
    # Clip raster
    api.clip_raster(output_path=output_directory + name + "_cliped.tif")
    # Denoise and Cluster
    api.denoise_and_cluster()
    # Polygonized the Clustered raster
    api.Polygonized(
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
    obj = Clean_gps_reads(filepath=path_to_gps_points)
    obj.load_points_and_clean()
    try:
        print(obj.gdf.shape)  # to_file
        obj.save(output_directory)
        obj.Area_of_Interest(output_directory)
        obj.intersect_poly(output_directory)
    except:
        print(f"Error with gps points file: {name}")
    # remove polygonized.shp files from folder
    for file in os.listdir(output_directory):
        if re.match("Polygonized", file):
            os.remove(output_directory + "/" + file)
    print("shell done!")
