from pyspark import SparkConf, SparkContext
import sys
import json
import csv
import itertools
import time
import math
import random
import operator
import os
import glob
import re

# variables
input_file_path = sys.argv[1]
n_cluster = int(sys.argv[2])
out_file1 = sys.argv[3]
out_file2 = sys.argv[4]

os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

alpha = 3
large_cluster = 2
all_points = dict()
myFiles = []
for v1, v2, v3 in os.walk(input_file_path):
    for v0 in v3:
        myFiles.append(os.path.join(v1, v0))
myFiles.sort()
print(myFiles)


# ======================================================== FUNCTIONS ========================================================
def ifdig(string):
    return int(string) if string.isdigit() else string


def natural_keys(string):
    return [ifdig(c) for c in re.split('(\d+)', string)]


def eucledian_dist(vector1, vector2):
    x = vector1
    y = vector2
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return distance


def findClosestCentroid(data):
    # This function is used to check which cluster the data point should be in by comparing distance between point and all centroids.
    # output: closest centroid's coordinates
    min_dist = []
    closest_centroid = []
    for centroid, point in data:  # centroid = (1, 1), point = (0, [1, 1])
        eulc_dist = eucledian_dist(centroid, point[1])
        min_dist.append(eulc_dist)
        closest_centroid.append(centroid)
    min_dist_val = min_dist.index(min(min_dist))
    closest_centroid_val = closest_centroid[min_dist_val]
    return closest_centroid_val


def recomputingCentroid(cluster_data):
    centroid = [0 for x in range(dimensions)]
    data_pts = set()
    no_of_items = len(cluster_data)
    for points in cluster_data:
        p = points[0]
        data_pts.add(points[1][0])
        for d in range(dimensions):
            centroid[d] += p[d]
    for c in range(len(centroid)):
        centroid[c] = centroid[c] / no_of_items
    return (tuple(centroid), data_pts)


def k_Means_old(data_points, n_cluster_local, tolerance, purposeForKMeans):

    data_points_list = list(data_points.items())

    ##################################################################### Step 0 - Generating initial centroid #####################################################################

    # Option 1: Generating n_clusters random centroids
    centroids = []
    if purposeForKMeans == "DS_again":
        centroids = random.sample(list(data_points.values()), n_cluster_local)

    # Option 2: KMeans++                     
    if purposeForKMeans == "DS" or purposeForKMeans == "CS_RS":
        centroids = []
        distinct_points = [x for x in data_points]
        randCentroid = random.choice(distinct_points)
        centroids.append(data_points[randCentroid])
        for c_id in range(n_cluster_local - 1):
            dist = []
            max_dist_point = []
            data_points_list = tuple(data_points.items())
            for i in range(len(data_points_list)):
                point = data_points_list[i][1]
                d = sys.maxsize
                for j in range(len(centroids)):
                    temp_dist = eucledian_dist(point, centroids[j])
                    d = min(d, temp_dist)
                dist.append(d)
                max_dist_point.append(data_points_list[i][0])
            max_dist_of_closest_centroid_index = dist.index(max(dist))
            next_centroid = max_dist_point[max_dist_of_closest_centroid_index]
            centroids.append(data_points[next_centroid])


    print("CENTROIDS:", centroids)
    # Centroids generated in variable centroids as a list.

    iteration = 0
    while (iteration <= 100):
        # print("\n\n#ITERATION:", iteration)
        s = time.time()

        ######################## Step 1 + 2 - KMeans : Find the distance of each point to both the centroids and choose a cluster for each point + Recalculate new centroids #########################
        # k-means - returns old centroid,new centroid and data point index of points in this cluster
        km = sc.parallelize(data_points_list).map(lambda x: [(c, x) for c in centroids]). \
            map(lambda x: (findClosestCentroid(x), (x[0][1][1], x[0][1]))) \
            .groupByKey(). \
            mapValues(list). \
            mapValues(recomputingCentroid) \
            .collect()                                                                      
        # DS NEEDS n_cluster number of clusters
        # if len(km) < n_cluster_local, exit and repeat to get n_cluster_local clusters
        if purposeForKMeans == "DS" or purposeForKMeans == "DS_again":
            if len(km) < n_cluster_local:
                return "False"
            for km0 in km:
                if len(km0[1][1]) == 1:
                    return "False"

        # This step is to create the previous centroids list
        # This step is to create the previous cs_rs list
        prev_centroids = []
        for i in range(len(km)):
            data = km[i]
            old_cluster = data[0];
            new_cluster = data[1][0];
            prev_centroids.append(old_cluster)
            centroids[i] = new_cluster

        ##################################################################### Step 3 - stopping condition. Check for tolerance #####################################################################
        optimized = True
        for c in range(len(centroids)):
            original_centroid = prev_centroids[c]  # [1,2,3]
            current_centroid = centroids[c]  # [11,12,13]
            zipped = zip(current_centroid, original_centroid)
            centroid_sum = 0
            for o1, o2 in zipped:
                if o2 != 0:
                    centroid_sum += (o1 - o2) / (o2 * 100)
            if centroid_sum > tolerance:
                optimized = False
        if optimized:
            # print("TOLERANCE LEVEL ISSUE. STOP K-means")
            break

        e = time.time()
        print("duration:", e - s)
        iteration += 1

    result = sc.parallelize(km).map(lambda x: (x[1][1], find_stats(data_points, x[1][1]))).collect()
    print ("purposeForKMeans:", purposeForKMeans, " | Iterations:", iteration)
    print ("centroids length:", n_cluster_local)
    print("===== K-means end =====")
    return result


def k_Means(data_points, n_cluster_local, tolerance, purposeForKMeans, good_centroids, bad_centroids):
    #print("\n===== K-means begin =====")
    LENGTHH = 0     #number of singles
    reverse_map = {v: k for k, v in data_points.items()}

    ##################################################################### Step 0 - Generating initial centroid #####################################################################

    # Option 1: Generating n_clusters random centroids
    data_points_list = list(data_points.items())
    centroids = []
    # Option 2: KMeans++             
    if purposeForKMeans == "DS" or purposeForKMeans == "CS_RS" or purposeForKMeans == "DS_again":
        centroids = []
        distinct_points = [x for x in data_points]
        randCentroid = random.choice(distinct_points)
        centroids.append(data_points[randCentroid])
        for c_id in range(n_cluster_local - 1):
            dist = []
            max_dist_point = []
            data_points_list = tuple(data_points.items())
            for i in range(len(data_points_list)):
                point = data_points_list[i][1]
                d = sys.maxsize
                for j in range(len(centroids)):
                    temp_dist = eucledian_dist(point, centroids[j])
                    d = min(d, temp_dist)
                dist.append(d)
                max_dist_point.append(data_points_list[i][0])
            max_dist_of_closest_centroid_index = dist.index(max(dist))
            next_centroid = max_dist_point[max_dist_of_closest_centroid_index]
            centroids.append(data_points[next_centroid])

    #print("CENTROIDS:", centroids)
    # Centroids generated in variable centroids as a list.
    iteration = 0
    o_n_cents = []

    for j in range(len(centroids)):  # new
        o_n_cents.append([centroids[j], centroids[j]])

    while (iteration <= 100):
        s = time.time()

        ######################## Step 1 + 2 - KMeans : Find the distance of each point to both the centroids and choose a cluster for each point + Recalculate new centroids #########################
        # k-means - returns old centroid,new centroid and data point index of points in this cluster

        # original
        km = sc.parallelize(data_points_list).map(lambda x: [(c, x) for c in centroids]). \
            map(lambda x: (findClosestCentroid(x), (x[0][1][1], x[0][1]))) \
            .groupByKey(). \
            mapValues(list). \
            mapValues(recomputingCentroid) \
            .collect()
        # DS NEEDS n_cluster number of clusters
        # if len(km) < n_cluster_local, exit and repeat to get n_cluster_local clusters
        if purposeForKMeans == "DS" or purposeForKMeans == "DS_again":
            if len(km) < n_cluster_local:
                return "False"

        # This step is to create the previous centroids list
        prev_centroids = []
        for i in range(len(km)):
            data = km[i]
            old_cluster = data[0];
            new_cluster = data[1][0];
            prev_centroids.append(old_cluster)
            centroids[i] = new_cluster

        for r in range(len(o_n_cents)):
            original, old = o_n_cents[r]
            for p in range(len(prev_centroids)):
                if old == prev_centroids[p]:
                    o_n_cents[r][1] = centroids[p]
        if (purposeForKMeans == "DS" or purposeForKMeans == "DS_again"):
            bad_cents = []
            for km0 in km:
                length = len(km0[1][1])
                new_cen = km0[1][0]
                if length == 1:
                    LENGTHH += 1
                    ch = 0
                    og_rem = "";
                    new_rem = ""
                    for og, new in o_n_cents:
                        if new == new_cen:
                            original_cent = og
                            bad_cents.append(original_cent)
                            ch = 1
                            og_rem = og;
                            new_rem = new
                            break
                    if ch == 1:
                        o_n_cents.remove([og_rem, new_rem])
            good_cents = []
            for gc in o_n_cents:
                good_cents.append(gc[0])
            if len(bad_cents) > 0:
                return ["False", good_cents, bad_cents]

        ##################################################################### Step 3 - stopping condition. Check for tolerance #####################################################################
        optimized = True
        for c in range(len(centroids)):
            original_centroid = prev_centroids[c]  # [1,2,3]
            current_centroid = centroids[c]  # [11,12,13]
            zipped = zip(current_centroid, original_centroid)
            centroid_sum = 0
            for o1, o2 in zipped:
                if o2 != 0:
                    centroid_sum += (o1 - o2) / (o2 * 100)
            if centroid_sum > tolerance:
                optimized = False
        if optimized:
            break

        e = time.time()
        print("duration:", e - s)
        iteration += 1
		
    result = sc.parallelize(km).map(lambda x: (x[1][1], find_stats(data_points, x[1][1]))).collect()
    return result


def find_stats(data_points, set_of_cluster_members):
    points = []
    for p in set_of_cluster_members:    points.append(data_points[p])
    N = len(points)
    points_sq = []
    for p in points:
        points_sq.append(tuple([p0 ** 2 for p0 in p]))
    SUMSQ = tuple([sum(a) for a in zip(*points_sq)])
    SUM = tuple([sum(a) for a in zip(*points)])

    return (N, SUM, SUMSQ)


# round 2 onwards
def updateStats(clusterStats, pt): 
    n_, sum_, sumsq_ = clusterStats
    sum_ = list(sum_);
    sumsq_ = list(sumsq_);
    n_ += 1
    for d in range(len(pt)):
        sum_[d] += pt[d]
        sumsq_[d] += pt[d] ** 2
    return (n_, tuple(sum_), tuple(sumsq_))


def calcCentroidVariance(clusterStats):
    n_, sum_, sumsq_ = clusterStats
    centroids_ = [];
    variances_ = []
    for i in range(len(sum_)):
        centroids_.append(sum_[i] / n_)
        variances_.append((sumsq_[i] / n_) - (sum_[i] / n_) ** 2)
    return (tuple(centroids_), tuple(variances_))


def calcCentroid(clusterStats):
    n_, sum_, sumsq_ = clusterStats
    centroids_ = [];
    variances_ = []
    for i in range(len(sum_)):
        centroids_.append(sum_[i] / n_)
    return tuple(centroids_)


def mahalanobisDist(point, centroids_, variances_): 
    y = 0
    for d in range(len(point)):
        if variances_[d] != 0:
            y += ((point[d] - centroids_[d]) / math.sqrt(variances_[d])) ** 2
    md = math.sqrt(y)
    return md


def checkIfPointsInDSorCS(data_points_, ds_set_, ds_points_):
    discarded_pts = []

    for pt_idx, pt in data_points_.items(): 
        for cl_idx, cl_stats in ds_set_.items():  
            centroids, variances = calcCentroidVariance(cl_stats)
            md = mahalanobisDist(pt, centroids, variances)
            # print ((md), alpha * math.sqrt(dimensions))
            if md < alpha * math.sqrt(dimensions):
                discarded_pts.append(pt_idx)  # delete point
                ds_points_[pt_idx] = cl_idx  # update ds_set_ and ds_points_
                ds_set_[cl_idx] = updateStats(cl_stats, pt)
                break

    for dp in discarded_pts:
        del data_points_[dp]
    return (data_points_, ds_set_, ds_points_)


def updateStatsWhenCSsCombined(bigClusterStats, smallClusterStats):
    big_n_, big_sum_, big_sumsq_ = bigClusterStats
    big_sum_ = list(big_sum_);
    big_sumsq_ = list(big_sumsq_);
    small_n_, small_sum_, small_sumsq_ = smallClusterStats

    big_n_ += small_n_
    for d in range(dimensions):
        big_sum_[d] += small_sum_[d]
        big_sumsq_[d] += small_sumsq_[d]

    return (big_n_, tuple(big_sum_), tuple(big_sumsq_))


def combineCSs(cs_set_, cs_points_):
    cs_set_ASC = sorted(list(cs_set_.items()), key=lambda x: x[1][0]) 
    no_of_merges = 0

    for small_idx, small_stats in cs_set_ASC:
        small_centroid = calcCentroid(small_stats)
        small_size = small_stats[0]
        cs_set_DESC = sorted(list(cs_set_.items()), key=lambda x: x[1][0],reverse=True)  
        for big_idx, big_stats in cs_set_DESC:
            big_size = big_stats[0]
            if (small_idx != big_idx) and (small_size <= big_size):
                big_centroids, big_variances = calcCentroidVariance(big_stats)
                md = mahalanobisDist(small_centroid, big_centroids, big_variances)
                if md < alpha * math.sqrt(dimensions):
                    no_of_merges += 1
                    # change cluster assignments for all points belonging to small cluster
                    for p in cs_points_:
                        if cs_points_[p] == small_idx:
                            cs_points_[p] = big_idx
                    # update stats of the big cluster
                    cs_set_[big_idx] = updateStatsWhenCSsCombined(big_stats, small_stats)
                    del cs_set_[small_idx]
                    break
    return (cs_set_, cs_points_)


def mergeCSwithDS(ds_set_, ds_points_, cs_set_, cs_points_):
   
    cs_set_list_ = list(
        cs_set_.items()) 
    for cs_idx, cs_stats in cs_set_list_:
        cs_centroid = calcCentroid(cs_stats)
        for ds_idx, ds_stats in ds_set_.items():
            ds_centroids, ds_variances = calcCentroidVariance(ds_stats)
            md = mahalanobisDist(cs_centroid, ds_centroids, ds_variances)
            if md < alpha * math.sqrt(dimensions):
              
                ds_set_[ds_idx] = updateStatsWhenCSsCombined(ds_stats, cs_stats)
                del cs_set_[cs_idx]
                # delete cluster from cs_set_ and cs_points
                to_delete = []
                for p in cs_points_:
                    if cs_points_[p] == cs_idx:
                        ds_points_[p] = ds_idx
                        to_delete.append(p)
                for ele in to_delete:
                    del cs_points_[ele]
                break
    return (ds_set_, ds_points_, cs_set_, cs_points_)

    # ======================================================== START ========================================================


start = time.time()

SparkContext.setSystemProperty('spark.executor.memory', '4g')
SparkContext.setSystemProperty('spark.driver.memory', '4g')
sc = SparkContext('local[*]', 'task1')
sc.setLogLevel("ERROR")

intermediate_results = [] 
ds_set = {}  
ds_points = dict() 
cs_set = {}  
cs_points = dict() 
rs_points_map = dict() 

for rounds in range(len(myFiles)):
    print("\n\nROUND:", rounds)

    fileInfos = []
    f = open(myFiles[rounds], "r");
    lines = f.readlines();
    f.close()
    for l in lines:
        l = l[:-1]
        items = l.split(',')
        arr = []
        for it in range(len(items)):
            if it == 0:
                arr.append(int(items[it]))
            else:
                arr.append(float(items[it]))
        fileInfos.append(arr)

    f = fileInfos
    data_points = dict()
    for d in f:
        data_points[d[0]] = tuple(d[1:])
    dimensions = 0
    for a in data_points:
        dimensions = len(data_points[a])
        break

    if rounds == 0:

        # Sampling the data
        data_points_list = tuple(data_points.items())
        data_size = len(data_points)
        ds_data = random.sample(data_points_list, int(data_size * 0.4));
        csrs_data = list(set(data_points_list) - set(ds_data))
        ds_data = dict(ds_data);
        csrs_data = dict(csrs_data)
        reverse_map = {v: k for k, v in data_points.items()}

        # Step DS processing
        DS = k_Means(ds_data, n_cluster, 0.5,"DS", [], [])  
        bad_cents = []; good_cents = []
        while (DS == "False" or DS[0] == "False"):
            if DS == "False":
                DS = k_Means(ds_data, n_cluster, 0.5, "DS_again", [], [])
            else:
                for b in DS[2]:
                    data_pt = b; data_pt_idx = reverse_map[b]
                    all_points[data_pt_idx] = -1
                    del ds_data[data_pt_idx]
                DS = k_Means(ds_data, n_cluster, 0.5, "DS_again", DS[1], DS[2])

        # first time ds_set created
        for cluster_idx in range(len(DS)):
            pts_idx, stats = DS[cluster_idx]  
            ds_set.update(
                {cluster_idx: stats}) 
            for pts in pts_idx:
                ds_points[pts] = cluster_idx  

        # Go through points and see if they can be added to DS - new
        csrs_data, ds_set, ds_points = checkIfPointsInDSorCS(csrs_data, ds_set, ds_points)



        #new:
        if len(csrs_data) != 0:
            #print("csrs empty")
            # CS and RS processing
            CS_RS = k_Means(csrs_data, min(2 * n_cluster, len(csrs_data)), 4, "CS_RS", [], []) 
            cs_tracker = 0
            for cluster_idx in range(len(CS_RS)):
                pts_idx, stats = CS_RS[
                    cluster_idx] 
                if len(pts_idx) == 1:  # RS
                    for p in pts_idx:
                        rs_points_map[p] = data_points[p] 
                elif len(pts_idx) > 1:
                    cs_set.update({cs_tracker: stats}) 
                    for pts in pts_idx:
                        cs_points[pts] = cs_tracker 
                    cs_tracker += 1

        intermediate_results.append([rounds + 1, len(ds_set), len(ds_points), len(cs_set), len(cs_points), len(rs_points_map)])

    elif rounds < len(myFiles):
        # Step 1: check if points belong to DS. Recompute those DS
        data_points, ds_set, ds_points = checkIfPointsInDSorCS(data_points, ds_set, ds_points)
        # Step 2: check if points belong to CS. Recompute those CS
        data_points, cs_set, cs_points = checkIfPointsInDSorCS(data_points, cs_set, cs_points)
        # Step 3: For those points that are not assigned to CS or RS, assign to RS
        if len(data_points) > 0:
            for pt_idx, pt in data_points.items():
                rs_points_map[pt_idx] = pt
				
        # Step 4: Merge the data points in RS by running K-Means with a large number of clusters to generate CS (clusters with more than one points) and RS (clusters with only one point).
        if len(rs_points_map) > 0:
            CS_RS = k_Means(rs_points_map, min(2 * n_cluster, len(rs_points_map)), 4,"CS_RS", [], [])  

            old_cs_set = cs_set.copy()
            old_cs_points = cs_points.copy()
            cs_tracker = len(old_cs_set)
            for cluster_idx in range(len(CS_RS)):
                pts_idx_set, stats = CS_RS[cluster_idx] 
                if len(pts_idx_set) > 1:  # RS
                    for pts in pts_idx_set:
                        del rs_points_map[pts]  
                        cs_points[pts] = cs_tracker 
                    cs_set.update({cs_tracker: stats})
                    cs_tracker += 1

            #new CS into old CS
            if len(cs_set) == 0: 
                cs_set = old_cs_set.copy()
                cs_points = old_cs_points.copy()
            else:
                old_cs_set, old_cs_points, cs_set, cs_points = mergeCSwithDS(old_cs_set, old_cs_points, cs_set, cs_points)
                keyz = [x for x in old_cs_set.keys()]
                if keyz == []:
                    cs_tracker = 0
                else:
                    cs_tracker = max(keyz) + 1
                for new_cs_idx, new_cs_stats in cs_set.items():
                    old_cs_set[cs_tracker] = new_cs_stats
                    for pt_idx, pt_cl in cs_points.items():
                        if pt_cl == new_cs_idx:
                            old_cs_points[pt_idx] = cs_tracker
                    cs_tracker += 1
                cs_set = old_cs_set.copy()
                cs_points = old_cs_points.copy()


        if (rounds == len(myFiles) - 1):
           
            # Merge CS into DS clusters
            ds_set, ds_points, cs_set, cs_points = mergeCSwithDS(ds_set, ds_points, cs_set, cs_points)
            # Leftover RS points and CS clusters
            rs_points_map, ds_set, ds_points = checkIfPointsInDSorCS(rs_points_map, ds_set, ds_points)


        intermediate_results.append([rounds + 1, len(ds_set), len(ds_points), len(cs_set), len(cs_points), len(rs_points_map)])

print()
for ir in intermediate_results:
    print(ir)

# Writing Intermediate data
f = open(out_file2, "w")
f.write("round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained\n")

for r in range(len(intermediate_results)):
    if r != len(intermediate_results) - 1:
        line = ""
        for info in intermediate_results[r]:
            line += str(info) + ","
        f.write(line[:-1] + "\n")
    else:
        line = ""
        for info in intermediate_results[r]:
            line += str(info) + ","
        f.write(line[:-1])
f.close()

# Writing final output

for k, v in ds_points.items():   all_points[k] = v
for k, v in cs_points.items():   all_points[k] = -1
for k, v in rs_points_map.items():   all_points[k] = -1
all_points_list = sorted(list(all_points.items()),key=lambda x: x[0]) 
f = open(out_file1, "w")
line = "{"
for pt_idx, cl in all_points_list:
    line += "\"" + str(pt_idx) + "\": " + str(cl) + ", "
f.write(line[:-2] + "}")
f.close()
end = time.time()
print("\n\nDuration:", end - start)

"""
#CHECKING RMSE
with open("cluster2.json", 'r') as f:  # KINI remove this
    ground = json.loads(f.read())

grounds = [xx[1] for xx in sorted([(int(x[0]), x[1]) for x in ground.items()])]

with open("out_file1", 'r') as f:
    label = json.loads(f.read())

labels = [xx[1] for xx in sorted([(int(x[0]), x[1]) for x in label.items()])]

nmi = normalized_mutual_info_score(grounds, labels)
print("NMI:", nmi)
"""

