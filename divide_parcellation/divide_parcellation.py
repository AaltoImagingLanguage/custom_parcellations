# -*- coding: utf-8 -*-
"""
==============================================================================================
NBE-modification of PCA and region growing algorithms from MNE Python & PySurfer distributions
==============================================================================================

License: BSD (3-clause)

The following citations must always be included when any part of this algorithm is used:

REFERENCES
[1] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, L. Parkkonen, M. Hämäläinen,
    "MNE software for processing MEG and EEG data", NeuroImage, Volume 86, 1 February 2014, Pages 446-460, ISSN 1053-8119.

[2] A. Gramfort, M. Luessi, E. Larson, D. Engemann, D. Strohmeier, C. Brodbeck, R. Goj, M. Jas, T. Brooks, L. Parkkonen, M. Hämäläinen,
    "MEG and EEG data analysis with MNE-Python", Frontiers in Neuroscience, Volume 7, 2013, ISSN 1662-453X.

[3] L. Cammoun, X. Gigandet, D. Meskaldji, J. P. Thiran, O. Sporns, K. Q. Do, ... & P. Hagmann,
    “Mapping the human connectome at multiple scales with diffusion spectrum MRI.” Journal of neuroscience methods, Volume 203.2, 2012. Pages 386-397.



25.6.2015
Original script,
Author: Pekka Laitio <pekka.laitio@aalto.fi>

05.07.2016
Updated script,
Author: Susanna Aro <susanna.aro@aalto.fi>

Key features (see pdf-instructions for further details):
    -Splitfile
    -Mergefile
    -PCA
    -Region growing

"""


import os
from os import path as op
import numpy as np
from scipy import linalg
import csv
import surfer
import mne
from mne.source_space import mesh_dist
from mne.parallel import check_n_jobs
import mne.label as lab
#from matplotlib.cm import get_cmap
#from random import shuffle

from scipy.spatial.distance import cdist
from collections import Counter

logger = surfer.utils.logging.getLogger('surfer')

def hsv_to_rgb(h, s, v):
    """
    Convert hsv values to rgb.
    Taken from:
    http://stackoverflow.com/questions/24852345/hsv-to-rgb-color-conversion, by Tcl
    """
    if s == 0.0: return [v, v, v]
    i = int(h*6.) # XXX assume int() truncates!
    f = (h*6.)-i; p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
    if i == 0: return [v, t, p, 1.0]
    if i == 1: return [q, v, p, 1.0]
    if i == 2: return [p, v, t, 1.0]
    if i == 3: return [p, q, v, 1.0]
    if i == 4: return [t, p, v, 1.0]
    if i == 5: return [v, p, q, 1.0]

def get_unique_colors(n, colormap='Paired'):
    """ Compute n unique colors.
    This is used for generating colors for the merged labels.
    For the split labels, we use mne.label._split_colors for generating the
    unique colors.

    Parameters
    ----------
    n : int
        Number of colors

    Returns
    -------
    colors : array, shape (n, 4)
        RGBA color values.

    """
    HSV_tuples1=[]
    HSV_tuples2=[]
    for i in range(n):
        hue = i*1.0/n
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        if i % 2 == 0:
            HSV_tuples1.append((hue, lightness, saturation))
        else:
            HSV_tuples2.append((hue, lightness, saturation))
    HSV_tuples = HSV_tuples1 + HSV_tuples2
    #HSV_tuples = [(x*1.0/n, 0.5, 0.95) for x in range(n)]
    RGB_tuples = map(lambda x: hsv_to_rgb(*x), HSV_tuples)
    #shuffle(RGB_tuples) # so the similiar colors are not always near each other
    return RGB_tuples

def compute_principal_axis_projection(surface, label):
    """ Project a label on its first PCA component i.e. principal axis.

    Parameters
    ----------
    surface : surf
        Surface
    label : Label
        Label

    Returns
    -------
    proj : array, shape (n, 4)
        Projection of the label on its principal axis.

    """
    # label vertices
    points = surface.coords[label.vertices]
    center = np.mean(points, axis=0)
    centered_points = points - center
    normal = center / linalg.norm(center)
    # project all vertex coordinates on the tangential plane
    q, _ = linalg.qr(normal[:, np.newaxis])
    tangent_u = q[:, 1:]
    m_obs = np.dot(centered_points, tangent_u)
    # compute covariance matrix, eigenvalues (w) and right eigenvectors (vr)
    m_cov = np.dot(m_obs.T, m_obs)
    w, vr = linalg.eig(m_cov)
    # find principal component
    k = np.argmax(w)
    eigendir_max = vr[:, k]
    # project axis back into 3d space and project the label on it
    axis_max = np.dot(tangent_u, eigendir_max)
    proj = np.dot(points, axis_max)
    proj -= proj.min()
    return proj


def compute_absolute_area(surface, verts):
    """ Compute label surface area.

    Parameters
    ----------
    surface : surf
        Surface
    verts : vertices
        Vertices inside label

    Returns
    -------
        Area of label

    UPDATE 04/07/2016:
        Function now calculates the label surface area instead of triangulation.
        -> Label surface area matches the freesurfer label_area result
        -> difference in n_labels between hemis is significantly smaller.

    """
    selection = np.where(np.in1d(surface.faces, verts).reshape(surface.faces.shape) == True)[0]
    triangles = surface.faces[selection]

    """
    CORRECTED AREA CALCULATION
    Label area is the area of the vertices inside it. Area of vertice is area of the triangles it's included in,
    divided by 3.
    """
    label_area = np.sum(np.linalg.norm(np.cross(surface.coords[triangles[:,1],:]-surface.coords[triangles[:,0],:], surface.coords[triangles[:,2]]-surface.coords[triangles[:,0],:]),
                axis=1)/2)/3

    return label_area



def sort_missing_verts(missing_vert, surface, annot_labels, k):
    """ Sorts vertices outside labels based on their k closest neighbors.
    The default k value is 100.

    The code to calculate distance between coordinates and vertices is from
    surfer.utils.find_closest_vertices (Pysurfer, 2011, Neuroimaging in Python Team)

    Manual checking for sorting result is strongly advised.

    Parameters
    ----------
    missing_vert : array
        the unsorted vertices
    surface : surf
        surface
    annot_labels : list
        list of label objects created by split_annotation()
    k: int
        number of neighbors
    Returns
    -------
    Modified annot_labels that include the unsorted vertices
    """

    # Create label array for vertices on surface
    ids = np.zeros(len(surface.coords))
    for i,label in enumerate(annot_labels):
        ids[label.vertices] = i

    ids[missing_vert] = -1
    """
    Calculate k closest vertices for each missing vertex.
    Distance used is euclidean and the code is from Pysurfer distribution.
    """
    missing_label = np.zeros((len(missing_vert)))

    # Calculate neighbors
    point = surface.coords[missing_vert]
    point = np.atleast_2d(point)

    neighbors = np.argsort(cdist(surface.coords, point), axis = 0).T[:,:k].astype(int)

    """
    if everything is working nicely, one loop should do the trick and sort the vertices
    """
    for i in range(len(missing_vert)):
        dist = ids[neighbors[i]]
        dist = dist[dist!=-1]
        if len(dist) == 0:
            missing_label[i] = -1
        else:
            missing_label[i] = Counter(dist).most_common(1)[0][0]



    """
    If the missing vertices are in clusters, we may still have unsorted vertices.
    So we update the label value array and check the neighbors again
    """
    ids[missing_vert] = missing_label


    count = 0
    while np.any(missing_label == -1) and count < 20:
        missing = np.where(missing_label == -1)[0]
        cluster_label = np.zeros(len(missing))
        for i in range(len(missing)):
            dist = ids[neighbors[missing[i]]]
            dist = dist[dist!=-1]
            if len(dist) == 0:
                missing_label[i] = -1
            else:
                cluster_label[i] = Counter(dist).most_common(1)[0][0]


        missing_label[missing] = cluster_label
        ids[missing_vert] = missing_label
        count += 1

    if count == 20:
        # We still didn't manage to sort all the missing vertices.
        print "Not all vertices could be sorted. Placing them to 'Unknown' label"

    """
    Update the labels accordingly
    """

    labels_to_change = list(Counter(missing_label))
    for i in labels_to_change:
        if int(i) == -1:
            unknown_exists = False
            for label in annot_labels:
                if label.name.lower() == 'unknown':
                    vert = label.vertices
                    annot_labels.remove(label)
                    vert_unknown = np.sort(np.append(vert, missing_vert[np.where(missing_label == i)[0]]))
                    pos = surface.coords[vert_unknown]
                    annot_labels.append(mne.label.Label(vert_unknown, pos, hemi=label.hemi, name="Unknown", subject=label.subject))
                    unknown_exists = True
                    break
            if not unknown_exists:
                vert = missing_vert[np.where(missing_label == i)[0]]
                pos = surface.coords[vert]
                annot_labels.append(mne.label.Label(vert, pos, hemi=label.hemi, name="Unknown", subject=label.subject))
        else:
            label = annot_labels[int(i)]
            verts = np.sort(np.append(label.vertices, missing_vert[np.where(missing_label == i)[0]]))
            pos = surface.coords[verts]
            annot_labels[int(i)] = mne.label.Label(verts, pos, hemi=label.hemi, name=label.name, subject=label.subject)

    return annot_labels




def grow_labels(subject, geo, current_label, seeds, size, circle, hemi,
                            subjects_dir=None, n_jobs=1, names=None, curv=False):
    """ Generate labels in source space with region growing

    This function generates a number of non-overlapping labels in source space
    by growing regions starting from the vertices defined in "seeds".

    Parameters
    ----------
    subject : string
        Name of the subject as in SUBJECTS_DIR.
    geo : surf
        Surface of interest.
    current_label : Label
        Label to parcellate.
    seeds : int | list
        Seed, or list of seeds. Each seed can be either a vertex number or
        a list of vertex numbers.
    size : float | False
        Maximum sublabel size (mm2). If False, the maximum size is label size
        divided by the number of seeds.
    circle : float | False
        Defines maximum label radius by equation circle = pi*r^2.
        [NOT WORKING CORRECTLY 25.6.2015]
    hemi : 'lh' | 'rh'
        Hemisphere to use
    subjects_dir : string
        Path to SUBJECTS_DIR if not set in the environment.
    n_jobs : int
        Number of jobs to run in parallel. Likely only useful if tens
        or hundreds of labels are being expanded simultaneously.
    names : None | list of str
        Assign names to the new labels (list needs to have the same length as
        seeds).
    curv : bool
        If True, gyrus-sulcus-curvature information is taken into account.
        [NOT WORKING CORRECTLY 25.6.2015]

    Returns
    -------
    labels : list of Label
        Labels computed with region growing.

    """

    n_jobs = check_n_jobs(n_jobs)

    # make sure the inputs are arrays
    if np.isscalar(seeds):
        seeds = [seeds]
    seeds = np.atleast_1d([np.atleast_1d(seed) for seed in seeds])
    n_seeds = len(seeds)
    if np.isscalar(names):
        names = [names]
    if len(names) != n_seeds:
        raise ValueError('The names parameter has to be of length len(seeds)')
    names = np.array(names)

    # load surfaces and create distance graphs
    vertices = geo.coords
    triangles = geo.faces
    graph = mesh_dist(triangles, vertices)

    labels = []
    label_area = compute_absolute_area(geo, current_label.vertices)
    # Compute number of ROIs
    if size:
        N_ROIs = np.around(label_area/size)
        if N_ROIs < 1:
            N_ROIs = 1 # at least one label
    else:
        N_ROIs = len(seeds)
    N_vertices_max = len(current_label.vertices)/N_ROIs # maximum number of vertices

    # prepare parcellation
    parc = np.empty(len(vertices), dtype='int32')
    parc[:] = -1
    # initialize active sources
    sources = {}  # vert -> (label, dist_from_seed)
    edge = []  # queue of vertices to process
    for label, seed in enumerate(seeds):
        if np.any(parc[seed] >= 0):
            raise ValueError("Overlapping seeds")
        parc[seed] = label
        for s in np.atleast_1d(seed):
            sources[s] = (label, 0.)
            edge.append(s)

    # grow from sources
    while edge:
        vert_from = edge.pop(0)
        label, old_dist = sources[vert_from]
        # add neighbors within allowable distance/conditions
        row = graph[vert_from, :]
        for vert_to, dist in zip(row.indices, row.data):
            new_dist = old_dist + dist

            """
            Tip: there are multiple abort-conditions listed below and you
                        can edit or remove them freely or make a new one! :)
            """
            # (1) abort if 'vert_to' is outside of label
            if not np.any(current_label.vertices == vert_to):
                continue

            # (2) abort if 'vert_to' is outside of 'circle' NOT WORKING
            if circle:
                if new_dist > np.sqrt(circle/np.pi):
                    continue

            # (3) abort if parcel area exceeds.
            # Notice: size delimeter can be ignored by setting 'size=False'
            if size:
                N_vertices = len(np.nonzero(parc == label)[0])
                if N_vertices > N_vertices_max-1:
                        continue

            # (4) abort if 'curv=True' and region growing crosses a boundary NOT WORKING
            if curv:
                if geo.bin_curv[vert_to] != geo.bin_curv[vert_from]:
                    continue

            vert_to_label = parc[vert_to]
            if vert_to_label >= 0:
                _, vert_to_dist = sources[vert_to]
                # (5) abort if the vertex is occupied by a closer seed
                if new_dist > vert_to_dist:
                    continue
                elif vert_to in edge:
                    edge.remove(vert_to)


            # assign label value
            parc[vert_to] = label
            sources[vert_to] = (label, new_dist)
            edge.append(vert_to) # Notice: algorithm stops when 'edge' is empty


    # convert parc to labels
    for i in xrange(len(seeds)):
        vertices = np.nonzero(parc == i)[0]
        pos = geo.coords[vertices]
        name = str(names[i])
        labels.append(mne.label.Label(vertices, pos, hemi=hemi, name=name, subject=subject))

    # add a unique color to each label
    colors = get_unique_colors(len(labels))
    for label, color in zip(labels, colors):
        label.color = color

    return labels




"""
------------
MERGE LABELS
------------
"""
def merge_labels(subjects_dir, subject_id, annotation,
                                hemi, map_surface, mergefile, split_name = None, ROI_size=False, annot_dir=None, write_dir=None):
    """ Creates a new annotation file with merged labels. Return nothing.

        Notice: Creates a splitfile template compatible
        with 'split_labels'-function. The path is label/splitfiles/.

    Parameters
    ----------
    subjects_dir : string
        Subject's directory
    subject_id : string
        Use if file is in register with subject's orig.mgz
    annotation: annot
        Annotation file
    hemi : [lh, rh]
        Hemisphere
    map_surface : string
        Surface name (e.g. 'white', 'pial' etc.)
    mergefile : .txt
        Textfile including labelnames separated with tabs,
        each row representing a group to merge.
    ROI_size : float
        Target size of subdivided ROIs. Used in
        creating splitfile template. If 'False', number of sub-ROIs
        is set to default value 1 for each label.
    annot_dir : string | None
        Annotation directory. Use only if annotation file is in another directory than subjects_dir.
        If None, subjects_dir parameter is used instead to find the annotation file.
    write_dir : string | None
        Writing path for the results including annotations, labels and splitfiles.
        If None, the results will be written to default label folder in subjects_dir.

    ADDED 04/07/2016
    split_name: string | None
        choose the end part for the splitfile. Otherwise use Pekka's default.

    Returns
    -------
        Nothing.

    """

    logger.info(" Starting 'merge_labels'..")

    # load surface geometry & curvature
    geo = surfer.utils.Surface(subject_id, hemi, map_surface, subjects_dir=subjects_dir)
    geo.load_geometry()
    geo.load_curvature()

    # read labels
    if annot_dir==None:
        labels = mne.read_labels_from_annot(subject_id, parc=annotation, hemi=hemi, surf_name=map_surface, subjects_dir=subjects_dir)
    else:
        labels = mne.read_labels_from_annot(subject_id, surf_name=map_surface, annot_fname=annot_dir+'/'+hemi+'.'+annotation+'.annot', subjects_dir=subjects_dir)
    # delete '-lh' or '-rh' extension
    for label in labels:
        if label.name[-3:]=='-lh' or label.name[-3:]=='-rh':
            label.name = label.name.replace(' ', '')[:-3]

    # read mergefile
    if not op.isfile(mergefile):
        raise ValueError('Mergefile does not exist.')
    with open(mergefile) as f:
        csv_reader = csv.reader(f, delimiter='\t')
        merge_list = list(csv_reader)

    # append merged labels
    for row in merge_list:
        if len(row) and row[0][0]!='#':
            vert = []
            pos = []
            lbl_name = ''
            for name in row:
                if name!='':
                    idx = np.nonzero([label.name==name for label in labels])[0][0]
                    vert.extend(labels[idx].vertices)
                    pos.extend(labels[idx].pos)
                    lbl_name += name + '+'
                    labels.pop(idx)
            # re-order vertices and coordinates
            vert = [v for (v,p) in sorted(zip(vert,pos))]
            pos = [p for (v,p) in sorted(zip(vert,pos))]
            # delete '+' from the end of the name
            lbl_name = lbl_name.replace(' ', '')[:-1]
            labels.append(mne.label.Label(vert, pos, hemi=hemi, name=lbl_name,
                                    subject=subject_id))

    if write_dir == None:
        directory = op.join(subjects_dir, subject_id, 'label', annotation + '_merged')
    else:
        directory = op.join(write_dir, annotation + '_merged')
    if not os.path.exists(directory):
        os.makedirs(directory)
    # write labels
    for label in labels:
        filename = op.join(directory, hemi + '.' + label.name + '.label')
        with open(filename, 'w') as f:
            n_vertices = len(label.vertices)
            data = np.zeros((n_vertices, 5), dtype=np.float)
            data[:, 0] = label.vertices
            data[:, 1:4] = label.pos
            data[:, 4] = label.values
            f.write("#%s\n" % label.comment)
            f.write("%d\n" % n_vertices)
            for d in data:
                f.write("%d %f %f %f %f\n" % tuple(d))


    # Write splitfile template with labels in order from anterior to posterior.
    if write_dir == None:
        splitfile_dir = op.join(subjects_dir, subject_id, 'label', 'splitfiles')
    else:
        splitfile_dir = op.join(write_dir, 'splitfiles')
    if not os.path.exists(splitfile_dir):
        os.makedirs(splitfile_dir)
    filename = op.join(splitfile_dir,
                    hemi+'.'+annotation+'_merged'+'-splitfile_' + split_name +'.txt')
    default = 1 # default value for number of ROIs if 'ROI_size=False'
    # sort according to y-position from anterior to posterior
    y_positions = [np.mean(label.pos[:,1]) for label in labels]
    sorted_labels = [label for (y,label) in sorted(zip(y_positions,labels))][::-1]
    with open(filename, 'w') as f:
        f.write("#Splitfile template.\n#Labels are in order from anterior to posterior.\n\n")
        for label in sorted_labels:
            if ROI_size:
                N = compute_absolute_area(geo, label.vertices)/ROI_size
                if np.around(N) < 1:
                    N_ROIs = 1
                else:
                    N_ROIs = np.around(N)
                f.write("%s\t%i\n" %(label.name, N_ROIs))
            else:
                f.write("%s\t%i\n" %(label.name, default))


    # define unique colors for annotation file

    colors = get_unique_colors(len(labels))
    for label, color in zip(labels, colors):
        label.color = color
    # write annotation file

    # We need to add ending zero to label names for freesurfer compatibility.

    for label in labels:
        label.name = label.name+'\0'

    if write_dir == None:
        annot_fname= op.join(subjects_dir, subject_id, 'label', hemi+'.'+annotation+'_merged'+'.annot')
    else:
        annot_fname= op.join(write_dir, hemi+'.'+annotation+'_merged'+'.annot')
    mne.write_labels_to_annot(labels=labels, subject=subject_id, overwrite=True, subjects_dir=subjects_dir, annot_fname=annot_fname, verbose=None, colormap='Paired')

    print 'Done with mergefile.\n'



"""
--------------
READ SPLITFILE
--------------
"""
def read_splitfile(ROI_name, splitfile):
    """ Returns number of split points corresponding to ROI-name.
        If no corresponding ROI is found, the default return value is 1.

    Parameters
    ----------
    ROI_name : string
        ROI name
    splitfile : .txt
        Textfile including ROIs with corresponding number of split points

    Returns
    -------
    n : int
        Number of ROIs corresponding 'ROI_name' in the splitfile. If 'ROI_name' is
        not found, the default output is 1.
    """

    if not op.isfile(splitfile):
        raise ValueError('Splitfile does not exist.')

    with open(splitfile) as f:
        csv_reader = csv.reader(f, delimiter='\t')
        search_list = list(csv_reader)

    for row in search_list:
        if len(row):
            if row[0] == ROI_name:
                return int(row[-1])

    print ("Number of ROIs for %s could not be detected. Returning default value 1 instead.\n" %ROI_name)
    return 1










"""
-----------------
SPLIT ANNOTATION
-----------------
"""
@surfer.utils.verbose
def split_annotation(subjects_dir, subject_id, annotation, hemi, ROI_size=100, x_limits=[0,1], y_limits=[0,1], z_limits=[0,1],
                    map_surface='inflated', verbose=None, splitfile=False, annot_name = None, method='PCA', annot_dir=None, write_dir=None,
                    sort_verts=100):
    """Create subdivided annotation file and corresponding label files.

    Parameters
    ----------
    subjects_dir : string
        Subject's directory. Recommended to set $SUBJECTS_DIR.
    subject_id : string
        Subject's name
    annotation : string
        Annotation name without '.annot' extension
    hemi : 'lh' | 'rh'
        Hemisphere
    ROI_size : float
        Target size for divided labels (mm2)
    x_limits : [float, float]
        List of two floats (increasing) between 0...1 representing relative
        annotation x-limits within the computing is done (from right to left).
    y_limits : [float, float]
        Relative y-limits (from anterior to posterior).
    z_limits : [float, float]
        Relative z-limits (from superior to inferior).
    map_surface : string
        Surface name
    verbose : bool, str, int, or None
        If not None, override default verbose level (see surfer.verbose).
    splitfile : string | False
    Splitfile to be applied, if not 'False'
    method : 'PCA' | 'RG'
    Algorithm method used in the label division before smoothing.
    annot_dir : string | None
        Annotation directory. Use only if annotation file is in another directory than subjects_dir.
        If None, subjects_dir parameter is used instead to find the annotation file.
    write_dir : string | None
        Writing path for the results including annotations, labels and splitfiles.
        If None, the results will be written to default label folder in subjects_dir.


    ADDED 06/07/2016:
    annot_name : string | None
        Customize the end part for the annotation file to be created.
    sort_verts: int | False
        How many neighbors to check for sorting missing vertices. If False, they are added to "Unknown" label. Default is 100.



    Returns
    -------
    missing_vert: numpy array
        vertices that were not included in labels after steps 1 and 2.
    """

    # number of parallel jobs
    n_jobs = 1

    # load surface geometry & curvature
    geo = surfer.utils.Surface(subject_id, hemi, map_surface, subjects_dir=subjects_dir)
    geo.load_geometry()
    geo.load_curvature()

    # read labels
    if annot_dir==None:
        labels = mne.read_labels_from_annot(subject_id, parc=annotation, hemi=hemi, surf_name=map_surface, subjects_dir=subjects_dir)
    else:
        labels = mne.read_labels_from_annot(subject_id, surf_name=map_surface, annot_fname=annot_dir+'/'+hemi+'.'+annotation+'.annot', subjects_dir=subjects_dir)

    # discard labels that are not between x-, y- or z-limits
    x_positions = [np.mean(label.pos[:,0]) for label in labels]
    labels = [label for (x,label) in sorted(zip(x_positions,labels))][::-1]
    labels = labels[int(len(labels) * x_limits[0]) : int(len(labels) * x_limits[1])]
    y_positions = [np.mean(label.pos[:,1]) for label in labels]
    labels = [label for (y,label) in sorted(zip(y_positions,labels))][::-1]
    labels = labels[int(len(labels) * y_limits[0]) : int(len(labels) * y_limits[1])]
    z_positions = [np.mean(label.pos[:,2]) for label in labels]
    labels = [label for (z,label) in sorted(zip(z_positions,labels))][::-1]
    labels = labels[int(len(labels) * z_limits[0]) : int(len(labels) * z_limits[1])]


    annot_labels = []
    vert_labels = np.array([], dtype=int)
    div = False # bool: True if N_ROIs > 1 and thus label will be divided
    original_colors = []
    for label in labels:
        # delete '-lh' or '-rh' extension
        if label.name[-3:]=='-lh' or label.name[-3:]=='-rh':
            label.name = label.name.replace(' ', '')[:-3]
        # compute label area and corresponding number of ROIs
        label_area = compute_absolute_area(geo, label.vertices)
        if splitfile:
            N_ROIs = read_splitfile(label.name, splitfile)
            ROI_size = label_area/N_ROIs
        else:
            N_ROIs = np.around(label_area/ROI_size)
            if N_ROIs < 1:
                N_ROIs = 1

        if N_ROIs > 1:
            div = True
        else:
            div = False

        original_colors.append((label.color, N_ROIs))
        if label.name == 'Unknown' or label.name == 'unknown':
            output_labels = [label]
            print '%s --SKIPPED--\n' %(label.name)
        else:
            """
            STEP 1: PCA or region growing (depends on the 'method' parameter).
            """
            # ALTERNATIVE 1: PCA
            if method=='PCA':
                proj_o = compute_principal_axis_projection(geo, label)

                # In order to produce labels of similar size, each sub-label's area rather
                # than projection length must be approximated along the principal axis.
                # Areas can be approximated directly from the number of vertices.
                K = 20 # number of intervals
                counter = 0 # counter for labels

                while K < 45 and counter != N_ROIs:
                    counter = 0
                    proj = proj_o / (proj_o.max() / K)
                    proj = proj // 1 # floor division
                    N_VERT = len(label.vertices)
                    idx = (proj < 0) # all indices are initialized 'False'
                    p = 0 # percentage initialized to zero
                    counter = 0 # counter for labels
                    new_labels = []
                    p_max = 1.0/float(N_ROIs) # maximum percentage of vertices to each sublabel
                    for j in range(K):
                        idx += (proj == j)
                        p = float(len(label.vertices[idx]))/float(N_VERT) # current percentage of vertices
                        if p>=p_max or j==K-1:
                            vert = label.vertices[idx]
                            pos = label.pos[idx]
                            values = label.values[idx]
                            hemi = label.hemi
                            comment = label.comment
                            name = label.name + '_PCA' + str(counter)
                            new_labels.append(mne.label.Label(vert, pos, values, hemi, comment, name, None, subject_id))
                            idx = (proj < 0) # all indices are set to 'False'
                            p = 0
                            counter += 1
                    K+=5

            # END OF IF-STATEMENT


            # ALTERNATIVE 2: Region growing
            elif method=='RG':
                # Set mean vertex of the first imaginary PCA-label the first seed point for region growing.
                # Notice: PCA is not used in subdivision itself but only to find the first seed point.
                proj = compute_principal_axis_projection(geo, label)
                K = 20 # number of intervals
                proj /= (proj.max() / K)
                proj = proj // 1 # floor division
                N_VERT = len(label.vertices)
                idx = (proj < 0) # all indices are initialized 'False'
                p = 0 # percentage initialized to zero
                k = 0 # index counter
                p_max = 1.0/float(N_ROIs) # maximum percentage of vertices to each sublabel
                seeds = []
                for j in range(K):
                    idx += (proj == j)
                    p = float(len(label.vertices[idx]))/float(N_VERT) # current percentage of vertices
                    if p>=p_max or j==K-1:
                        vert = label.vertices[idx]
                        index = surfer.utils.find_closest_vertices(geo.coords[label.vertices], np.mean(geo.coords[vert], axis=0))
                        v = label.vertices[index]
                        seeds.append(v)
                        idx = (proj < 0) # all indices are set to 'False'
                        p = 0
                        k += 1

                new_labels = []
                n_vertices_settled = 0
                areaFull = False
                label_ = label
                v = seeds[0] # first seed vertex
                idx = 0
                # grow label-by-label until area is 95 % covered
                while not areaFull:
                    index = surfer.utils.find_closest_vertices(geo.coords[label_.vertices], geo.coords[v])
                    seed = label_.vertices[index]
                    name = label.name + '_RG' + str(idx)
                    # Compute region growing from 'seed'. ROI-size is limited to target size (absolute).
                    temp_label_list = grow_labels(subject_id, geo, label_, seed, size=ROI_size,
                                                        circle=False, hemi=hemi, subjects_dir=subjects_dir,
                                                            n_jobs=n_jobs, names=name, curv=False)
                    new_labels.append(temp_label_list[0]) # Notice: 'temp_label_list' includes only one label
                    # update number of settled vertices
                    n_vertices_settled += len(new_labels[idx].vertices)
                    print 'Settled: %i' %n_vertices_settled
                    # Update remaining label area i.e. the next splittable area. (All positions are initialized zeros.)
                    mask = np.in1d(label_.vertices, new_labels[idx].vertices, assume_unique=True, invert=True)
                    label_ = mne.label.Label(label_.vertices[mask], hemi=hemi,
                                        name='temporary_label', subject=subject_id)
                    # find nearest vertex from the center of previously separated sub-ROI
                    if len(label_.vertices) > 0:
                        mean = np.mean(geo.coords[new_labels[idx].vertices], axis=0)
                        index = surfer.utils.find_closest_vertices(geo.coords[label_.vertices], mean)
                        v = label_.vertices[index]
                    # check area coverage proportionally to vertices
                    if n_vertices_settled > 0.95*len(label.vertices):
                        areaFull = True
                    # update index
                    idx += 1

            # END OF ELIF-STATEMENT

            else:
                raise ValueError('The method parameter must be "PCA" or "RG"')





            """
            STEP 2: Smoothing of the computed labels
            """
            # Select 'N_ROIs' number of biggest labels and repeat the region growing now simultaneously:
            print 'Number of labels : %i @ %s \n' %(N_ROIs, label.name)

            # compute absolute areas
            areas = []
            for j in range(len(new_labels)):
                areas.append(compute_absolute_area(geo, new_labels[j].vertices))
            # Sort new_labels in descending order according to areas (why needed?)
            new_labels = [new_label for (area, new_label) in sorted(zip(areas,new_labels))][::-1]

            names = []
            seeds = []
            # set seed points
            for new_label in new_labels:
                idx = new_labels.index(new_label)
                if idx > N_ROIs-1:
                    continue
                new_label_mean = np.mean(geo.coords[new_label.vertices], axis=0)
                if div:
                    names.append(label.name + '_sub' + str(idx+1))
                else:
                    names.append(label.name)
                # Notice indexing on the next two lines!
                index = surfer.utils.find_closest_vertices(geo.coords[label.vertices], new_label_mean)
                seeds.append(label.vertices[index])
            # sort seeds with respect to y-position from anterior to posterior
            y_positions = [geo.coords[seed][0,1] for seed in seeds]
            seeds = [seed for (y,seed) in sorted(zip(y_positions,seeds))][::-1]
            output_labels = grow_labels(subject=subject_id, geo=geo,
                                        current_label=label, seeds=seeds, size=False, circle=False, hemi=hemi,
                                        subjects_dir=subjects_dir, n_jobs=n_jobs, names=names, curv=False) # Notice: size=False

        """
        Append computed 'output_labels' to final 'annot_labels'.
        Name 'output_labels' refers to the output of the division algorithm above.
        """
        for output_label in output_labels:
            vert_labels = np.append(vert_labels, np.array(output_label.vertices))
            annot_labels.append(output_label)



    """
    If there are still vertices (along the border of unknown label) belonging to no label, add those vertices to "Unknown" label.
    """

    vert_total = np.arange(len(geo.coords))
    mask = np.in1d(vert_total, vert_labels, invert=True)
    missing_vert = np.sort(vert_total[mask])

    if sort_verts:
        annot_labels = sort_missing_verts(missing_vert, geo, annot_labels, sort_verts)
    else:
        unknown_exists = False
        for annot_label in annot_labels:
           if annot_label.name == 'Unknown' or annot_label.name == 'unknown':
                vert = annot_label.vertices
                annot_labels.remove(annot_label)
                vert_unknown = np.sort(np.append(vert, missing_vert))
                pos = geo.coords[vert_unknown]
                annot_labels.append(mne.label.Label(vert_unknown, pos, hemi=hemi, name=annot_label.name, subject=subject_id))
                unknown_exists = True
                break
        if not unknown_exists:
            pos = geo.coords[missing_vert]
            annot_labels.append(mne.label.Label(missing_vert, pos, hemi=hemi, name="Testi", subject=subject_id))

    """
    Write files
    """
    # write .label files
    if write_dir == None:
        directory = op.join(subjects_dir, subject_id, 'label', annotation + '_' + annot_name)
    else:
        directory = op.join(write_dir, annotation + annot_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for annot_label in annot_labels:
        filename = op.join(directory, hemi + '.' + annot_label.name + '.label')
        with open(filename, 'w') as f:
            n_vertices = len(annot_label.vertices)
            data = np.zeros((n_vertices, 5), dtype=np.float)
            data[:, 0] = annot_label.vertices
            data[:, 1:4] = annot_label.pos
            data[:, 4] = annot_label.values
            f.write("#%s\n" % annot_label.comment)
            f.write("%d\n" % n_vertices)
            for d in data:
                f.write("%d %f %f %f %f\n" % tuple(d))

    # define unique colors for annotation file
    colors = []
    for (color, N_colors) in original_colors:
        new_colors = lab._split_colors(color, N_colors)
        colors += new_colors
    """
    colors = get_unique_colors(len(annot_labels))
    """
    for label, color in zip(annot_labels, colors):
        label.color = color

    # write .annot file

    # add the terminating zero byte

    for label in annot_labels:
        label.name = label.name+'\0'

    if write_dir == None and annot_name == None:
        annot_fname= op.join(subjects_dir, subject_id, 'label', hemi+'.'+annotation+'_subdivision'+'.annot')
    else:
        if write_dir == None:
             annot_fname= op.join(subjects_dir, subject_id, 'label', hemi+'.'+annotation+'_sub_'+ annot_name + '.annot')
        elif annot_name == None:
            annot_fname= op.join(write_dir, hemi+'.'+annotation+'_subdivision'+'.annot')
        else:
            annot_fname= op.join(write_dir, hemi+'.'+annotation+'_sub_'+ annot_name + '.annot')
    print annot_fname
    mne.write_labels_to_annot(labels=annot_labels, subject=subject_id, overwrite=True, subjects_dir=subjects_dir, annot_fname=annot_fname, verbose=None)


    return missing_vert


    print '\nAll done. Total number of labels for %s is:\n%i\n' %(annot_fname.split("/")[-1], len(annot_labels))

