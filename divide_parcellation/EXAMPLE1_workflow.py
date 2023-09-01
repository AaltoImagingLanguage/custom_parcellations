"""
=======================================
Generate Surface Labels Subparcellation
=======================================

- method: 'PCA'=PCA, 'RG'=region growing
-for fast computing choose method='PCA'

"""

# Author: Pekka Laitio <pekka.laitio@aalto.fi>
# Edits: Susanna Aro <susanna.aro@aalto.fi>

print(__doc__)

import os
import nibabel as nib
from surfer import Brain
import divide_parcellation
import numpy as np

# paths
subject_id = 'fsaverage-5.3'
subjects_dir = os.environ['SUBJECTS_DIR']

hemis = ['lh', 'rh']


map_surface='inflated' # Must be inflated for PCA!!
ROI_size = 500 # Choose the target surface area for ROIs
method = 'PCA'

missing_verts = []

for hemi in hemis:

    # Merge labels. Tip: an empty mergefile can also be used in order to get a 'splitfile'.

    mergefile = os.path.join(subjects_dir, subject_id, 'label', 'mergefiles', '%s.empty.txt' % hemi)

    # This is where you can specify the end part for the splitfile name, for example:
    split_name = method + '_'  + str(ROI_size) + 'mm2'

    divide_parcellation.merge_labels(subjects_dir, subject_id, annotation = 'aparc.a2009s', hemi=hemi,
                        map_surface=map_surface, mergefile=mergefile, split_name=split_name, ROI_size=ROI_size)#, annot_dir=annot_dir1, write_dir=write_path)


    # Split annotation.
    splits = hemi +'.aparc.a2009s_merged-splitfile_' + split_name + '.txt'
    splitfile = os.path.join(subjects_dir, subject_id, 'label', 'splitfiles', splits)
    name = split_name

    missing_vert = divide_parcellation.split_annotation(subjects_dir, subject_id, annotation = 'aparc.a2009s_merged',
                        hemi=hemi, ROI_size=ROI_size, x_limits=[0, 1], y_limits=[0, 1], z_limits=[0, 1],
                        map_surface=map_surface, splitfile=splitfile, annot_name = name, method=method, sort_verts = 100)#, annot_dir=annot_dir2, write_dir=write_path)

    missing_verts.append(missing_vert)


"""
Visualisation for quality check.
"""


brain = Brain(subject_id, "split", "inflated", views=['lat', 'med'],
            config_opts=dict(background="black", size=1200))

annot = 'aparc.a2009s_merged_sub_' + split_name

brain.add_annotation(annot, borders=2); brain.toggle_toolbars()

"""
Check the location of the missing verts and the labels they were sorted into.
"""

annot = 'aparc.a2009s_merged_sub_' + split_name
annot_path = subjects_dir +'/'+ subject_id + '/label/lh.' + annot + '.annot'
ids, ctab, names = nib.freesurfer.read_annot(annot_path)

roi_data = np.zeros(len(names))
for i in range(len(missing_verts[0])):
    roi_data[int(ids[missing_verts[0][i]])] = 1


vtx_data = roi_data[ids]

scale_factor = 0.3

brain = Brain(subject_id, "lh", "inflated")

brain.add_foci(missing_verts[0], coords_as_verts=True,
               scale_factor=scale_factor, color="#A52A2A")

brain.add_data(vtx_data, 0, 1, colormap="GnBu", alpha=.8)
brain.add_annotation(annot, borders=2); brain.toggle_toolbars()


"""
annot_path = subjects_dir +'/'+ subject_id + '/label/rh.' + annot + '.annot'
ids, ctab, names = nib.freesurfer.read_annot(annot_path)

roi_data = np.zeros(len(names))
for i in range(len(missing_verts[1])):
    roi_data[int(ids[missing_verts[1][i]])] = 1

vtx_data = roi_data[ids]
brain = Brain(subject_id, "rh", "inflated")
brain.add_foci(missing_verts[1], coords_as_verts=True,
               scale_factor=scale_factor, color="#A52A2A")

brain.add_data(vtx_data, 0, 1, colormap="GnBu", alpha=.8)
brain.add_annotation(annot, borders=2); brain.toggle_toolbars()

"""
