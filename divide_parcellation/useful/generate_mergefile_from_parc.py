"""
Generates mergefile from a splifile.

Useful if:
    - The mergefile is not available and it is needed for other subparcellations.
"""
# Author: Susanna Aro <susanna.aro@aalto.fi>

print(__doc__)


subjects_dir = '/neuro/data/susanna/meg/divparc/'
subject_id = 'fsaverage-5.3'

annot_dir = subjects_dir + subject_id + '/label/splitfiles/'

fsdir = subjects_dir + subject_id + '/label/mergefiles/'

hemi = 'rh'
name = 'aparc.a2009s_generated.txt'


"""
Read splitfile. Splitfile has the label name and the number of sublabels on each row.
"""

fname1 = annot_dir + hemi + '.' + name

with open(fname1) as f:
    content1 = f.readlines()

# If the splitfile follows the divide_parcellation.py format, delete the first 3 rows.
content1 = content1[3:]

rows = []
labels = []
for cont in content1:
    merged_labels = cont.split('\t')[0]
    label = merged_labels.split('+')
    label = "\t".join(label)
    rows.append(label + "\n")

"""
Write the mergefile.
"""
"""
fname2 = fsdir + hemi + '.Heschl.txt'
with open(fname2, 'w') as f:
    f.write("\n")
    for row in rows:
        f.write(row)

"""
