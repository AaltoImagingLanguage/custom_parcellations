"""
Generates splitfile from a parcellation that was created using divide_parcellation.py.

Useful if:
    - The splitfile was accidentally deleted/overwritten and the correct mergefile
    can't be used for one reason or another.
    - We only have the parcellation and need to generate splitfile and/or mergefile
"""
# Author: Susanna Aro <susanna.aro@aalto.fi>

print(__doc__)

import nibabel as nib


hemi = 'rh'

subjects_dir = '/neuro/data/susanna/meg/divparc/'
subject_id = 'fsaverage-5.3'


"""
Read the annot file.
"""

annot_dir = subjects_dir + subject_id + '/label/'
annot_name ='.aparc.a2009s_merged_sub_gyrus_sulcus_PCA_500mm2.annot'

annot = annot_dir + hemi + annot_name

labels, ctab, names = nib.freesurfer.read_annot(annot)


if 'Unknown' in names:
    # divide_parcellation does not split Unknown-label, so discard it.
    names.remove('Unknown')

labels = []
subs = []
i = 0
while i<len(names):

    """
    All the sublabels have string "_sub". By counting them, we have all the info
    for the splitfile
    """
    name = names[i].split('_sub')
    count = 1

    if len(name) == 1:
        # Label wasn't split
        labels.append(name[0])
        subs.append(count)
    else:
        # We have sublabels?
        try:
            # We do have sublabels!
            val = int(name[len(name)-1])

            if len(name) > 2:
                if val < 10:
                    # The name had sub as well! Remove the end from name.
                    labels.append(names[i][0:len(names[i])-5])
                else:
                    labels.append(names[i][0:len(names[i])-6])
            else:
                labels.append(name[0])

            while count + i < len(names) and i < len(names)-1 and names[i+count][0:len(names[i+count])-6] == names[i][0:len(names[i])-6] :
                count+=1
            subs.append(count)

        except ValueError:
            # We just had a label with a word sub in it.
            labels.append(names[i])
            subs.append(count)

    i+=count


"""
Create matching splitfile
"""

fsdir = subjects_dir + 'fsaverage-5.3' + '/label/splitfiles/'

fname = hemi + '.aparc.a2009s_generated.txt'
filename = fsdir + fname

with open(filename, 'w') as f:
    f.write("#Splitfile template.\n#Labels are in order from anterior to posterior.\n\n")
    for i, name in enumerate(labels):
        f.write("%s\t%i\n" %(name, subs[i]))
