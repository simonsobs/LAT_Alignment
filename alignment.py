import adjustments as adj
import mirror_fit as mf

"""
File structure:
    Each measurement session will have its own directory with some useful naming scheme
        Something like YYYYMMDD_num is probably good
    Two options for next layer:
        Option 1: (easier for the code, but may be harder on the faro side)
            Each mirror will have its own subdir
            Within each mirror directory have a file for each panel whose name is the panel name
            File will contain pointcloud of measurements
        Option 2: (harder for the code, but may be easier on the faro side)
            Each mirror has a file containing pointcloud
            Code figures out which measurements are for what panel
        Option 1 is heavily preffered since it levaes less room for erroneous logic to break thinks in some subtle way
    In root of directory we need some sort of config file that tells you what coordinate system the measurements were taken in as well as any adjustments that need to be applied to the model (ie: an origin shift to account for something that is in the wrong place but can't be moved)
        (these could also just be command line arguments)
    Also need to have some sort of lookup table that contains the positions of the alignmnt points and adjustors for each panel (in the mirror coordinates)
    
Workflow:
    Read in config file/parse command line arguments
    If option 1 above:
        Load in measurements for mirror
        Split up into panels
    If option 2:
        Load measurements on a per panel basis
    Transform panel from measurement coordinates to mirror coordinates
    Fit using mf
    Transform adjustor and alignment point locations with fit params
    Fit for adjustments with adj
    Print out adjustments and save to a file in root of measurement dir
"""
