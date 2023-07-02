# LAT Alignment
Tools for LAT mirror alignment

## Installation
Technically after cloning this repository you can just run `python lat_alignment/alignment.py PATH/TO/CONFIG`,
but it is recommended that you install this as a package instead.

To do this just run: `pip install -e .` from the root of this repository.

This has two main benefits over running the script directly:
1. It will handle dependencies for you.
2. This sets up an entrypoint called `lat_alignment` so that you can call the code from anywhere.
This is nice because now you can call the code from the measurement directory where you are most likely editing files,
saving you the hassle of having to `cd` or wrangle long file paths.

## Usage
1. Create the appropriate directory structure for your measurement (see [File Structure](#file-structure) for details).
2. Place the measurement files in the appropriate place in your created directory (see [Measurement Files](#measurement-files) for details).
3. Create a file with any information about the measurement that could prove useful (see [Description File](#description-file) for details).
4. Create a config file for your measurement (see [Config File](#config-file) for details).
5. Run the alignment script with `lat_alignment /PATH/TO/CONFIG`
6. Follow the instructions in the output to align panels. This output will both be printed in the terminal and written to an output file (see [Output File](#output-file))

### File Structure
Measurements should be organized in the following file structure
```
measurements
|
└───YYYYMMDD_num
|   |config.txt
|   |description.txt
|   |output.txt
|   |adjusters.yaml
|   |
|   └───M1
|   |   |XX-XXXXXX.txt
|   |   |XX-XXXXXX.txt
|   |   |...
|   |
|   └───M2
|   |   |XX-XXXXXX.txt
|   |   |XX-XXXXXX.txt
|   |   |...
|   |
|   └───plots
|       └───M1
|       |   |XX-XXXXXX_surface.png
|       |   |XX-XXXXXX_hist.png
|       |   |XX-XXXXXX_ps.png
|       |   |...
|       |
|       └───M2
|           |XX-XXXXXX_surface.png
|           |XX-XXXXXX_hist.png
|           |XX-XXXXXX_ps.png
|           |...
|       
└───YYYYMMDD_num
|   |config.txt
|   |description.txt
|   |adjusters.yaml
|   |
|   └───M1
|   |   |XX-XXXXXX.txt
|   |   |XX-XXXXXX.txt
|   |   |...
|   |
|   └───M2
|       |XX-XXXXXX.txt
|       |XX-XXXXXX.txt
|       |...
|   |
|   └───plots
|       └───M1
|       |   |XX-XXXXXX_surface.png
|       |   |XX-XXXXXX_hist.png
|       |   |XX-XXXXXX_ps.png
|       |   |...
|       |
|       └───M2
|           |XX-XXXXXX_surface.png
|           |XX-XXXXXX_hist.png
|           |XX-XXXXXX_ps.png
|           |...
|...
```

#### Measurement Directories
Each directory `YYYYMMDD_num` refers to a specific measurement session. Where `YYYYMMDD` refers to the date of the measurement and `num` refers to which number measurement on that date it was. For example the second measurement taken on January 1st, 2022 would be `20220101_02`.

This is the file path that should be provided to `alignment.py` as the `measurement_dir` argument.

#### Config File
The file `config.yaml` contains configuration options. Below is an annotated example with all possible options.

```yaml
# The measurement directory
# If not provided the dirctory containing the config will be used
measurement_dir: PATH/TO/MEASUREMENT

# The path the the dirctory containing the cannonical adjuster locations
# If not provided the can_points directory in the root of this repository is used
cannonical_points: PATH/TO/CAN/POINTS

# Coordinate system of measurements
# Possible vaules are ["cad", "global", "primary", "secondary"]
coordinates: cad # default value

# Amount to shift the origin of the measurements by
# Should be a 3 element list
origin_shift: [0, 0, 0] # default value

# FARO compensation
compensation: 0.0 # default value

# Set to True to apply common mode subtraction
cm_sub: False # default value

# Set to True to make plots if panels 
plots: False # default value

# Where to save log
# If not provided log is saved to a file called output.txt
# in the measurement_dir for this measurement
log_file: null # Set to null to only print output and not save

# Path to a yaml file with the current adjuster positions
# If null (None) then all adjusters are assumed to be at 0
# You probably want to point this to the file generated
# in the previous alignment run if you have it
adj_path: null # default value

# Path to where to store the adjuster postions after aligning
# If null (None) will store in a file called adjusters.yaml
# in the measurement_dir for this measurement
adj_out: null # default value

# Defines the allowed adjuster range in mm
adj_low: -1 # default value
adj_high: 1 # default value
```

If you are using all default values make a blank config with `touch config.yaml`

#### Description File
Each measurement directory should contain a file `description.txt` with information on the measurement. Any information that could provide useful context when looking at the measurement/alignment after the fact should be included here (ie: who performed the measurement, where the measurement was taken, etc.).

#### Output File
Output generated by `alignment.py`.
By default this is saved at `measurement_dir/output.txt`

Note that this file gets overwritten when `lat_alignment` is run, so if you want to store multiple copies with different configs or something rename them or change the `log_file` in the config.

#### Adjuster Positions
Positions of adjusters after applying the calculated adjustments.
This is a yaml file nominally saved at `measurement_dir/adjusters/yaml`

Each element in the file is in the format:
```
PANEL_NUMBER: [X, Y, ADJ_1, ADJ_2, ADJ_3, ADJ_4, ADJ_5] 
```

#### Mirror Directories
Directories containing the measurements files within each root measurement directory. `M1` contains the measurements for the primary mirror and `M2` contains the measurements for the secondary mirror. If you don't have measurements for one of the mirrors you do not need to create an empty directory for it.

#### Measurement Files
Files containing the point cloud measurements for a given panel. Should live in the mirror directory that the panel belongs to. Files should be named `XX-XXXXXX.txt` where `XX-XXXXXX` is the panel number. The numbering system is as follows:
* First four digits (`XX-XX`) are the telescope number. For the LAT this is `01-01`
* Fifth digit is the mirror number. This is `1` for the primary and `2` for the secondary.
* Sixth digit is the panel row
* Seventh digit is the panel column
* Eight digit is the panel number (current, spare, replacement, etc.)

#### Plot Directory
If the `plots` option is set to `True` then the root measurement will contain a directory called `plots`. Within this directory will be directories for each mirror measured, `M1` for the primary and `M2` for the secondary. Each of these will contain three plots per panel measured:
* `XX-XXXXXX_surface.png`, a plot of the panel's surface in the mirror's coordinate system.
* `XX-XXXXXX_hist.png`, a histogram of the residuals from the panel's fit.
* `XX-XXXXXX_ps.png`, a plot of the power spectrum of the residuals from the panel's fit.

Where `XX-XXXXXX` is the panel number.

## Coordinate Systems
The relevant coordinate systems are marked in the diagram below:

![LAT coordinate systems](./imgs/coords.png)

Where the orange circle marks the `global` coordinate system, the green circle marks the `primary` coordinate system, and the blue circle marks the `secondary` coordinate system.

Additionally there is a `cad` coordinate system that is defined as the coordinate system from the SolidWorks model. It is given by the following transformation from the `global` coordinate system:
```
x -> y - 200 mm
y -> x
z -> -z
```
It is currently unclear why the 200 mm offset exists.

Note that the files in the `can_points` directory are in the `cad` coordinate system.

All measurements should be done in one of these four coordinate systems modulo a known shift in the origin.

## Bugs and Feature Requests
For low priority bugs and feature requests submit an issue on the [git repo](https://github.com/simonsobs/LAT_Alignment).

For higher priority issues (or questions that require an expedient answer) [email](mailto:haridas@sas.upenn.edu), Slack, or call me.

## Contributing
If you wish to contribute to this repository (either code or adding measurement files) contact me via [email](mailto:haridas@sas.upenn.edu) or Slack.

If you are contributing code please do so by creating a branch and submitting a pull request. Try to keep things as close to PEP8 as possible.
