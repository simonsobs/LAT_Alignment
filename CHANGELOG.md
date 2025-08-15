## 7.3.0 (2025-08-15)

### Feat

- add animations of residuals

## 7.2.0 (2025-08-15)

### Feat

- add plot of difference between points on the same element

## 7.1.0 (2025-08-15)

### Feat

- allow tracking trajectory of non ref points

### Fix

- move trajectory error legend to scatter plot

## 7.0.0 (2025-08-14)

### Feat

- add support for tracker data in main alignment and make sure method=mean is set for trajectory analysis
- add dataset for handling reference points
- move label information to the reference struct
- move dataset to a base class and reorganize some modules

## 6.3.1 (2025-08-13)

### Fix

- missing import

## 6.3.0 (2025-08-03)

### Feat

- add option to save reconstructed TODs
- add option to plot transforms in local coordindates
- reorganize trajectory data into dataclasses

## 6.2.0 (2025-07-31)

### Feat

- add plots for residual and scale factors

### Fix

- include skspatial in deps

## 6.1.0 (2025-07-30)

### Feat

- add csv of measured surface to output

### Fix

- add cylinder_fitting to requirements

## 6.0.0 (2025-07-21)

### Feat

- switch to 64 bit
- add tod padding and mark padded points in plots
- add angle correction
- add loading of stepped data
- add thresh to help with floating point errors
- plot all points for a mirror on the same plot
- add entrypoint script to analyze motions
- add function to load tracker data from a txt file
- genralize hwfe script to include pointing error and rename

### Fix

- consider sign when applying thresh
- only branch angle for mirrors
- remove unused imports
- cast to float64 during basis transform
- put angles into global basis before adding
- don't include coordinate transform in affine basis shift

## 5.6.0 (2025-06-20)

### Feat

- add script and supporting code for computing hwfe from tracker data

### Fix

- remove unused import

## 5.5.0 (2025-06-18)

### Feat

- add ability to use fixed codes to align and try cutting points when rms is high

## 5.4.0 (2025-04-10)

### Feat

- much better handling of the CCW adjustemnts, ensure that we dont overwrite fields in ways that cause the adjustments to latch when in source batch mode, add templane files to data directory, output csvs of adjustments, readd argument that went missing

### Fix

- slight numerical fix
- compose in the correct direction
- fix sign flip on redidual correction

## 5.3.0 (2025-03-07)

### Feat

- add bootstrapping from another optical element

## 5.2.0 (2025-03-05)

### Feat

- add ability to kill points on mirror not near an adjustor and fix plotting with sparse measurements
- print all adjustors to hit

### Fix

- better error handling and setup for future bootstrapping feature
- mask out points too far from mirror surface

### Refactor

- split mirror into two parts and fix sign on thread direction

## 5.1.0 (2025-01-24)

### Feat

- add entrypoint script that uploads a result to the ixb tool
- add tools for connecting too and initializing ixb tool

### Fix

- output all adjustments

## 5.0.1 (2025-01-09)

### Fix

- flip signs of adjustments for primary to make things more intuitive

## 5.0.0 (2025-01-09)

### Feat

- make distance to count as a double tunable
- more robust vector alignment
- add init alignment to all elements and include an option to not scale during alignment
- first pass at adding the bearing and receiver also some fixes to coord transforms and better plotting

### Fix

- sign flip
- tranform bearing points before returning

### Refactor

- Move to a more flexible interface for the photogrammetry data

## 4.3.1 (2024-11-19)

### Fix

- swap remaining prints to logger

## 4.3.0 (2024-11-19)

### Feat

- switch to logger

## 4.2.0 (2024-11-19)

### Feat

- move reference points to data file

## 4.1.0 (2024-11-18)

### Feat

- first pass at optical element alignment

## 4.0.1 (2024-11-16)

### Fix

- remove mask used for debugging
- fix bug where dy was being returned for dx

## 4.0.0 (2024-11-14)

### Feat

- compute overall common mode and add verbosity flag
- better plotting tools, improved transforms, adjuster residuals, and common mode fit

### Fix

- remove doubles from primary datasets
- better outlier rejection
- Don't look at Z for doubles finding
- use mean when computing panel shifts
- apply correct shift for adjusters

### Refactor

- version 2, no focused on photogrammetry, also much more streamlined code
