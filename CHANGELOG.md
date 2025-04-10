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
