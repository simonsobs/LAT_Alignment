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
