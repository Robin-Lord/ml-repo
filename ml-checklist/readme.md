# This project

This is a planned project to codify and provide guidance for key Machine Learning project steps, it's partially intended to be a learning resource, partially a way to remind practitioners of some simple easy-to-forget steps in the initial data exploration and model testing phases.

This code is not designed for deployment because;
1. By that point the main steps here should have been completed
2. This will add unneeded overhead.

Some good use cases include;
- Junior data scientists or learning ML practitioners starting to get to grips with ML questions through things like Kaggle competitions
- The initial stages of investigating an ML opportunity where it's important to make sure you are going through the validation steps first.


## ML problems covered

Current:
- Category prediction problems

Planned:
- Regression problems
- NLP
- Image processing


## ML stages and tasks covered

### Data ingestion
- Splitting train, test, and (if necessary) validation sets

### Benchmarking
- Performance measurement for the simplest heuristic-based approaches before getting into machine learning

### Initial data analysis
- Highlighting class imbalance

### Data processing
- Ensuring parity in columnar changes for different data sets, while making sure that things like overall average is not leaked into the training set - giving access to pandas dataframe methods at class-level

**Wishlist**
- Highlighting class imbalance



### Model comparison

**Wishlist**
- Generating calibration curves


## Roadmap and wishlist

### Past

--


### In progress
v 0.0.0
- [ ] Loading data
- [ ] Splitting data
- [ ] Simple heuristic

### Future
v 0.1.0
- [ ] Highlighting class imbalance
- [ ] Generating calibration curves
