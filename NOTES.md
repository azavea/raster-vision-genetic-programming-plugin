TODO:
- Run on AWS?

Done
- Don't chip; just do a single chip per scene
- Try to get parallelism working

Notes
- Try profiling
- Look into Numba and/or PyPy
- Look for landcover datasets to train against
- Target is a repo with an example
- Add ability to seed with a function
- Force function into sigmoid before evaluating. Train on a continuum, but use a threshold for predictions.
- Simulate a remote run with S3
- Reshape images to n x 8 for evaluating
