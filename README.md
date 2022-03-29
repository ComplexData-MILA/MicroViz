# MicroViz
An interactive visual system designed for domain experts to annotate suspicious clusters of escort ads

## Creating a virtual environment

It is highly recommended to use a virtual environment for running this app as it helps dealing with dependencies of packages.

If you don't have virtualenv installed, run
```
pip install virtualenv
```

After installing virtualenv, create a virtual env for the app by running
```
python -m venv microviz_env
```

After creating a virtual env, activate it by running
```
source microviz_env/bin/activate
```

Once the virtual env has been activated, the necessary packages can be installed within that environment.

When you're done using the app, the virtual env can be deactivated by running
```
deactivate
```


## Install necessary libraries:
```
pip install -r requirements.txt
```

# Usage:

All the required data files are included

## Micro-cluster visualization
```
python app.py
```

<!-- ## Labeling app
```
streamlit run run_app.py
```

There are 6 main components in the dashboard.

1. Metadata summary of the current cluster which appears in the header of the dashboard
2. Feature embeddings of the micro-cluster. You can choose between ICA (Independent Component Analysis) and TSNE, both of which are dimensionality reduction techniques, for plotting the 12 dimensional micro-cluster feature vectors into 2 dimensions. Additionally, you can also choose between KMeans (based on distance from centroid) and HDBSCAN (based on dense blocks) to further group the points. Each group is represented by a different color. Each point on the plot represents a micro-cluster in the data and on hovering on a point, you can see its feature values. The current micro-cluster (whose information is displayed in the rest of the dashboard) is indicated by a red star. You may have to enlarge the plot and zoom in (or out) for the best viewing experience.
3. Metadata over time in the current micro-cluster. 
4. Geographical spread of the ads in the current micro-cluster.
5. Description of ads in the current micro-cluster.
6. Labeling panel to indicate the likelihood of the micro-cluster belonging to each of the categories. When done, click on `save and next` to save the labels and move to the next micro-cluster. -->

