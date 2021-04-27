# GPS-based robot path planning tool
A master thesis graduation project, by Jochem Postmes

To do: cite conference paper

## Installing the tool
Make sure you are using Python 3.8 or higher. Install all dependencies with 
```
pip install -r requirements.txt
```
Currently there is an issue with `shapely` working incorrectly if you are using a `conda` environment. To circumvent
this issue install shapely via `conda` instead of via `pip`:
```
pip uninstall shapely
conda config --add channels conda-forge
conda install shapely
```

## Running the tool
Run the tool with
```
python main.py
```
Insert picture of GUI

## Using the tool
Press `Exit` to exit the tool.

### File input
As input, there are two options to choose from: load a Google Earth `kml` file or load a premade scenario from file.

For instructions how to create a topology in Google Earth for use with the tool, see [this guide](google_earth_guide.md)

To load an earlier stored scenario, select either the `scenario.dictionary`, `topology.dictionary` or `world.dictionary` file for loading. The others will be included automatically.

After choosing a file to load, press the `make environment` button. An environment for planning will now be created with the settings from **Input settings**. If loading a premade scenario, **Input settings** will be updated to match the settings of the loaded scenario.

### File output
To select a custom output folder, click `Select output directory` and select the folder where you want the output of the tool to be stored. By default, this is the root folder containing the tool files.


### Input settings
There are various input settings that can be set when creating an environment. 

`Random spawn location` randomly relocates the agent starting location if set.

`Ground-based agents` sets the agents to aerial agents when unchecked, and ground agents when checked.

`Amount` sets the amount of agents, or landmarks, respectively.

Under `Type` the type of landmarks generated can be selected, from one of the four options: `From topology`, `Grid`, `Random` (default) or `Random (clustered)`.

`Row obstacles` creates row-based obstacles based on the underlying grid when checked.

`Grid size` sets the grid size in local coordinates - a grid size of `1.0` would divide the environment in the middle in both directions.

`Grid rotation` sets the grid rotation in degrees.


### Output settings

Under **Output settings**, the algorithm used to create a solution can be selected from the list of available algorithms.

If `Render output` is unchecked, the environment and output solution will not be rendered.

If all settings are as they should be, press `Create solution` to generate a solution to the loaded environment for the specified number of agents.
