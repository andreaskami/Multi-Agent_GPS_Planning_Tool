# Making a topology in Google Earth

This is a step-by-step guide of how to create a topology in Google Earth for use with the simulation environment.

## Step-by-step process
1. Navigate to the area you're interested in in Google Earth Pro (desktop version, free to download).
2. Create a new folder.
3. Add any number of Polygons and Placemarks on the map, and store them in your folder. An overview of which objects to create can be found below.
4. Make sure your topology has at least one *boundary* polygon and at least one *agent* placemark.
5. Right-click your folder and select *Save as...*
6. Save the file as a *.kml* file.
7. You can now import the topology as a simulation world by passing the file path to the *.kml* file as the `top` argument when creating the `Topology` class.

## Object options
### Boundary: 
Only one instance allowed. Adding a boundary is obligatory.

Create a polygon to mark the simulation environment's boundary. Make sure its name contains **"boundary"**. Can be either convex or concave.

### Obstacle / Wall:
Any number of instances allowed.

Create a polygon to mark an obstacle or wall. Obstacles should be convex polygons. 
If they are not convex, their convex hull will be imported instead when creating the environment.

Add the word **"hard"** to an obstacle's name to make sure it is not possible for drones to fly over the obstacle.
If it is *not* **"hard"**, ground agents will collide with it, but drones will be able to fly over it.
Besides this, the obstacle's name does not matter, but it should *not* contain **"boundary"**.

### Agent / Agent spawn location:
Any number of instances allowed. At least one instance is obligatory for correct functioning of the simulator.

Create a placemark to either mark the starting spot for a single agent, or a group of agents.
Its name should contain the word **"agent"**. Whether one or more agents will be spawned at/around this position is determined by the `random_spawn_a` flag.

### Landmark / point of interest:
Any number of instances allowed.

Create a placemark to mark a point of interest. The point's name does not matter, although it should *not* contain the word **"agent"**.
If the `random_spawn_l` flag is set to `True`, these points will not be used and the POIs will be generated randomly instead.

## Other tips
If you set a polygon area to *outlined* instead of *filled* or *filled+outlined*, it is much easier to work with.