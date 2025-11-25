
import os
import numpy as np
import pydot
from IPython.display import SVG, display

# Import necessary parts of Drake
from pydrake.geometry import StartMeshcat, SceneGraph, Box as DrakeBox, HalfSpace
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant, CoulombFriction
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer

# --------------------------------------------------------------------
# Global settings
# --------------------------------------------------------------------
# Start Meshcat visualizer: this will give you a link in the notebook/terminal
meshcat = StartMeshcat()

visualize = False  # If True: use ModelVisualizer interactively, if False: run a simulation
#False lets you view the full simulation, True: you can not move anything
add_bodies = True  # If True: add ground and a box to the scene

# Path to Panda robot URDF

#Path to the robot we want to use
model_path = os.path.join(
    "..", "models", "descriptions", "project_06_TAMP.sdf"
)

def create_sim_scene(sim_time_step):   
    """
    Create a simulation scene with the Panda robot and optional extra bodies.

    Args:
        sim_time_step (float): The discrete time step for the plant.

    Returns:
        diagram (Diagram): A system diagram with the robot, optional objects, and visualization.
    """
    # Clean up the Meshcat window (so we start from an empty scene each time)
    meshcat.Delete()
    meshcat.DeleteAddedControls()

    # A DiagramBuilder is where we construct our system
    builder = DiagramBuilder()

    # Add a MultibodyPlant (for physics) and a SceneGraph (for geometry/visualization)
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    #Set up a multibodyplant and register to the scene
    #Time step: decides if we have a discrete or continuous plant(nonzero=discrete,zero=cont)


    # Load the Panda robot from URDF
    panda_model = Parser(plant).AddModelsFromUrl("file://" + os.path.abspath(model_path))[0]

    # Fix the Panda base to the world so it doesn’t fall
    #base_link = plant.GetBodyByName("panda_link0")
    #plant.WeldFrames(plant.world_frame(), base_link.body_frame())

    # ----------------------------------------------------------------
    # Extra bodies (floor + box) if requested
    # ----------------------------------------------------------------
    if add_bodies:
        # --- Ground plane ---
        # A HalfSpace is an infinite plane (we put it at z=0).
        # We add both collision (for physics) and visual (for rendering).
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            RigidTransform(),    # Pose (identity, so plane at z=0)
            HalfSpace(),         # Geometry type
            "ground_collision",
            CoulombFriction(0.9, 0.8)   # Friction coefficients
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            RigidTransform(),
            HalfSpace(),
            "ground_visual",
            [0.5, 0.5, 0.5, 1.0]  # RGBA color (gray)
        )
    plant.Finalize()

    # ----------------------------------------------------------------
    # Inspect contents: print bodies and joints of the Panda
    # ----------------------------------------------------------------
    print("\nBodies in the Panda model:")
    for body_index in plant.GetBodyIndices(panda_model):
        print("  -", plant.get_body(body_index).name())

    print("\nJoints in the Panda model:")
    for joint_index in plant.GetJointIndices(panda_model):
        print("  -", plant.get_joint(joint_index).name())

    # ----------------------------------------------------------------
    # Set initial/default states
    # ----------------------------------------------------------------
    # Panda arm default joint configuration
    #plant.SetDefaultPositions(panda_model, [
        #0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.774, 0, 0
    #])
    
    # ----------------------------------------------------------------
    # Inspect initial state
    # ----------------------------------------------------------------
    plant_context = plant.CreateDefaultContext()
    print("\nInitial Panda joint state (positions + velocities):")
    print(plant.GetPositionsAndVelocities(plant_context, panda_model))

    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    # Build the final diagram (plant + scene graph + viz)
    diagram = builder.Build()
    return diagram

def run_simulation(sim_time_step):
    """
    Either run an interactive visualizer, or simulate the system.
    """
    if visualize:
        # If visualize=True, just load and display the robot interactively
        visualizer = ModelVisualizer(meshcat=meshcat)
        visualizer.parser().AddModelsFromUrl("file://" + os.path.abspath(model_path))
        visualizer.Run()
        
    else:
        # Otherwise, build the scene and simulate
        diagram = create_sim_scene(sim_time_step)

        # Create and configure the simulator
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(1.0)  # Try to match real time
        simulator.Initialize()
        simulator.set_publish_every_time_step(True)  # publish at each step
        #Non-actuated robot but robot falls down due to gravity

        sim_time = 5.0  # seconds of simulated time

        meshcat.StartRecording()         # Start recording the sim
        simulator.AdvanceTo(sim_time)    # Runs the simulation for sim_time seconds
        meshcat.PublishRecording()       # Publish recording to replay in Meshcat
            
        # Save system block diagram as PNG
        svg_data = diagram.GetGraphvizString(max_depth=2)
        graph = pydot.graph_from_dot_data(svg_data)[0]
        image_path = "figures/block_diagram_02.png"
        graph.write_png(image_path)
        print(f"\nBlock diagram saved as: {image_path}")
        #Block diagram handy for inspecting connections of system


# --------------------------------------------------------------------
# Run the simulation
# --------------------------------------------------------------------
# Try playing with the time step (e.g. 0.001 vs 0.01 vs 0.1)
run_simulation(sim_time_step=0.01)
    
