import os
import numpy as np
from ompl import base as ob
from ompl import control as oc
from ompl import geometric as og
import pydot
from IPython.display import SVG, display
import matplotlib.pyplot as plt
from pydrake.common import temp_directory
from pydrake.geometry import StartMeshcat, Box as DrakeBox
from pydrake.math import RotationMatrix, RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.visualization import AddDefaultVisualization
from pydrake.systems.framework import LeafSystem
from pydrake.systems.primitives import ConstantVectorSource, LogVectorOutput
from pydrake.all import Variable, MakeVectorVariable

from helper.dynamics import CalcRobotDynamics
from pydrake.all import (
    InverseKinematics, Solve,
    SpatialInertia, UnitInertia,
    RigidTransform, CoulombFriction
)
from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.trajectories import PiecewisePolynomial

# Start the visualizer and clean up previous instances
meshcat = StartMeshcat()
meshcat.Delete()
meshcat.DeleteAddedControls()

# Path to Panda robot URDF and the world SDF
#world_path = os.path.join("..", "models", "descriptions", "project_06_TAMP.sdf")
#robot_path = os.path.join("..", "models", "descriptions", "robots", "arms",
                          #"franka_description", "urdf", "panda_arm_hand.urdf")



#I changed the path - this is proper to me
world_path = os.path.join("..", "models", "project", "project_06_TAMP_16_Dec.sdf")
robot_path = os.path.join("..", "models", "descriptions", "robots", "arms",
                          "franka_description", "urdf", "panda_arm_hand.urdf")



######################################################################################################
#                             ########Define PD+G Controller as a LeafSystem #######
######################################################################################################

class Controller(LeafSystem):
    def __init__(self, plant, robot):
        super().__init__()

        # Declare input ports for desired and current states
        self._current_state_port = self.DeclareVectorInputPort(name="Current_state", size=18)
        self._desired_state_port = self.DeclareVectorInputPort(name="Desired_state", size=9)

        # PD+G gains (Kp and Kd)
        self.Kp_ = np.array([120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120, 120])
        self.Kd_ = np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 5, 5])
        self.robot = robot

        # Store plant and context for dynamics calculations
        self.plant, self.plant_context_ad = plant, plant.CreateDefaultContext()

        # Declare discrete state and output port for control input (tau_u)
        state_index = self.DeclareDiscreteState(9)  # 9 state variables.
        self.DeclareStateOutputPort("tau_u", state_index)  # output: y=x.
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=1 / 1000,  # One millisecond time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.compute_tau_u)  # Call the Update method defined below.

    def compute_tau_u(self, context, discrete_state):
        num_positions = self.plant.num_positions(self.robot)
        num_velocities = self.plant.num_velocities(self.robot)

        # Evaluate the input ports
        self.q_d = self._desired_state_port.Eval(context)[0:num_positions]
        self.q = self._current_state_port.Eval(context)

        # Compute gravity forces for the current state
        self.plant.SetPositionsAndVelocities(self.plant_context_ad, self.robot, self.q)
        gravity = -self.plant.CalcGravityGeneralizedForces(self.plant_context_ad)[:num_positions]

        tau = self.Kp_ * (self.q_d - self.q[:num_positions]) - self.Kd_ * self.q[num_positions:] + gravity
        # print(tau)
        # Update the output port = state
        discrete_state.get_mutable_vector().SetFromVector(tau)


######################################################################################################
#                     ########Define Trajectory Generator as a LeafSystem #######
######################################################################################################
class JointSpaceValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, check_fn, num_dof):
        super().__init__(si)
        self.check_fn = check_fn
        self.num_dof = num_dof

    def isValid(self, state):
        q = np.array([state[i] for i in range(self.num_dof)])
        return self.check_fn(q)


class MotionProfile(LeafSystem):
    def __init__(self, waypoints):
        super().__init__()

        self.waypoints = [np.asarray(w) for w in waypoints]
        self.num_points = 40

        self.index_path = 0
        self.index_q_next = 0
        self.flag = True
        self.min_distance = 0.001
        # update_period = time_period / self.num_points
        update_period = 1 / 50
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.01)
        parser = Parser(plant)
        parser.AddModelsFromUrl("file://" + os.path.abspath(robot_path))
        base_link = plant.GetBodyByName("panda_link0")
        plant.WeldFrames(plant.world_frame(), base_link.body_frame())

        # --- Table ---
        table_model = plant.AddModelInstance("table_source")

        table_inertia = SpatialInertia(mass=5.0, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(0.01, 0.01, 0.01))
        table_body = plant.AddRigidBody("table_top_link", table_model, table_inertia)

        table_shape = DrakeBox(0.4, 0.75, 0.05)
        plant.RegisterCollisionGeometry(table_body, RigidTransform(), table_shape, "table_collision",
                                        CoulombFriction(0.3, 0.3))
        plant.RegisterVisualGeometry(table_body, RigidTransform(), table_shape, "table_visual", [0.82, 0.71, 0.55, 1.0])

        table_pose = RigidTransform(RollPitchYaw(0, 0, 0), [0.55, 0.0, 0.025])
        plant.WeldFrames(plant.world_frame(), table_body.body_frame(), table_pose)

        # --- Wall ---
        wall_model = plant.AddModelInstance("wall")

        wall_inertia = SpatialInertia(mass=2.0, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(0.01, 0.01, 0.01))
        wall_body = plant.AddRigidBody("wall_top_link", wall_model, wall_inertia)

        wall_shape = DrakeBox(0.35, 0.02, 0.3)
        plant.RegisterCollisionGeometry(wall_body, RigidTransform(), wall_shape, "wall_collision",
                                        CoulombFriction(0.3, 0.3))
        plant.RegisterVisualGeometry(wall_body, RigidTransform(), wall_shape, "wall_visual", [0.92, 0.51, 0.55, 1.0])

        wall_pose = RigidTransform(RollPitchYaw(0, 0, 0), [0.55, 0.0, 0.2])
        plant.WeldFrames(plant.world_frame(), wall_body.body_frame(), wall_pose)

        plant.Finalize()
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        self.plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
        self.plant = plant
        self.diagram = diagram
        self.table = plant.GetModelInstanceByName("table_source")
        self.wall = plant.GetModelInstanceByName("wall")
        self.robot = plant.GetModelInstanceByName("panda")
        self.num_positions = self.plant.num_positions(self.robot)
        self._state_port = self.DeclareVectorInputPort(name="state", size=2 * self.num_positions)

        q_ctrl = self.DeclareDiscreteState(self.num_positions)
        self.DeclareStateOutputPort("q_ctrl", q_ctrl)

        self.DeclarePeriodicDiscreteUpdateEvent(period_sec=update_period, offset_sec=0.0,
                                                update=self.compute_trajectory)

    def compute_trajectory(self, context, discrete_state):
        """
        Periodically updates the next target joint configuration (q_next)
        along a collision-free path planned by RRTConnect.

        This function is triggered by a discrete update event inside Drake’s
        simulation loop. It computes a sequence of joint-space waypoints that
        guide the robot from its current configuration to the desired goal
        configuration while avoiding obstacles.
        """

        # Read current robot state (q + v)
        state = self._state_port.Eval(context)
        q_i = state[:self.num_positions]

        # Update plant position for collision checking
        self.plant.SetPositions(self.plant_context, q_i)

        # Compute trajectory only once
        if self.flag:
            self.trajectorys = self.get_trajectorys(self.waypoints, q_i, 10)
            self.index_path = 0
            self.index_q_next = 0
            self.flag = False

        # Default fallback: hold current position (prevents uninitialized q_next)
        q_next = q_i.copy()

        # If trajectories exist, follow them
        if self.trajectorys is not None:

            # Still inside trajectory list?
            if self.index_path < len(self.trajectorys):

                path = self.trajectorys[self.index_path]

                # Still waypoints inside this path?
                if self.index_q_next < len(path):

                    # Pick next waypoint
                    q_next = path[self.index_q_next]

                    # If robot is close enough → go to next waypoint
                    if np.linalg.norm(q_next - q_i) < 0.2:
                        self.index_q_next += 1

                else:
                    # End of this sub-path → move to next path
                    print(f"[SEQ] Reached waypoint {self.index_path}")
                    self.index_path += 1
                    self.index_q_next = 0

            else:
                # End of ALL paths → hold last waypoint
                last_path = self.trajectorys[-1]
                q_next = last_path[-1]

        # Update the system output
        discrete_state.get_mutable_vector().SetFromVector(q_next)

    def check_configuration_validity(self, q):

        # This updates the positions of all robot bodies in the scene.
        self.plant.SetPositions(self.plant_context, q)

        # This provides access to Drake's geometry engine for computing
        # distances and collision information at the current configuration.
        query_object = self.plant.get_geometry_query_input_port().Eval(self.plant_context)

        # Retrieve an inspector to map geometry IDs to frames/bodies.
        inspector = query_object.inspector()

        # Compute signed distances between all geometry pairs.
        distances = query_object.ComputeSignedDistancePairwiseClosestPoints()

        # Iterate and find the smallest *relevant* distance
        min_dist = float("inf")

        for pair in distances:

            # Extract the two bodies associated with this distance pair.
            body_A, body_B = self._get_bodies_from_pair(pair, inspector)

            # Retrieve the model names (e.g., "panda", "box") for filtering.
            robot_A_name = self._get_robot_name(body_A)
            robot_B_name = self._get_robot_name(body_B)

            # Skip if both belong to the same body or same model instance
            if robot_A_name == robot_B_name:
                continue

            # Otherwise, consider this pair
            min_dist = min(min_dist, pair.distance)

        # If nothing relevant found, it's valid
        if min_dist == float("inf"):
            return True

        # Check against threshold
        return min_dist >= self.min_distance

    def plan_joint_space(self, q_start, q_goal, timeout=10):
        """
        Plans a collision-free path in the robot's joint space using OMPL's RRT-Connect algorithm.

        Parameters
        ----------
        q_start : array-like
            The starting joint configuration of the robot.
        q_goal : array-like
            The desired target joint configuration.
        timeout : float
            The maximum time allowed for the planner to search for a valid path (in seconds).

        Returns
        -------
        sampled : np.ndarray or None
            A set of intermediate joint configurations representing the collision-free path.
            Returns None if the planner fails to find a path within the timeout.
        """
        num_dof = self.num_positions  # number of joints in the manipulator
        space = ob.RealVectorStateSpace(num_dof)  # N-dimensional Euclidean space ℝⁿ

        # Set the upper and lower limits for each joint
        bounds = ob.RealVectorBounds(num_dof)
        lower_limits = self.plant.GetPositionLowerLimits()
        upper_limits = self.plant.GetPositionUpperLimits()

        # Apply per-joint bounds to the OMPL state space
        for i in range(num_dof):
            bounds.setLow(i, lower_limits[i])
            bounds.setHigh(i, upper_limits[i])
        space.setBounds(bounds)

        # Configure the SpaceInformation (contains state validity)
        si = ob.SpaceInformation(space)
        validity_checker = JointSpaceValidityChecker(si, self.check_configuration_validity, num_dof)
        si.setStateValidityChecker(validity_checker)  # geometry engine to ensure the robot is collision-free.
        si.setup()

        # Define start and goal configurations
        start = ob.State(space)
        goal = ob.State(space)
        for i in range(num_dof):
            start[i] = q_start[i]
            goal[i] = q_goal[i]

        # Problem definition: connect q_start -> q_goal with a feasible path
        pdef = ob.ProblemDefinition(si)
        # The last argument (1e-2) is the acceptable goal tolerance in joint space
        pdef.setStartAndGoalStates(start, goal, 1e-2)

        # Choose the planner (RRT-Connect) and initialize it
        planner = og.RRTConnect(si)  # create planner object
        planner.setProblemDefinition(pdef)  # assign start/goal/problem
        planner.setup()  # initialize internal structures

        # The planner will attempt to find a collision-free path
        # connecting q_start and q_goal within the given timeout.
        solved = planner.solve(timeout)
        # Extract, simplify, and interpolate the path if found
        if solved:
            path = pdef.getSolutionPath()

            # Simplify the path by shortcutting redundant waypoints
            simplifier = og.PathSimplifier(si)
            simplifier.ropeShortcutPath(path)

            # Interpolate to obtain evenly spaced samples along the path
            path.interpolate(self.num_points)

            # Convert OMPL path states into a NumPy array [N × num_dof]
            sampled = np.array([
                [state[i] for i in range(num_dof)]
                for index, state in enumerate(path.getStates())
            ])
            return sampled
        else:
            print("RRT failed to find a path.")
            return None

    def _get_bodies_from_pair(self, pair, inspector):
        # Get the frame IDs for the two geometries in the distance pair.
        frame_id_A = inspector.GetFrameId(pair.id_A)
        frame_id_B = inspector.GetFrameId(pair.id_B)

        # Map the frame IDs to their corresponding Body objects in the plant.
        body_A = self.plant.GetBodyFromFrameId(frame_id_A)
        body_B = self.plant.GetBodyFromFrameId(frame_id_B)

        return body_A, body_B

    def _get_robot_name(self, body):
        # Retrieve the human-readable name of that model instance.
        # This allows you to distinguish which robot or object the body comes from.

        model_instance = body.model_instance()
        return self.plant.GetModelInstanceName(model_instance)

    def get_trajectorys(self, way_pts, init_q, timeout):
        self.traj = []

        self.traj.append(
            self.plan_joint_space(init_q, way_pts[0], timeout=timeout)
        )

        for i in range(len(way_pts) - 1):
            q_start = way_pts[i]
            q_goal = way_pts[i + 1]
            path = self.plan_joint_space(q_start, q_goal, timeout=timeout)
            # If planner fails, fall back to holding the current position
            if path is None:
                path = q_start
            self.traj.append(path)
        return self.traj


######################################################################################################

def plot_joint_tracking(logger_state, logger_traj, simulator_context, num_joints=9):
    """
    Plot actual vs reference joint positions and velocities from logs.
    """
    log_state = logger_state.FindLog(simulator_context)
    log_traj = logger_traj.FindLog(simulator_context)

    time = log_state.sample_times()
    q_actual = log_state.data()[:num_joints, :]
    qdot_actual = log_state.data()[num_joints:, :]

    q_ref = log_traj.data()[:num_joints, :]

    # --- Joint positions ---
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    for i in range(7):
        axes[i].plot(time, q_actual[i, :], label='q_actual')
        axes[i].plot(time, q_ref[i, :], '--', label='q_ref')
        axes[i].set_ylabel(f'Joint {i + 1} [rad]')
        axes[i].legend()
        axes[i].grid(True)
        axes[i].set_ylim(-3, 3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Joint Positions: Actual vs Reference')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def make_panda_ik(panda_path, time_step):
    plant = MultibodyPlant(time_step)
    parser = Parser(plant)
    parser.AddModelsFromUrl("file://" + os.path.abspath(panda_path))

    base = plant.GetBodyByName("panda_link0")
    plant.WeldFrames(plant.world_frame(), base.body_frame())

    plant.Finalize()
    return plant


def solve_ik(plant, context, frame_E, X_WE_desired):
    """
    Solves inverse kinematics for a given end-effector pose.

    Args:
        plant: MultibodyPlant
        context: plant.CreateDefaultContext() or similar
        frame_E: End-effector Frame (e.g. plant.GetFrameByName("ee"))
        X_WE_desired: RigidTransform of desired world pose of end-effector

    Returns:
        q_solution: numpy array of joint positions if successful, else None
    """
    ik = InverseKinematics(plant, context)

    # Set nominal joint positions to current positions
    q_nominal = plant.GetPositions(context).reshape((-1, 1))

    # Constrain position and orientation
    # Position constraint
    p_AQ = X_WE_desired.translation().reshape((3, 1))
    ik.AddPositionConstraint(
        frameB=frame_E,
        p_BQ=np.zeros((3, 1)),  # Here, p_BQ = [0, 0, 0] means we’re constraining the origin of the E frame.
        frameA=plant.world_frame(),
        p_AQ_lower=p_AQ,
        p_AQ_upper=p_AQ
    )

    # Orientation constraint
    theta_bound = 1e-2  # radians
    ik.AddOrientationConstraint(
        frameAbar=plant.world_frame(),  # world frame
        R_AbarA=X_WE_desired.rotation(),  # desired orientation
        frameBbar=frame_E,  # end-effector frame
        R_BbarB=RotationMatrix(),  # current orientation
        theta_bound=theta_bound  # allowable deviation
    )

    # Access the underlying MathematicalProgram to add costs and constraints manually.
    prog = ik.prog()
    q_var = ik.q()  # decision variables (joint angles)
    # Add a quadratic cost to stay close to the nominal configuration:
    #   cost = (q - q_nominal)^T * W * (q - q_nominal)
    W = np.identity(q_nominal.shape[0])
    prog.AddQuadraticErrorCost(W, q_nominal, q_var)

    # Enforce joint position limits from the robot model.
    lower = plant.GetPositionLowerLimits()
    upper = plant.GetPositionUpperLimits()
    prog.AddBoundingBoxConstraint(lower, upper, q_var)

    # Solve the optimization problem using Drake’s default solver.
    # The initial guess is the nominal configuration (q_nominal).
    result = Solve(prog, q_nominal)

    # Check if the solver succeeded and return the solution.
    if result.is_success():
        q_sol = result.GetSolution(q_var)
        return q_sol
    else:
        print("IK did not converge!")
        return None


def get_cube_poses(plant, context):
    cubes = ["red_link", "green_link", "blue_link"]
    poses = {}

    for link_name in cubes:
        body = plant.GetBodyByName(link_name)
        X_WB = plant.EvalBodyPoseInWorld(context, body)
        poses[link_name] = X_WB

    return poses


def pick_and_place(final_configuration, plant, context):

    current_poses = get_cube_poses(plant, context)
    operations = []

    # Paramètres zone intermédiaire
    intermediate_amount = 0
    m, n = 0.075, 0.075
    intermediate_x_pattern = [1, -1, 0, 0, -1, 1, 1, 0]
    intermediate_y_pattern = [0, 0, 1, -1, 1, 1, -1, -1]

    # On crée une liste des cubes à placer (dans l'ordre du bas vers le haut)
    # final_configuration doit être ordonnée : [cube_bas, cube_milieu, cube_haut]
    cubes_to_place = list(final_configuration.keys())

    while cubes_to_place:
        # On essaie de placer le prochain cube nécessaire pour la tour finale
        target_cube = cubes_to_place[0]

        sorted_cubes = sorted(current_poses.keys(),
            key=lambda c: current_poses[c].translation()[2],
            reverse=True)

        # 1. Est-ce que target_cube est libre (rien au-dessus) ?
        current_pos = current_poses[target_cube].translation()
        blocker = None
        for other in sorted_cubes:
            other_pose = current_poses[other]

            if other == target_cube:
                continue

            other_pos = other_pose.translation()
            # Si un cube est au-dessus (même XY, Z plus grand)
            if (abs(current_pos[0] - other_pos[0]) < 0.05 and
                    abs(current_pos[1] - other_pos[1]) < 0.05 and
                    other_pos[2] > current_pos[2]):
                blocker = other
                break

        if blocker is None:

            # Le cube est libre, on l'envoie direct à sa position finale
            actual_pos_in_memory = current_poses[target_cube].translation()
            pick_z = actual_pos_in_memory.copy()
            if pick_z[2] < 0.1675:
                pick_z[2] = 0.168

            operations.append((target_cube, "pick", RigidTransform(RollPitchYaw(np.pi, 0, 0), pick_z)))
            dest_pos1 = final_configuration[target_cube].translation()
            dest_pos = dest_pos1.copy()
            if dest_pos[2] < 0.1675:
                dest_pos[2] = 0.168

            operations.append((target_cube, "place",RigidTransform(RollPitchYaw(np.pi, 0, 0), dest_pos)))

            # Mise à jour
            current_poses[target_cube] = final_configuration[target_cube]
            cubes_to_place.pop(0)  # On passe au cube suivant de la tour finale

        else:
            # Le cube voulu est bloqué par 'blocker ' ==> on doit donc dégager le 'blocker' vers une pos intermediaire

            blocker_pick_z = current_poses[blocker].translation().copy()
            blocker_pick_z[2] += 0.06
            if blocker_pick_z[2] < 0.1675:
                blocker_pick_z[2] = 0.168

            operations.append((blocker, "pick", RigidTransform(RollPitchYaw(np.pi, 0, 0), blocker_pick_z)))

            # Calcul position intermédiaire
            pos_in_cycle = intermediate_amount % 8
            scale = (intermediate_amount // 8) + 1
            inter_trans = [
                0.5 + intermediate_x_pattern[pos_in_cycle] * scale * n,  # Offset 0.3 pour éloigner de la tour
                0.2 + intermediate_y_pattern[pos_in_cycle] * scale * m,
                0.168 ]

            inter_pose = RigidTransform(RollPitchYaw(np.pi, 0, 0), inter_trans)
            operations.append((blocker, "place", inter_pose))
            current_poses[blocker] = inter_pose
            intermediate_amount += 1


    return operations


def gripper_action(pt, offset):
    p = pt.copy()
    p[7] = offset
    p[8] = offset
    return p


# Function to Create Simulation Scene
def create_sim_scene(sim_time_step):
    builder = DiagramBuilder()
    trajectory_mode = "trapezoidal"  # Options: "trapezoidal" or "s_curve"
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)

    Parser(plant).AddModelsFromUrl("file://" + os.path.abspath(world_path))[0]  # loads tables, cubes, etc.
    panda_model = Parser(plant).AddModelsFromUrl("file://" + os.path.abspath(robot_path))[0]  # loads the Panda robot

    base_link = plant.GetBodyByName("panda_link0")  # replace with your robot’s root link name
    plant.WeldFrames(plant.world_frame(), base_link.body_frame())

    plant.Finalize()

    robot = plant.GetModelInstanceByName("panda")
    # Set the initial joint position of the robot otherwise it will correspond to zero positions
    q_start = [-1.0, -0, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0]
    plant.SetDefaultPositions(robot, q_start)
    # print(plant.GetDefaultPositions())

    # Add visualization to see the geometries in MeshCat
    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    # Add a PD+G controller to regulate the robot
    controller = builder.AddNamedSystem("PD+G controller", Controller(plant, robot))

    # Create a constant source for desired positions
    panda_ik = make_panda_ik(panda_path=robot_path, time_step=sim_time_step)
    panda_ik.SetDefaultPositions([-1.0, -0, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0])
    context_panda_ik = panda_ik.CreateDefaultContext()
    frame_E = panda_ik.GetFrameByName("panda_hand")

    context = plant.CreateDefaultContext()

    # final configuration of the cubes (from bottom to top)
    # desired_order = ["red_link", "blue_link", "green_link"]
    desired_order = ["blue_link", "green_link", "red_link"]

    # desired_circle = "link_initial"  # "link_initial" or "link_target"
    # X_WB_circle = plant.EvalBodyPoseInWorld(context, plant.GetBodyByName(desired_circle))
    cylinder_target_instance = plant.GetModelInstanceByName("cylinder_target")

    cylinder_target_body = plant.GetBodyByName("link_target", cylinder_target_instance)
    X_WB_circle = plant.EvalBodyPoseInWorld(context, cylinder_target_body)

    final_configuration = {}
    for i, cube_name in enumerate(desired_order):
        height = 0.1675 + (i) * 0.025  # wall height + cube height/2 + stack height
        final_configuration[cube_name] = RigidTransform(RotationMatrix.Identity(),
                                                        [X_WB_circle.translation()[0], X_WB_circle.translation()[1],
                                                         height])

    pick_place_sequence = pick_and_place(final_configuration, plant, context)
    print("Pick and Place Sequence:", pick_place_sequence)
    X_WE_desired_1 = RigidTransform(RollPitchYaw(np.pi, 0, 0), [0.5, 0.25, 0.1675])
    X_WE_desired_2 = RigidTransform(RollPitchYaw(np.pi, 0, 0), [0.5, -0.25, 0.1675])
    q_t1 = solve_ik(panda_ik, context_panda_ik, frame_E, X_WE_desired_1)
    q_t2 = solve_ik(panda_ik, context_panda_ik, frame_E, X_WE_desired_2)

    way_pts = []
    offset = 0.1

    # apply an offset in z  + ik
    for i in range(len(pick_place_sequence)):
        X = pick_place_sequence[i][2]
        p = X.translation().copy()  # np.array (3,)
        p[2] += offset

        way_pts.append(
            ('----', solve_ik(panda_ik, context_panda_ik, frame_E, RigidTransform(RollPitchYaw(np.pi, 0, 0), p))))
        way_pts.append((pick_place_sequence[i][1], solve_ik(panda_ik, context_panda_ik, frame_E,
                                                            RigidTransform(RollPitchYaw(np.pi, 0, 0),
                                                                           X.translation()))))
        way_pts.append(
            ('----', solve_ik(panda_ik, context_panda_ik, frame_E, RigidTransform(RollPitchYaw(np.pi, 0, 0), p))))

    way_pts_bis = []
    for i in (way_pts):
        prev = ''
        if i[0] == 'place':
            way_pts_bis.append(gripper_action(i[1], 0.00))
            way_pts_bis.append(gripper_action(i[1], 0.04))
        elif i[0] == 'pick':
            way_pts_bis.append(gripper_action(i[1], 0.04))
            way_pts_bis.append(gripper_action(i[1], 0.00))
        else:
            way_pts_bis.append(i[1])

    # print(len(way_pts_bis))

    path_planner = builder.AddNamedSystem("Motion Profile", MotionProfile(waypoints=way_pts_bis))

    # Connect systems: plant outputs to controller inputs, and vice versa

    builder.Connect(plant.GetOutputPort("panda_state"), path_planner.GetInputPort("state"))
    builder.Connect(path_planner.get_output_port(0), controller.GetInputPort("Desired_state"))
    builder.Connect(plant.GetOutputPort("panda_state"), controller.GetInputPort("Current_state"))
    builder.Connect(controller.GetOutputPort("tau_u"), plant.GetInputPort("panda_actuation"))

    logger_state = LogVectorOutput(plant.GetOutputPort("panda_state"), builder)
    logger_state.set_name("State logger")

    logger_traj = LogVectorOutput(path_planner.get_output_port(0), builder)
    logger_traj.set_name("Trajectory logger")

    # Build and return the diagram
    diagram = builder.Build()
    return diagram, logger_state, logger_traj


# Create a function to run the simulation scene and save the block diagram:
def run_simulation(sim_time_step):
    diagram, logger_state, logger_traj = create_sim_scene(sim_time_step)
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.)

    # Save the block diagram as an image file
    svg_data = diagram.GetGraphvizString(max_depth=2)
    graph = pydot.graph_from_dot_data(svg_data)[0]
    image_path = "block_diagram_04_path_planner.png"  # Change this path as needed
    graph.write_png(image_path)
    print(f"Block diagram saved as: {image_path}")

    # Run simulation and record for replays in MeshCat
    meshcat.StartRecording()
    simulator.AdvanceTo(60.0)  # Adjust this time as needed
    meshcat.PublishRecording()

    # At the end of the simulation
    plot_joint_tracking(logger_state, logger_traj, simulator.get_context())


# Run the simulation with a specific time step. Try gradually increasing it!
run_simulation(sim_time_step=0.0005)