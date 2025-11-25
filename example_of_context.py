import os
import numpy as np
import csv
import pydot
from IPython.display import SVG, display
from pydrake.all import ( InverseDynamics, 
    AddMultibodyPlantSceneGraph, AddDefaultVisualization, StartMeshcat, BasicVector, AbstractValue, JacobianWrtVariable, ContactWrenchFromForceInWorldFrameEvaluator,
    Simulator, ConstantVectorSource, DifferentialInverseKinematicsParameters, MultibodyPlant, MakeVectorVariable, SpatialForce,
    DifferentialInverseKinematicsIntegrator, Value, RigidTransform, RollPitchYaw, InverseDynamicsController, ContactResults, Quaternion, AddMultibodyPlant, MultibodyPlantConfig
)
from pydrake.multibody.tree import MultibodyForces_
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import ApplyVisualizationConfig, VisualizationConfig
from dynamics import RobotDynamics, CalcRobotDynamics
from ObjectDynamics import KDObject

########################################################################################
#################### ####################
########################################################################################

def add_viz(builder, plant):
    ApplyVisualizationConfig(
        config=VisualizationConfig(
                   publish_period = 1 / 256.0,
                   publish_contacts = True),
        builder=builder, meshcat=meshcat)
    
    return builder, plant
# Helper: Retrieve relative path to the script's directory
def get_relative_path(path):
    """Returns the absolute path of a file relative to this script."""
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))

# Helper function to write data to a CSV file
def write_to_csv(file_path, data):
    """Writes a row of data to the specified CSV file."""
    with open(file_path, mode='a', newline='') as file:
        csv.writer(file).writerow(data)

# LeafSystem to report the dynamics of the object
class DynamicReporter(LeafSystem):
    def __init__(self, plant, context, inversedynamic, scene_graph, state_file="kinematic_states.csv", forces_file="dynamics_forces.csv"):
        LeafSystem.__init__(self)
        self.plant = plant
        self.plant_context = context
        
        self.inspector = scene_graph.model_inspector()
        self.id_robot = inversedynamic

        self.num_joints = plant.num_actuated_dofs()
        self.state_file = state_file
        self.forces_file = forces_file
        self.contact_force = [0, 0, 0]
        self.contact_point = [0, 0, 0]
        self.p_WCa = [0, 0, 0] 
        # self.p_ACa
        self.p_WCb = [0, 0, 0]
        self.box_Fb = np.array(6)
        # self.p_BCb
        self.counter = 0
        self.Jv_WC = []

        self.initial_pose = plant.EvalBodyPoseInWorld(self.plant_context, plant.GetRigidBodyByName("planar_end_eff_link"))
        self.inital_twist = plant.EvalBodySpatialVelocityInWorld(self.plant_context, plant.GetRigidBodyByName("planar_end_eff_link"))
        
        x_new = self.initial_pose.translation()[0] # 
        y_new = self.initial_pose.translation()[1]  #         
        z_new = self.initial_pose.translation()[2] #  
        self.translation_new = np.array([x_new, y_new, z_new])
        self.X_WT_v = RigidTransform(self.initial_pose.rotation(), self.translation_new) # 
        
        
        # Declare input and output ports
        self.DeclareVectorInputPort("current_state", plant.num_positions() + plant.num_velocities())
        self.DeclareAbstractInputPort("contact_results", AbstractValue.Make(ContactResults())) #Value(ContactResults())
        # self.DeclareVectorInputPort("object_acceleraion", 6) 
        self.DeclareVectorInputPort("applied_force", self.num_joints)
        self.DeclareVectorInputPort("acceleraion", plant.num_velocities())

        state_index = self.DeclareDiscreteState(self.num_joints)  # One state variable.
        self.DeclareStateOutputPort("tau_u", state_index)  # One output: y=x.
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=0.001,  # One second time step.
            offset_sec=0.0,  # The first event is at time zero.
            update=self.NPControl) # Call the Update method defined below.
        
        

    def NPControl(self, context, discrete_state):
        print(f"\nContactReporter::Publish() called at time={context.get_time()}")
        # Get current initial object state [Done]
        current_state = self.get_input_port(0).Eval(context)
        contact_results = self.get_input_port(1).Eval(context)
        # obj_acc = self.get_input_port(2).Eval(context)
        tau_u_idc = self.get_input_port(3).Eval(context)
        # acc = self.get_input_port(4).Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, current_state)

        # Check for contact and compute force if present
        if contact_results.num_hydroelastic_contacts() > 0:
            contact_info = contact_results.hydroelastic_contact_info(0)
            self.box_Fb = np.concatenate((contact_info.F_Ac_W().rotational(), contact_info.F_Ac_W().translational()))
            contact_surface = contact_info.contact_surface()
            self.p_WC = contact_surface.centroid()
        else:
            self.box_Fb = np.zeros(6)  # No contact forces

        # Compute current and desired poses/velocities
        X_WT = self.plant.EvalBodyPoseInWorld(self.plant_context, self.plant.GetBodyByName("planar_end_eff_link"))
        V_WT = self.plant.EvalBodySpatialVelocityInWorld(self.plant_context, self.plant.GetBodyByName("planar_end_eff_link"))
        
        # Pose error calculation
        R_error = X_WT.rotation().inverse().multiply(self.X_WT_v.rotation())
        position_error = self.X_WT_v.translation() - X_WT.translation()
        angle_axis = R_error.ToAngleAxis()
        if np.linalg.norm(angle_axis.angle()) > 1e-6:
            orientation_error = angle_axis.axis() * angle_axis.angle()
        else:
            orientation_error = np.zeros(3)
        # orientation_error = Quaternion(R_error.matrix()).wxyz()[1:]
        e_X_E = np.hstack((orientation_error, position_error))
        e_dX_E = np.concatenate([np.array(V_WT.rotational()), np.array(V_WT.translational())])


        # Compute the spatial velocity Jacobian
        R_EW = X_WT.rotation().inverse()  # Rotation from world to end-effector
        J_WE_W = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,
            self.plant.GetBodyByName("planar_end_eff_link").body_frame(),
            np.zeros(3),
            self.plant.world_frame(),
            self.plant.world_frame()
        )

        # Rotate the Jacobian to the end-effector frame
        J_WE_E = np.zeros_like(J_WE_W)
        J_WE_E[:3, :] = R_EW @ J_WE_W[:3, :]  # Rotational part
        J_WE_E[3:, :] = R_EW @ J_WE_W[3:, :]  # Translational part


        
        # Gain matrices for PD control
        Kp = np.diag([0.8, 0.8, 0.8, 1.8, 1.8, 1.8])
        Kd = np.diag([0.02, 0.02, 0.02, 0.5, 0.5, 0])

        Kpm=[120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0]
        Kdm=[8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0]
        
        # Compute desired acceleration in operational space
        xdd_des = np.dot(Kp, e_X_E) - np.dot(Kd, e_dX_E)
        print(e_X_E, e_dX_E)
        # Mass matrix in task and joint and gravity compensation
        Mq = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
        Mx = np.linalg.pinv(J_WE_E @ np.linalg.solve(Mq, J_WE_E.T))# Compute operational space mass matrix Mx 
        tau_g = self.plant.CalcGravityGeneralizedForces(self.plant_context)
        Cv = self.plant.CalcBiasTerm(self.plant_context)
        forces = MultibodyForces_(self.plant)
        self.plant.CalcForceElementsContribution(self.plant_context, forces)
        tauExt = forces.generalized_forces()

        # Compute primary torque --motion Command--
        F_primary_cmd =  xdd_des # (Mx @  
        
        # Null-space projection for secondary tasks
        N = np.eye(J_WE_E.shape[1]) - J_WE_E.T @ np.linalg.pinv(J_WE_E).T #  null-space projection matrix (N):
        
        # Secondary task joint control
        lamda = np.array([0.0, 1.57, -1.57, -1.2, -1.57, 1.57, -0.37]) # secondary task
        joint_pos_error = lamda - np.array(current_state[:7])
        joint_vel_error = np.array(current_state[7:])
        aux_joint_cc = np.multiply(Kpm, joint_pos_error) - np.multiply(Kdm, joint_vel_error)
        tau_secondary = Mq @ aux_joint_cc
        tau_secondary_cmd = N @ tau_secondary

        # Final torque command
        qdd_des = Mq @ (J_WE_E.T @ xdd_des) + Cv + tauExt - tau_g #  + tau_secondary_cmd #  
        tau_u = J_WE_E.T @ F_primary_cmd + Cv + tauExt - tau_g #  + tau_secondary_cmd #  
        discrete_state.get_mutable_vector().SetFromVector(tau_u)

    def compute_rotation_error(self, current_rotation, desired_rotation):
        # Compute rotation difference using axis-angle representation
        rotation_diff = current_rotation.inverse() @ desired_rotation
        axis_angle = rotation_diff.ToAngleAxis()
        # Return rotation error as a vector (scaled axis)
        return axis_angle.angle() * axis_angle.axis()

# LeafSystem to augment joint positions with zero velocities
class JointStateAugmenter(LeafSystem):
    def __init__(self, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.DeclareVectorInputPort("joint_positions", BasicVector(num_joints))
        self.DeclareVectorOutputPort("state", BasicVector(num_joints * 2), self.DoCalcOutput)

    def DoCalcOutput(self, context, output):
        positions = self.get_input_port(0).Eval(context)
        state = np.zeros(2 * self.num_joints)
        # state[:self.num_joints] = positions
        output.SetFromVector(state)

# LeafSystem to generate a circular end-effector trajectory
class CircularTrajectory(LeafSystem):
    def __init__(self, radius, angular_velocity, initial_pose):
        super().__init__()
        self.radius = radius
        self.angular_velocity = angular_velocity
        self.initial_pose = initial_pose
        self.DeclareAbstractOutputPort("pose", lambda: Value(RigidTransform()), self.DoCalcOutput)

    def DoCalcOutput(self, context, output):
        time = context.get_time()
        # Compute circular path in the XY plane, keeping Z constant
        x_new = self.radius * np.cos(self.angular_velocity * time)
        y_new = self.radius * np.sin(self.angular_velocity * time)
        z_new = self.initial_pose.translation()[2]  # 1.8 * np.cos(self.angular_velocity * time) 
        translation_new = np.array([x_new, y_new, z_new])
        rotation = self.initial_pose.rotation()
        output.set_value(self.initial_pose) #RigidTransform(rotation, translation_new)


# Path to the robot model
robot_path = get_relative_path("../../../models/descriptions/robots/panda_fr3/urdf/panda_fr3_tray.urdf")


# Initialize MeshCat visualizer
meshcat = StartMeshcat()
meshcat.Delete()
meshcat.DeleteAddedControls()

# Create simulation scene
def create_sim_scene(sim_time_step):
    """Creates a simulation diagram with a robot and controller."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlant(
        MultibodyPlantConfig(
            time_step=sim_time_step,
            discrete_contact_approximation="similar"),
        builder)
    parser = Parser(plant)
    parser.AddModelsFromUrl(f"file://{robot_path}")
    # parser.AddModels(url="file:///home/cory/Documents/drake_brubotics/models/objects&scenes/objects/box.sdf")
    # plant.SetUseSampledOutputPorts(use_sampled_output_ports=True)  # We're not stepping time.
    plant.Finalize()

    # Set default positions for the robot
    plant.SetDefaultPositions(
        plant.GetModelInstanceByName("panda"),[0.0, 1.57, -1.57, -1.2, -1.57, 1.57, -0.37] # [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]# 
    )

    # Set the initial pose of the box
    plant_context = plant.CreateDefaultContext()
    # box_pose = plant.EvalBodyPoseInWorld(plant_context, plant.GetRigidBodyByName("planar_end_eff_link"))
    # box_pose.set_translation(box_pose.translation() + np.array([0, 0, 0.048]))
    # plant.SetDefaultFreeBodyPose(plant.GetBodyByName("box_link"), box_pose)

    # Add Inverse Dynamics Controller
    plant_id = MultibodyPlant(0.0)
    Parser(plant_id).AddModelsFromUrl(f"file://{robot_path}")
    plant_id.Finalize()

    # idc = builder.AddSystem(
    #     InverseDynamicsController(
    #         plant_id,
    #         kp=[120.0, 120.0, 120.0, 100.0, 50.0, 45.0, 15.0],
    #         ki=np.ones(plant_id.num_positions()),
    #         kd=[8.0, 8.0, 8.0, 5.0, 2.0, 2.0, 1.0],
    #         has_reference_acceleration=False
    #     )
    # )

    # Circular trajectory generator
    default_pose = plant.EvalBodyPoseInWorld(plant_context, plant.GetRigidBodyByName("planar_end_eff_link"))
    circular_trajectory = builder.AddNamedSystem(
        "CircularTrajectory",
        CircularTrajectory(radius=0.8, angular_velocity=0.1 * np.pi, initial_pose=default_pose)
    )

    # Configure Differential Inverse Kinematics
    params = DifferentialInverseKinematicsParameters(7, 7)
    params.set_time_step(0.001)
    panda_velocity_limits = np.array([2.175] * 4 + [2.61] * 3)
    panda_q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    panda_q_max = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    factor = 0.5
    params.set_joint_velocity_limits((-factor * panda_velocity_limits, factor * panda_velocity_limits))
    params.set_joint_position_limits((panda_q_min, panda_q_max))

    differential_ik = builder.AddSystem(DifferentialInverseKinematicsIntegrator(
        plant_id, plant_id.GetFrameByName("planar_end_eff_link"), time_step=0.001, parameters=params
    ))

    # Joint state augmenter
    joint_state_augmenter = builder.AddSystem(JointStateAugmenter(plant_id.num_positions()))

    inversedynamic = builder.AddSystem(InverseDynamics(plant, InverseDynamics.InverseDynamicsMode.kInverseDynamics))

    # Add DynamicReporter system with a custom file name
    reporter = builder.AddSystem(DynamicReporter(plant, plant_context, inversedynamic, scene_graph))

    
    # # Add visualization
    # AddDefaultVisualization(builder, meshcat)

    # Connect systems
    builder.Connect(circular_trajectory.get_output_port(), differential_ik.GetInputPort("X_AE_desired"))
    builder.Connect(plant.GetOutputPort("panda_state"), differential_ik.GetInputPort("robot_state"))
    builder.Connect(differential_ik.GetOutputPort("joint_positions"), joint_state_augmenter.get_input_port(0))
    builder.Connect(reporter.get_output_port(0), inversedynamic.get_input_port_desired_acceleration())
    builder.Connect(plant.GetOutputPort("panda_state"), inversedynamic.get_input_port_estimated_state())
    


    # builder.Connect(plant.GetOutputPort("panda_state"), idc.get_input_port_desired_state())
    # builder.Connect(idc.get_output_port_control(), plant.GetInputPort("panda_actuation"))
    # builder.Connect(plant.GetOutputPort("panda_state"), idc.get_input_port_estimated_state())

    # builder.Connect(reporter.get_output_port(0), idc.get_input_port_desired_state())

    builder.Connect(plant.get_state_output_port(), reporter.get_input_port(0))
    builder.Connect(plant.get_contact_results_output_port(), reporter.get_input_port(1))
    # builder.Connect(plant.GetOutputPort("box_generalized_acceleration"), reporter.get_input_port(2))
    builder.Connect(plant.GetOutputPort("generalized_acceleration"), reporter.get_input_port(3))
    builder.Connect(plant.GetOutputPort("panda_net_actuation"), reporter.get_input_port(2))

    # builder.Connect(inversedynamic.get_output_port_generalized_force(), plant.GetInputPort("panda_actuation"))
    builder.Connect(reporter.get_output_port(0), plant.GetInputPort("panda_actuation"))
    
    return builder, plant

# Run simulation
def run_simulation(sim_time_step):
    """Runs the simulation and saves the block diagram."""
    builder, plant = create_sim_scene(sim_time_step)
    add_viz(builder, plant)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)

    # Save block diagram as PNG
    svg_data = diagram.GetGraphvizString(max_depth=2)
    graph = pydot.graph_from_dot_data(svg_data)[0]
    image_path = os.path.join(os.getcwd(), "block_diagram_NP.png")

    try:
        graph.write_png(image_path)
        print(f"Block diagram saved as: {image_path}")
    except Exception as e:
        print(f"Failed to save block diagram: {e}")

    # Record and run simulation
    meshcat.StartRecording()
    simulator.AdvanceTo(10.0)
    meshcat.PublishRecording()

# Run the simulation with a specific time step
run_simulation(sim_time_step=0.001)
