"""
Helper utilities to build scenarios/experiments
"""
import os
import sys
import numpy as np
from pyvirtualdisplay import Display

import pydrake.all
from pydrake.all import (
    RigidTransform, RollPitchYaw, MakeRenderEngineVtk, RenderEngineVtkParams,
    DepthRenderCamera, RenderCameraCore, CameraInfo, ClippingRange, DepthRange,
    RgbdSensor
)


def AddRgbdSensor(
    builder,
    scene_graph,
    X_PC,
    camera_param=None,
    depth_camera=None,
    renderer=None,
    parent_frame_id=None,
):
    """ Adds a RgbdSensor to to the scene_graph at (fixed) pose X_PC relative to
    the parent_frame.  If depth_camera is None, then a default camera info will
    be used.  If renderer is None, then we will assume the name 'my_renderer',
    and create a VTK renderer if a renderer of that name doesn't exist.  If
    parent_frame is None, then the world frame is used.
    """
    """ Rendering: https://drake.mit.edu/doxygen_cxx/group__render__engines.html """
    """It's important to carefully coordinate depth range and clipping planes. It might seem reasonable to use the depth range as clipping planes, but that would be a mistake. Objects closer than the depth range's minimum value have an occluding effect in reality. If the near clipping plane is set to the minimum depth range value, those objects will be clipped away and won't occlude as they should. In essence, the camera will see through them and return incorrect values from beyond the missing geometry. The near clipping plane should always be closer than the minimum depth range. How much closer depends on the scenario. Given the scenario, evaluate the closest possible distance to the camera that geometry in the scene could possibly achieve; the clipping plane should be slightly closer than that. When in doubt, some very small value (e.g., 1 mm) is typically safe.
    """
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        virtual_display = Display(visible=0, size=(1400, 900))
        # virtual_display = Display(backend="xvfb")
        virtual_display.start()

    if not renderer:
        renderer = "my_renderer"

    if not parent_frame_id:
        parent_frame_id = scene_graph.world_frame_id()

    if not scene_graph.HasRenderer(renderer):
        renderer_instance = MakeRenderEngineVtk(RenderEngineVtkParams())
        scene_graph.AddRenderer(renderer, renderer_instance)

    if not depth_camera:
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer,
                CameraInfo(
                    width=camera_param.img_W, height=camera_param.img_H,
                    fov_y=camera_param.fov * np.pi / 180.0
                ), ClippingRange(near=0.01, far=10.0), RigidTransform()
            ), DepthRange(0.1, camera_param.max_depth)
        )

    rgbd = builder.AddSystem(
        RgbdSensor(
            parent_id=parent_frame_id, X_PB=X_PC, depth_camera=depth_camera,
            show_window=False
        )
    )

    builder.Connect(
        scene_graph.get_query_output_port(), rgbd.query_object_input_port()
    )
    return rgbd, renderer_instance


def AddPanda(
    plant,
    q0=[0.0, 0.1, 0, -1.2, 0, 1.6, 0],
    X_WB=RigidTransform(),
    joint_damping=200,
    hand_type=None,
):
    """ Adds a franka panda arm without any hand to the mutlibody plant and welds it to the world frame

    plant: the multibody plant to add the panda to
    q0: the initial joint positions (optional)
    X_WB: the desired transformation between the world frame (W) and the base link of the panda (B)
    """

    # Use urdf with hand inertia added if hand_type specified. For controller plant.
    if hand_type == 'wsg':
        urdf_file = 'asset/franka_description/urdf/hand_inertia/panda_arm_wsg_inertia.urdf'
    elif hand_type == 'plate':
        urdf_file = 'asset/franka_description/urdf/hand_inertia/panda_arm_plate_inertia.urdf'
    elif hand_type == 'panda':
        urdf_file = 'asset/franka_description/urdf/hand_inertia/panda_arm_panda_inertia.urdf'
    else:
        urdf_file = 'asset/franka_description/urdf/panda_arm.urdf'
    parser = pydrake.multibody.parsing.Parser(plant)
    panda_model_instance = parser.AddModelFromFile(urdf_file)

    # Weld panda to world
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("panda_link0"), X_WB
    )

    # Set default joint
    index = 0
    for joint_index in plant.GetJointIndices(panda_model_instance):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, pydrake.multibody.tree.RevoluteJoint):
            joint.set_default_angle(q0[index])
            joint.set_default_damping(joint_damping)  # joint.damping()
            index += 1
    return panda_model_instance


def AddHand(
    plant,
    panda_model_instance=None,
    # welded=False,
    type='wsg',
):
    """Adds a hand to the panda arm (panda_link8)

    plant: the multibody plant 
    panda_model_instance: the panda model instance to add the hand to
    roll: the rotation of the hand relative to panda_link8
    welded: if we want the version with welded fingers (for control)
    """
    parser = pydrake.multibody.parsing.Parser(plant)
    if type == 'wsg':
        file_path = 'asset/wsg_50_description/sdf/schunk_wsg_50_box.sdf'
        gripper_base_frame_name = 'gripper_base'
        X_8G = RigidTransform(
            RollPitchYaw(np.pi / 2.0, 0, 0), [0, 0, 0.03625 + 0.01]
        )  # 0.03625: half dim of gripper base; 0.01: connector on real robot
    elif type == 'panda':
        file_path = 'asset/franka_description/urdf/panda_hand.urdf'
        gripper_base_frame_name = 'panda_hand'
        X_8G = RigidTransform(RollPitchYaw(0, 0, -np.pi / 2), [0, 0, 0])
    elif type == 'panda_foam':
        file_path = 'asset/franka_description/urdf/panda_hand_foam.urdf'
        gripper_base_frame_name = 'panda_hand'
        X_8G = RigidTransform(RollPitchYaw(0, 0, -np.pi / 2), [0, 0, 0])
    elif type == 'plate':
        file_path = 'asset/hand_plate/hand_plate.sdf'
        gripper_base_frame_name = 'plate_base'
        X_8G = RigidTransform(RollPitchYaw(0, 0, 0), [0, 0, 0.06])
    else:
        raise NotImplementedError
    gripper = parser.AddModelFromFile(file_path)

    # Add coupler constraint.
    if type == 'panda' or type == 'panda_foam':
        left_slider = plant.GetJointByName("panda_finger_joint1")
        right_slider = plant.GetJointByName("panda_finger_joint2")
        coupler_index = plant.AddCouplerConstraint(
            joint0=left_slider, joint1=right_slider, gear_ratio=-1.0, offset=0
        )
    # TODO: constraint for WSG

    # Get body
    gripper_body = plant.GetBodyByName(gripper_base_frame_name, gripper)

    # Weld gripper frame with arm
    if panda_model_instance is not None:
        plant.WeldFrames(
            plant.GetFrameByName("panda_link8", panda_model_instance),
            plant.GetFrameByName(gripper_base_frame_name, gripper), X_8G
        )
    return gripper, gripper_body
