import numpy as np
from pydrake.all import (
    RigidTransform, RotationMatrix, GeometryInstance, Cylinder,
    MakePhongIllustrationProperties, PiecewisePose, CoulombFriction,
    RoleAssign, AddCompliantHydroelasticProperties,
    AddRigidHydroelasticProperties
)


def add_property(proximity_properties, group, name, value):
    if proximity_properties.HasProperty(group, name):
        proximity_properties.UpdateProperty(group, name, value)
    else:
        proximity_properties.AddProperty(group, name, value)


def set_collision_properties(
    scene_graph,
    sg_context,
    context_inspector,
    plant,
    geometry_id,
    hc_dissipation,
    sap_dissipation,
    static_friction,
    dynamic_friction,
    hydro_modulus,
    hydro_resolution,
    compliance_type,
):
    """For all properties, update only specified, otherwise keep the existing one after retrieving the current properties."""

    # Get current properties
    proximity_properties = context_inspector.GetProximityProperties(
        geometry_id
    )

    # Update hc/sap dissipation and friction coefficient if specified
    if hc_dissipation is not None:
        add_property(
            proximity_properties, 'material', 'hunt_crossley_dissipation',
            hc_dissipation
        )
    if sap_dissipation is not None:
        add_property(
            proximity_properties, 'material', 'relaxation_time',
            sap_dissipation
        )
    if static_friction is not None and dynamic_friction is not None:
        add_property(
            proximity_properties, 'material', 'coulomb_friction',
            CoulombFriction(static_friction, dynamic_friction)
        )

    # Hydro - remove first since cannot replace - HydroelasticType not exposed
    if compliance_type is not None:
        hydro_property_list = [
            'compliance_type', 'hydroelastic_modulus', 'resolution_hint'
        ]
        assert hydro_modulus, hydro_resolution
        for hydro_property in hydro_property_list:
            if proximity_properties.HasProperty(
                'hydroelastic', hydro_property
            ):
                proximity_properties.RemoveProperty(
                    'hydroelastic', hydro_property
                )
        if compliance_type == 'rigid':
            AddRigidHydroelasticProperties(
                resolution_hint=hydro_resolution,
                properties=proximity_properties
            )
        elif compliance_type == 'compliant':
            AddCompliantHydroelasticProperties(
                resolution_hint=hydro_resolution,
                hydroelastic_modulus=10**hydro_modulus,
                properties=proximity_properties
            )

    # Replace
    scene_graph.AssignRole(
        context=sg_context, source_id=plant.get_source_id(),
        geometry_id=geometry_id, properties=proximity_properties,
        assign=RoleAssign.kReplace
    )


def get_new_trajectory(poses, times):
    traj = PiecewisePose.MakeLinear(times, poses)
    # traj = PiecewisePose.MakeCubicLinearWithEndLinearVelocity(times, poses)
    traj_V_G = traj.MakeDerivative()
    return traj, traj_V_G


def AddMultibodyTriad(frame, scene_graph, length=.25, radius=0.01, opacity=1.):
    plant = frame.GetParentPlant()
    AddTriad(
        plant.get_source_id(),
        plant.GetBodyFrameIdOrThrow(frame.body().index()), scene_graph, length,
        radius, opacity, frame.GetFixedPoseInBodyFrame()
    )


def AddTriad(
    source_id,
    frame_id,
    scene_graph,
    length=.25,
    radius=0.01,
    opacity=1.,
    X_FT=RigidTransform(),
    name="frame",
):
    """
    Adds illustration geometry representing the coordinate frame, with the
    x-axis drawn in red, the y-axis in green and the z-axis in blue. The axes
    point in +x, +y and +z directions, respectively.

    Args:
      source_id: The source registered with SceneGraph.
      frame_id: A geometry::frame_id registered with scene_graph.
      scene_graph: The SceneGraph with which we will register the geometry.
      length: the length of each axis in meters.
      radius: the radius of each axis in meters.
      opacity: the opacity of the coordinate axes, between 0 and 1.
      X_FT: a RigidTransform from the triad frame T to the frame_id frame F
      name: the added geometry will have names name + " x-axis", etc.
    """
    # x-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeYRotation(np.pi / 2), [length / 2., 0, 0]
    )
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " x-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([1, 0, 0, opacity])
    )
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # y-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2., 0]
    )
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " y-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 1, 0, opacity])
    )
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " z-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 0, 1, opacity])
    )
    scene_graph.RegisterGeometry(source_id, frame_id, geom)
