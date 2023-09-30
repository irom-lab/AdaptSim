from adaptsim.env.push_env import PushEnv


class PushOverlapEnv(PushEnv):
    """
    Dynamic pushing environment in Drake
    """

    def __init__(
        self,
        dt=0.01,
        render=False,
        visualize_contact=False,
        hand_type='plate',
        diff_ik_filter_hz=200,
        contact_solver='tamsi',
        panda_joint_damping=1.0,
        table_type='overlap',
        flag_disable_rate_limiter=True,
        use_goal=True,
        max_dist_bottle_goal=0.2,
        **kwargs  # for init_range rn
    ):
        super(PushOverlapEnv, self).__init__(
            dt=dt,
            render=render,
            visualize_contact=visualize_contact,
            hand_type=hand_type,
            diff_ik_filter_hz=diff_ik_filter_hz,
            contact_solver=contact_solver,
            panda_joint_damping=panda_joint_damping,
            table_type=table_type,
            flag_disable_rate_limiter=flag_disable_rate_limiter,
            use_goal=use_goal,
            max_dist_bottle_goal=max_dist_bottle_goal,
        )

    @property
    def parameter(self):
        return [
            self.task.obj_mu,
            self.task.obj_modulus,
            self.task.overlap_mu,
            self.task.overlap_y,
        ]

    def reset(self, task=None):
        """
        Call parent to reset arm and gripper positions (build if first-time). Reset veggies and task. Do not initialize simulator.
        """
        obs = super().reset(task)

        # Get context
        context = self.simulator.get_mutable_context()
        plant_context = self.plant.GetMyContextFromRoot(context)
        sg_context = self.sg.GetMyMutableContextFromRoot(context)
        query_object = self.sg.get_query_output_port().Eval(sg_context)
        context_inspector = query_object.inspector()

        # Get overlap body
        overlap_body_index = self.plant.GetBodyIndices(self.table_model_index
                                                      )[1]  # assume not tri
        overlap_body = self.plant.get_body(overlap_body_index)

        # Change overlap location
        if 'overlap_y' not in task:
            task.overlap_y = 0
        old_geom_id = self.plant.GetCollisionGeometriesForBody(overlap_body)[0]
        frame_id = context_inspector.GetFrameId(old_geom_id)
        self.replace_body(
            context=sg_context,
            context_inspector=context_inspector,
            body=overlap_body,
            frame_id=frame_id,
            geom_type='box',
            x_dim=0.05,
            y_dim=0.05,
            z_dim=0.01,
            x=0.05,  # x=0.5+0.2 is origin, 0.5 from table, 0.2 from overlap; thus right now the patch is 0.70-0.80 in x
            y=task.overlap_y,
            z=0.01,
        )

        # Set overlap properties
        self.set_obj_dynamics(
            context_inspector,
            sg_context,
            overlap_body,
            hc_dissipation=1.0,
            sap_dissipation=0.1,
            mu=max(0.01, task.overlap_mu),
            hydro_modulus=7,
            hydro_resolution=0.1,  # does not matter
            compliance_type='rigid'
        )

        # Set bottle to overlap mu
        self.set_obj_dynamics(
            context_inspector,
            sg_context,
            self.bottle_body,
            hc_dissipation=1.0,
            sap_dissipation=0.1,
            mu=max(0.01, task.overlap_mu),
            hydro_modulus=max(3, task.obj_modulus),
            hydro_resolution=0.01,  # matters
            compliance_type='compliant'
        )

        return obs
