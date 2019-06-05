from network.baseline_network import tf_baseline_network


def build_baseline_network(args, session, name_scope,
                           observation_size, gnn_placeholder_list,
                           obs_placeholder):
    assert not args.use_gnn_as_value

    baseline_network = tf_baseline_network(
        session=session,
        name_scope=name_scope + '_baseline',
        input_size=observation_size,
        ob_placeholder=obs_placeholder,
        args=args
    )
    raw_obs_placeholder = baseline_network.get_ob_placeholder()

    target_return_placeholder = \
        baseline_network.get_target_return_placeholder()

    return baseline_network, target_return_placeholder, raw_obs_placeholder
