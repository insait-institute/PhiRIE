"""Private, session-owned openpi server. Run with the configured openpi interpreter."""
import argparse
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--policy-id', required=True)
    parser.add_argument('--token', required=True)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    from openpi.training.config import get_config
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    import jax
    logging.basicConfig(level=logging.INFO)
    devices = jax.devices()
    if not any(device.platform == 'gpu' for device in devices):
        raise RuntimeError('PhiView policy inference requires the allocated GPU')
    policy = create_trained_policy(get_config(args.config), args.checkpoint)
    metadata = dict(policy.metadata)
    metadata.update(phiview_session=args.token, policy_id=args.policy_id,
                    checkpoint=args.checkpoint, training_config=args.config,
                    gpu=[device.device_kind for device in devices],
                    action_convention='absolute_joint_position')
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
