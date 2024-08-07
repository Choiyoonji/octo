import tensorflow as tf
traj = {'gripper_closedness_action': <tf.Tensor 'args_3:0' shape=(None,) dtype=float32>, 'terminate_episode': <tf.Tensor 'args_5:0' shape=(None,) dtype=float32>, 'world_vector': <tf.Tensor 'args_6:0' shape=(None, 3) dtype=float32>, 'rotation_delta': <tf.Tensor 'args_4:0' shape=(None, 3) dtype=float32>}

print(tf.shape(traj))