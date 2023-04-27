import os

import numpy as np
import tensorflow as tf


def make_tfrecord():
    with tf.io.TFRecordWriter(
        f"{root}/{folder}/dataset/{filename.replace('.npz', '')}.tfrecord"
    ) as writer:
        with np.load(f"{root}/{folder}/trajectories/{filename}") as file:
            agent_obs = np.uint8(file["agent_obs"])
            human_obs = np.uint8(file["human_obs"])
            ram_obs = np.uint8(file["ram_obs"])

            q_values = file["q_values"]
            actions = np.int32(file["actions"])
            rewards = file["rewards"]

            assert len(agent_obs) == len(human_obs) == len(ram_obs)
            assert len(q_values) == len(actions) == len(rewards)

        for index in range(len(agent_obs)):
            data = {
                "agent_obs": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.io.serialize_tensor(agent_obs[index]).numpy()]
                    )
                ),
                "human_obs": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.io.serialize_tensor(human_obs[index]).numpy()]
                    )
                ),
                "ram_obs": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.io.serialize_tensor(ram_obs[index]).numpy()]
                    )
                ),
                "q_values": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[tf.io.serialize_tensor(q_values[index]).numpy()]
                    )
                ),
                "actions": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[actions[index]])
                ),
                "rewards": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[rewards[index]])
                ),
            }
            features = tf.train.Features(feature=data)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    for root in ["training", "testing"]:
        for folder in os.listdir(root):
            print(f"{root} {folder=}")
            for filename in sorted(os.listdir(f"{root}/{folder}/trajectories")):
                print(f"\t{filename=}")
                if not os.path.exists(f"{root}/{folder}/dataset"):
                    os.mkdir(f"{root}/{folder}/dataset")

                if "trajectory-" in filename and ".npz" in filename:
                    if not os.path.exists(
                        f"{root}/{folder}/dataset/{filename.replace('.npz', '')}.tfrecord"
                    ):
                        make_tfrecord()
