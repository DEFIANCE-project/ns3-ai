import numpy as np
import gymnasium as gym
from gymnasium import spaces
import messages_pb2 as pb
import ns3ai_gym_msg_py as py_binding
from ns3ai_utils import Experiment
from .ns3_environment import Ns3Env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Any

class Ns3MultiAgentEnv(Ns3Env, MultiAgentEnv):

    def __init__(self, targetName, ns3Path, ns3Settings=None, shmSize=4096):
        self.action_space: dict[str, spaces.Space] = {}
        self.observation_space: dict[str, spaces.Space] = {}
        super().__init__(targetName, ns3Path, ns3Settings, shmSize)

    def initialize_env(self):
        simInitMsg = pb.MultiAgentSimInitMsg()
        self.msgInterface.PyRecvBegin()
        request = self.msgInterface.GetCpp2PyStruct().get_buffer()
        simInitMsg.ParseFromString(request)
        self.msgInterface.PyRecvEnd()

        self.action_space = self._create_space(simInitMsg.actSpaces)
        self.observation_space = self._create_space(simInitMsg.obsSpaces)

        reply = pb.SimInitAck()
        reply.done = True
        reply.stopSimReq = False
        reply_str = reply.SerializeToString()
        assert len(reply_str) <= py_binding.msg_buffer_size

        self.msgInterface.PySendBegin()
        self.msgInterface.GetPy2CppStruct().size = len(reply_str)
        self.msgInterface.GetPy2CppStruct().get_buffer_full()[:len(reply_str)] = reply_str
        self.msgInterface.PySendEnd()
        return True

    def rx_env_state(self):
        if self.newStateRx:
            return

        envStateMsg = pb.MultiAgentEnvStateMsg()
        self.msgInterface.PyRecvBegin()
        request = self.msgInterface.GetCpp2PyStruct().get_buffer()
        envStateMsg.ParseFromString(request)
        self.msgInterface.PyRecvEnd()

        self.obsData = self._create_data(envStateMsg.obsData)
        self.reward = envStateMsg.reward
        self.gameOver = envStateMsg.isGameOver
        self.gameOverReason = envStateMsg.reason
        self.agent_selection = envStateMsg.agentID

        if self.gameOver:
            self.send_close_command()

        self.extraInfo = dict(envStateMsg.info)

        self.newStateRx = True

    def wrap(self, data) -> dict[str, Any]:
        return {self.agent_selection: data}

    def get_obs(self):
        return self.wrap(super().get_obs())

    def get_reward(self):
        return self.wrap(super().get_reward())

    def is_game_over(self):
        return self.wrap(super().is_game_over())

    def get_extra_info(self):
        return self.wrap(super().get_extra_info())

    def send_actions(self, actions):
        reply = pb.EnvActMsg()

        actionMsg = self._pack_data(actions, self.action_space[self.agent_selection])
        reply.actData.CopyFrom(actionMsg)

        replyMsg = reply.SerializeToString()
        assert len(replyMsg) <= py_binding.msg_buffer_size
        self.msgInterface.PySendBegin()
        self.msgInterface.GetPy2CppStruct().size = len(replyMsg)
        self.msgInterface.GetPy2CppStruct().get_buffer_full()[:len(replyMsg)] = replyMsg
        self.msgInterface.PySendEnd()
        self.newStateRx = False
        return True

    def get_random_action(self):
        act = self.action_space[self.agent_selection].sample()
        return act
