from typing import TYPE_CHECKING, Any, Literal, TypeVar

import messages_pb2 as pb
import ns3ai_gym_msg_py as py_binding

# if TYPE_CHECKING:
#     from gymnasium import spaces
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .ns3_environment import Ns3Env

T = TypeVar("T")


class Ns3MultiAgentEnv(Ns3Env, MultiAgentEnv):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._action_space: dict[str, spaces.Space] = {}
        self._observation_space: dict[str, spaces.Space] = {}
        self.agent_selection: str | None = None
        super().__init__(*args, **kwargs)

    def initialize_env(self) -> Literal[True]:
        init_msg = pb.MultiAgentSimInitMsg()
        self.msgInterface.PyRecvBegin()
        request = self.msgInterface.GetCpp2PyStruct().get_buffer()
        init_msg.ParseFromString(request)
        self.msgInterface.PyRecvEnd()

        for agent, space in init_msg.actSpaces.items():
            self._action_space[agent] = self._create_space(space)
        self.action_space = spaces.Dict(self._action_space)
        print(self.action_space)

        for agent, space in init_msg.obsSpaces.items():
            self._observation_space[agent] = self._create_space(space)
        self.observation_space = spaces.Dict(self._observation_space)
        print(self.observation_space)

        reply = pb.SimInitAck()
        reply.done = True
        reply.stopSimReq = False
        reply_str = reply.SerializeToString()
        assert len(reply_str) <= py_binding.msg_buffer_size

        self.msgInterface.PySendBegin()
        self.msgInterface.GetPy2CppStruct().size = len(reply_str)
        self.msgInterface.GetPy2CppStruct().get_buffer_full()[: len(reply_str)] = (
            reply_str
        )
        self.msgInterface.PySendEnd()
        return True

    def rx_env_state(self) -> None:
        if self.newStateRx:
            return

        state_msg = pb.MultiAgentEnvStateMsg()
        self.msgInterface.PyRecvBegin()
        request = self.msgInterface.GetCpp2PyStruct().get_buffer()
        state_msg.ParseFromString(request)
        self.msgInterface.PyRecvEnd()

        self.obsData = self._create_data(state_msg.obsData)
        self.reward = state_msg.reward
        self.gameOver = state_msg.isGameOver
        self.gameOverReason = state_msg.reason
        self.agent_selection = state_msg.agentID

        if self.gameOver:
            self.send_close_command()

        self.extraInfo = dict(state_msg.info)

        self.newStateRx = True

    def send_actions(self, actions: dict[str, Any]) -> bool:
        reply = pb.EnvActMsg()

        action_msg = self._pack_data(
            actions[self.agent_selection], self.action_space[self.agent_selection]
        )
        reply.actData.CopyFrom(action_msg)

        reply_msg = reply.SerializeToString()
        assert len(reply_msg) <= py_binding.msg_buffer_size
        self.msgInterface.PySendBegin()
        self.msgInterface.GetPy2CppStruct().size = len(reply_msg)
        self.msgInterface.GetPy2CppStruct().get_buffer_full()[: len(reply_msg)] = (
            reply_msg
        )
        self.msgInterface.PySendEnd()
        self.newStateRx = False
        return True

    def wrap(self, data: T) -> dict[str, T]:
        return {self.agent_selection: data}

    def step(self, actions: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        return tuple(self.wrap(state) for state in super().step(actions))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        return tuple(self.wrap(state) for state in super().reset(seed, options))

    def get_random_action(self) -> Any:
        return self.action_space[self.agent_selection].sample()
